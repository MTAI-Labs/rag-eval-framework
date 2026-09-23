"""The judge panel: parsing, majority vote, disagreement flags and retries."""

import json

import pytest
from conftest import chunk, item, ref, trace

from rag_eval.config import JudgeConfig, JudgeModel
from rag_eval.judges.client import ScriptedChatClient
from rag_eval.judges.panel import JudgePanel, JudgeParseError, flagged_rows, parse_verdict
from rag_eval.judges.rubric import DIMENSIONS, majority_vote


def scores(f=5, c=5, comp=5, cite=5) -> str:
    return json.dumps({
        "scores": {"faithfulness": f, "correctness": c, "completeness": comp,
                   "citation_accuracy": cite},
        "rationale": "because",
    })


def panel(responses: dict[str, str], **config_kwargs) -> JudgePanel:
    config = JudgeConfig(
        models=[JudgeModel(id="a"), JudgeModel(id="b"), JudgeModel(id="c")], **config_kwargs
    )
    return JudgePanel(ScriptedChatClient(responses), config, max_workers=1)


GOLD = item("q1", ref("dr", "2026-06-22", 3))
TRACE = trace("q1", [chunk(1, "dr_2026-06-22", 3)], cited=[ref("dr", "2026-06-22", 3)])


# -- parsing ---------------------------------------------------------------
def test_parses_a_clean_rubric_object():
    verdict = parse_verdict(scores(5, 4, 3, 2), "a")
    assert verdict.scores == {"faithfulness": 5, "correctness": 4,
                              "completeness": 3, "citation_accuracy": 2}
    assert verdict.ok


def test_parses_a_flat_object_and_a_fenced_one():
    flat = json.dumps({d: 4 for d in DIMENSIONS})
    assert parse_verdict(flat, "a").scores["faithfulness"] == 4
    assert parse_verdict(f"Here you go:\n```json\n{scores()}\n```", "a").ok


def test_out_of_range_scores_are_clamped_to_the_rubric_scale():
    assert parse_verdict(scores(9, 0, 5, 5), "a").scores["faithfulness"] == 5
    assert parse_verdict(scores(9, 0, 5, 5), "a").scores["correctness"] == 1


def test_a_missing_dimension_is_a_parse_error_not_a_silent_zero():
    partial = json.dumps({"scores": {"faithfulness": 5}})
    with pytest.raises(JudgeParseError, match="missing"):
        parse_verdict(partial, "a")


def test_prose_without_json_is_a_parse_error():
    with pytest.raises(JudgeParseError):
        parse_verdict("The answer looks pretty good to me.", "a")


# -- voting ----------------------------------------------------------------
@pytest.mark.parametrize(
    "votes,expected,agreement",
    [
        ((4, 4, 4), 4, "unanimous"),
        ((4, 4, 5), 4, "majority"),   # adjacent disagreement: the panel agrees enough
        ((5, 5, 2), 5, "split"),      # a majority, but a dissenter two points away
        ((1, 3, 5), 3, "split"),      # no majority at all -> provisional median
    ],
)
def test_majority_vote(votes, expected, agreement):
    assert majority_vote(votes, spread_threshold=2) == (expected, agreement)


def test_unanimous_panel_flags_nothing():
    verdict = panel({"a": scores(), "b": scores(), "c": scores()}).judge_one(GOLD, TRACE)
    assert verdict.scores == {d: 5 for d in DIMENSIONS}
    assert verdict.flagged_dimensions == []
    assert not verdict.needs_human_review


def test_a_split_flags_only_the_dimension_that_split():
    verdict = panel({
        "a": scores(5, 5, 5, 5),
        "b": scores(5, 1, 5, 5),
        "c": scores(5, 3, 5, 5),
    }).judge_one(GOLD, TRACE)

    assert verdict.flagged_dimensions == ["correctness"]
    assert verdict.agreement["correctness"] == "split"
    assert verdict.agreement["faithfulness"] == "unanimous"
    assert verdict.scores["correctness"] == 3  # provisional median, pending a human
    assert verdict.needs_human_review


def test_flagged_rows_lists_what_a_human_still_owes():
    good = panel({"a": scores(), "b": scores(), "c": scores()}).judge_one(GOLD, TRACE)
    bad = panel({"a": scores(1), "b": scores(3), "c": scores(5)}).judge_one(GOLD, TRACE)
    assert [v.question_id for v in flagged_rows([good, bad])] == ["q1"]


# -- failure handling ------------------------------------------------------
def test_one_failing_judge_still_yields_a_verdict_from_the_other_two():
    p = panel({"a": scores(4), "b": scores(4), "c": "not json at all"})
    verdict = p.judge_one(GOLD, TRACE)

    assert verdict.scores["faithfulness"] == 4
    assert "1 of 3 judges failed" in (verdict.error or "")
    assert p.usage.failed_calls == 3  # the bad judge is retried, then given up on


def test_all_judges_failing_produces_an_unjudged_row_not_a_crash():
    verdict = panel({"a": "no", "b": "no", "c": "no"}).judge_one(GOLD, TRACE)
    assert verdict.scores == {}
    assert verdict.needs_human_review
    assert "every judge failed" in (verdict.error or "")


def test_an_adapter_error_is_not_sent_to_the_panel():
    # Judging a failed RAG call would spend three model calls to learn nothing.
    failed = trace("q1", [], error="HTTP 503")
    p = panel({"a": scores(), "b": scores(), "c": scores()})
    verdict = p.judge_one(GOLD, failed)

    assert p.usage.calls == 0
    assert verdict.flagged_dimensions == list(DIMENSIONS)
    assert "not judged" in (verdict.error or "")


def test_usage_accounting_tracks_cost_per_model():
    config = JudgeConfig(models=[
        JudgeModel(id="a", input_cost_per_1k=1.0, output_cost_per_1k=2.0)
    ])
    p = JudgePanel(ScriptedChatClient({"a": scores()}), config, max_workers=1)
    p.judge_one(GOLD, TRACE)

    assert p.usage.calls == 1
    assert p.usage.cost_usd > 0


def test_the_whole_panel_sees_the_same_prompt():
    client = ScriptedChatClient({"a": scores(), "b": scores(), "c": scores()})
    JudgePanel(client, JudgeConfig(models=[JudgeModel(id=m) for m in "abc"]),
               max_workers=1).judge_one(GOLD, TRACE)

    prompts = {json.dumps(messages) for _, messages in client.calls}
    assert len(prompts) == 1, "a disagreement must be about the models, not the wording"


def test_the_prompt_carries_the_golden_reference_and_the_retrieved_context():
    client = ScriptedChatClient({}, default=scores())
    JudgePanel(client, JudgeConfig(models=[JudgeModel(id="a")]), max_workers=1).judge_one(
        GOLD, TRACE
    )
    prompt = client.calls[0][1][1]["content"]
    assert "dr_2026-06-22" in prompt
    assert GOLD.expected_answer in prompt


# -- calibration sampling and inter-judge agreement ------------------------
def test_stratified_sample_spreads_across_owners_and_sittings():
    from rag_eval.judges.calibration import sample_coverage, stratified_sample
    from conftest import item as mk

    items = []
    for owner, sitting in (("Sofia", ("dr", "2026-06-22")), ("Amirah", ("dn", "2026-08-04")),
                           ("Syahir", ("kkdr", "2026-07-14"))):
        for i in range(20):
            g = mk(f"{owner}-{i}", ref(*sitting, 1))
            g.owner = owner
            items.append(g)

    chosen = stratified_sample(items, 9)
    cov = sample_coverage(chosen)
    # A sample drawn from one reviewer would calibrate against that reviewer's
    # habits rather than the rubric.
    assert set(cov["owners"]) == {"Sofia", "Amirah", "Syahir"}
    assert all(n == 3 for n in cov["owners"].values())
    assert len(cov["sittings"]) == 3


def test_stratified_sample_is_reproducible():
    from rag_eval.judges.calibration import stratified_sample
    from conftest import item as mk
    items = [mk(f"q{i}", ref("dr", "2026-06-22", 1)) for i in range(30)]
    assert [i.id for i in stratified_sample(items, 10, seed=7)] == \
           [i.id for i in stratified_sample(items, 10, seed=7)]


def test_stratified_sample_handles_a_stratum_smaller_than_its_share():
    from rag_eval.judges.calibration import sample_coverage, stratified_sample
    from conftest import item as mk
    items = []
    for i in range(20):
        g = mk(f"big-{i}", ref("dr", "2026-06-22", 1)); g.owner = "Sofia"; items.append(g)
    g = mk("tiny-0", ref("kr", "2026-07-14", 1)); g.owner = "Syahir"; items.append(g)

    cov = sample_coverage(stratified_sample(items, 10))
    assert cov["owners"]["Syahir"] == 1      # represented, not swamped
    assert cov["size"] == 10


def test_inter_judge_agreement_is_measured_without_human_labels():
    from rag_eval.judges.calibration import inter_judge_agreement
    unanimous = panel({"a": scores(4), "b": scores(4), "c": scores(4)}).judge_one(GOLD, TRACE)
    split = panel({"a": scores(5), "b": scores(3), "c": scores(1)}).judge_one(GOLD, TRACE)

    report = inter_judge_agreement([unanimous, split])
    assert report["questions"] == 2
    assert report["flagged_questions"] == 1
    assert report["per_dimension"]["faithfulness"]["pairs_compared"] == 6   # 3 pairs x 2 rows
    assert 0.0 < report["per_dimension"]["faithfulness"]["exact_agreement"] < 1.0
    assert report["per_pair"]


def test_agreement_report_states_whether_it_meets_the_target():
    from rag_eval.judges.calibration import CalibrationSample, agreement_report
    v = panel({"a": scores(4), "b": scores(4), "c": scores(4)}).judge_one(GOLD, TRACE)
    perfect = CalibrationSample(question_id="q1", human_scores={d: 4 for d in DIMENSIONS})

    report = agreement_report([v], [perfect])
    assert report["panel_human_agreement"] == 1.0
    assert report["target"] == 0.8
    assert report["meets_target"] is True
    assert "inter_judge" in report


# -- CSV labelling round-trip ---------------------------------------------
def _sheet(**over):
    row = {"question_id": "tp-0001", "owner": "Sofia", "sitting_id": "dr_2026-06-22",
           "golden_ms": [3], "question": "Who resigned?", "expected_answer": "Pandan.",
           "generated_answer": "Pandan dan Setiawangsa.",
           "retrieved": [{"rank": 1, "sitting_id": "dr_2026-06-22", "page": 9, "text": "…"}],
           "panel_scores": {d: 4 for d in DIMENSIONS},
           "judge_scores": {"a": {d: 4 for d in DIMENSIONS}},
           "judge_rationales": {"a": "because"}, "flagged_dimensions": [],
           "human_scores": {d: None for d in DIMENSIONS},
           "labelled_by": "", "notes": "", "adapter_error": None}
    row.update(over)
    return {"rows": [row]}


def test_csv_puts_the_panel_scores_after_the_human_columns():
    from rag_eval.judges.calibration import CSV_COLUMNS
    # A labeller who sees the judges' answers first is anchored by them, which
    # defeats the point of an independent opinion.
    human_at = min(CSV_COLUMNS.index(f"human_{d}") for d in DIMENSIONS)
    panel_at = min(CSV_COLUMNS.index(f"panel_{d}") for d in DIMENSIONS)
    content_at = CSV_COLUMNS.index("generated_answer")
    assert content_at < human_at < panel_at


def test_sheet_flattens_to_csv_rows():
    from rag_eval.judges.calibration import sheet_to_rows
    row = sheet_to_rows(_sheet())[0]
    assert row["question_id"] == "tp-0001"
    assert row["golden_ms"] == "3"
    assert row["human_faithfulness"] == ""        # blank, awaiting a person
    assert row["panel_faithfulness"] == "4"
    assert "dr_2026-06-22 p.9" in row["retrieved_context"]


def test_filled_csv_rows_become_calibration_samples():
    from rag_eval.judges.calibration import rows_to_samples, sheet_to_rows
    row = sheet_to_rows(_sheet())[0]
    row.update({"human_faithfulness": "5", "human_correctness": "3",
                "human_completeness": "4", "human_citation_accuracy": "2"})
    samples = rows_to_samples([row], labelled_by="Sofia")

    assert len(samples) == 1
    assert samples[0].human_scores == {"faithfulness": 5, "correctness": 3,
                                       "completeness": 4, "citation_accuracy": 2}
    assert samples[0].labelled_by == "Sofia"


def test_an_unlabelled_row_is_skipped_not_recorded_as_blank():
    from rag_eval.judges.calibration import rows_to_samples, sheet_to_rows
    assert rows_to_samples(sheet_to_rows(_sheet())) == []


def test_a_partially_labelled_row_keeps_what_was_scored():
    from rag_eval.judges.calibration import rows_to_samples, sheet_to_rows
    row = sheet_to_rows(_sheet())[0]
    row["human_faithfulness"] = "5"        # only one dimension scored
    samples = rows_to_samples([row])
    assert samples[0].human_scores == {"faithfulness": 5}


def test_out_of_range_labels_are_clamped_to_the_rubric_scale():
    from rag_eval.judges.calibration import rows_to_samples, sheet_to_rows
    row = sheet_to_rows(_sheet())[0]
    row.update({"human_faithfulness": "9", "human_correctness": "0"})
    scores = rows_to_samples([row])[0].human_scores
    assert scores["faithfulness"] == 5 and scores["correctness"] == 1


# -- the labelling workbook ------------------------------------------------
def test_workbook_hides_the_panel_scores(tmp_path):
    from openpyxl import load_workbook
    from rag_eval.judges.workbook import write_workbook

    path, n = write_workbook(_sheet(), tmp_path / "s.xlsx")
    ws = load_workbook(path)["labelling"]
    headers = [c.value for c in ws[1]]
    hidden = {k for k, v in ws.column_dimensions.items() if v.hidden}
    from openpyxl.utils import get_column_letter
    pretty = {h.replace("_", " ") if h else h: i for i, h in enumerate(headers)}
    panel_cols = {get_column_letter(pretty[f"panel {d}".replace("_", " ")] + 1)
                  for d in DIMENSIONS}
    human_cols = {get_column_letter(pretty[f"human {d}".replace("_", " ")] + 1)
                  for d in DIMENSIONS}

    assert n == 1
    assert panel_cols <= hidden       # anchoring is the thing being prevented
    assert not (human_cols & hidden)  # the columns to fill must be visible


def test_workbook_assigns_labellers_round_robin(tmp_path):
    from openpyxl import load_workbook
    from rag_eval.judges.workbook import write_workbook

    sheet = {"rows": _sheet()["rows"] * 7}
    path, _ = write_workbook(sheet, tmp_path / "s.xlsx", labellers=["A", "B", "C"])
    ws = load_workbook(path)["labelling"]
    assigned = [ws.cell(r, 1).value for r in range(2, 9)]
    # Interleaved, not blocked: the sample is ordered by owner, so contiguous
    # blocks would give one labeller mostly one reviewer's questions.
    assert assigned == ["A", "B", "C", "A", "B", "C", "A"]


def test_a_workbook_exported_back_to_csv_still_imports(tmp_path):
    # Excel keeps the prettified headers ("human faithfulness"), so an importer
    # that only accepts underscores would silently find no labels at all.
    from rag_eval.judges.calibration import rows_to_samples
    spaced = {"question id": "tp-0001", "human faithfulness": "5",
              "human correctness": "4", "human completeness": "3",
              "human citation accuracy": "2", "labelled by": "Sofia"}
    samples = rows_to_samples([spaced])

    assert len(samples) == 1
    assert samples[0].question_id == "tp-0001"
    assert samples[0].human_scores == {"faithfulness": 5, "correctness": 4,
                                       "completeness": 3, "citation_accuracy": 2}


def test_workbook_carries_a_rubric_tab(tmp_path):
    from openpyxl import load_workbook
    from rag_eval.judges.workbook import write_workbook

    path, _ = write_workbook(_sheet(), tmp_path / "s.xlsx")
    wb = load_workbook(path)
    assert "how to label" in wb.sheetnames
    text = " ".join(str(c.value) for row in wb["how to label"].iter_rows()
                    for c in row if c.value)
    for dim in DIMENSIONS:
        assert dim in text


# -- calibration reports from a sheet ------------------------------------

def _verdict_sheet(**overrides):
    row = {
        "question_id": "tp-0001",
        "panel_scores": {"faithfulness": 5, "correctness": 5,
                         "completeness": 4, "citation_accuracy": 3},
        "judge_scores": {
            "judge-a": {"faithfulness": 5, "correctness": 5,
                        "completeness": 4, "citation_accuracy": 3},
            "judge-b": {"faithfulness": 5, "correctness": 5,
                        "completeness": 4, "citation_accuracy": 1},
        },
        "judge_rationales": {"judge-a": "cites the right page"},
        "flagged_dimensions": ["citation_accuracy"],
        "adapter_error": None,
    }
    row.update(overrides)
    return {"rows": [row]}


def test_a_sheet_rebuilds_the_panel_verdict_it_recorded():
    from rag_eval.judges.calibration import verdicts_from_sheet

    v = verdicts_from_sheet(_verdict_sheet())[0]
    assert v.question_id == "tp-0001"
    assert v.scores["citation_accuracy"] == 3
    assert v.flagged_dimensions == ["citation_accuracy"]
    assert {j.model for j in v.verdicts} == {"judge-a", "judge-b"}
    assert all(j.ok for j in v.verdicts)
    assert v.verdicts[0].rationale == "cites the right page"


def test_a_partly_judged_row_yields_only_the_judges_that_answered():
    # A row judged 2/3 must not grow a third verdict with invented scores --
    # that would let a missing judge quietly count towards agreement.
    from rag_eval.judges.calibration import verdicts_from_sheet

    sheet = _verdict_sheet(judge_scores={"judge-a": {"faithfulness": 5, "correctness": 5,
                                             "completeness": 4, "citation_accuracy": 3}})
    assert len(verdicts_from_sheet(sheet)[0].verdicts) == 1


def test_sheet_verdicts_feed_the_same_agreement_report_a_run_would():
    from rag_eval.judges.calibration import (CalibrationSample, agreement_report,
                                             verdicts_from_sheet)

    verdicts = verdicts_from_sheet(_verdict_sheet())
    human = CalibrationSample(
        question_id="tp-0001",
        human_scores={"faithfulness": 5, "correctness": 5,
                      "completeness": 4, "citation_accuracy": 1},
        labelled_by="tester",
    )
    report = agreement_report(verdicts, [human])

    assert report["labelled_samples"] == 1
    # three dimensions agree exactly; citation_accuracy is panel 3 vs human 1
    assert report["per_dimension"]["correctness"]["exact_agreement"] == 1.0
    assert report["per_dimension"]["citation_accuracy"]["exact_agreement"] == 0.0
    assert report["per_dimension"]["citation_accuracy"]["mean_bias"] == 2.0
    # the report carries both halves criterion 2 asks for
    assert report["inter_judge"]["per_dimension"]["citation_accuracy"]["pairs_compared"] == 1
    assert "panel_human_agreement" in report
