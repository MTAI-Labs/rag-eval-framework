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
