"""Generation metrics over panel verdicts, including adjudication overrides."""

from conftest import trace

from rag_eval.judges.rubric import JudgeVerdict, PanelVerdict
from rag_eval.metrics import generation


def verdict(qid: str, scores: dict[str, int], *, flagged: list[str] | None = None,
            judges: list[dict[str, int]] | None = None) -> PanelVerdict:
    return PanelVerdict(
        question_id=qid,
        scores=scores,
        flagged_dimensions=flagged or [],
        verdicts=[
            JudgeVerdict(model=f"judge-{i}", scores=s)
            for i, s in enumerate(judges or [scores, scores, scores], 1)
        ],
    )


BASE = {"faithfulness": 5, "correctness": 4, "completeness": 4, "citation_accuracy": 3}


def test_dimension_means_and_pass_rates():
    verdicts = [verdict("q1", BASE), verdict("q2", {**BASE, "correctness": 2})]
    metrics = generation.aggregate(verdicts, pass_threshold=4)

    assert metrics["faithfulness"] == 5.0
    assert metrics["correctness"] == 3.0
    assert metrics["correctness_pass_rate"] == 0.5


def test_hallucination_rate_counts_faithfulness_below_the_threshold():
    verdicts = [
        verdict("q1", {**BASE, "faithfulness": 5}),
        verdict("q2", {**BASE, "faithfulness": 2}),
        verdict("q3", {**BASE, "faithfulness": 3}),
        verdict("q4", {**BASE, "faithfulness": 1}),
    ]
    # Strictly below 3: two of four.
    assert generation.aggregate(verdicts, hallucination_threshold=3)["hallucination_rate"] == 0.5


def test_human_adjudication_overrides_the_panel():
    v = verdict("q1", {**BASE, "correctness": 2}, flagged=["correctness"])
    v.human_scores = {"correctness": 5}

    metrics = generation.aggregate([v])
    assert metrics["correctness"] == 5.0
    assert metrics["panel_disagreement"]["adjudicated"] == 1
    assert metrics["panel_disagreement"]["awaiting_human_review"] == 0


def test_unadjudicated_splits_are_surfaced_not_hidden():
    verdicts = [verdict("q1", BASE), verdict("q2", BASE, flagged=["faithfulness", "correctness"])]
    disagreement = generation.aggregate(verdicts)["panel_disagreement"]

    assert disagreement["flagged_questions"] == 1
    assert disagreement["awaiting_human_review"] == 1
    assert disagreement["by_dimension"]["faithfulness"] == 1
    assert disagreement["by_dimension"]["completeness"] == 0


def test_unjudged_questions_are_counted_separately():
    failed = PanelVerdict(question_id="q9", error="adapter error, not judged: boom")
    metrics = generation.aggregate([verdict("q1", BASE), failed])

    assert metrics["questions"] == 2
    assert metrics["judged"] == 1
    assert metrics["unjudged"] == 1
    assert metrics["faithfulness"] == 5.0  # the failed row does not dilute the mean


def test_empty_answer_rate_uses_the_traces():
    verdicts = [verdict("q1", BASE)]
    traces = [trace("q1", [], answer="something"), trace("q2", [], error="timeout")]
    assert generation.aggregate(verdicts, traces)["empty_answer_rate"] == 0.5


def test_per_judge_scores_expose_a_drifting_panel_member():
    verdicts = [
        verdict("q1", BASE, judges=[
            {"faithfulness": 5, "correctness": 5, "completeness": 5, "citation_accuracy": 5},
            {"faithfulness": 5, "correctness": 5, "completeness": 5, "citation_accuracy": 5},
            {"faithfulness": 1, "correctness": 1, "completeness": 1, "citation_accuracy": 1},
        ])
    ]
    stats = generation.per_judge_scores(verdicts)
    assert stats["judge-1"]["mean_scores"]["faithfulness"] == 5.0
    assert stats["judge-3"]["mean_scores"]["faithfulness"] == 1.0
    assert stats["judge-3"]["failure_rate"] == 0.0
