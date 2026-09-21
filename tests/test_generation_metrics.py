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


def test_a_run_with_no_judgements_reports_none_not_zero():
    # Nothing judged yet is not the same as everything scoring zero.
    metrics = generation.aggregate([])

    assert metrics["questions"] == 0
    assert metrics["judged"] == 0
    assert metrics["faithfulness"] is None
    assert metrics["hallucination_rate"] is None


def test_hallucination_rate_ignores_unjudged_questions():
    from rag_eval.judges.rubric import PanelVerdict
    verdicts = [verdict("q1", {**BASE, "faithfulness": 1}),
                PanelVerdict(question_id="q2", error="adapter error, not judged: boom")]
    metrics = generation.aggregate(verdicts)

    # 1 of 1 judged answers hallucinated — the unjudged one must not halve it.
    assert metrics["hallucination_rate"] == 1.0
    assert metrics["unjudged"] == 1


# -- answer quality: is the RAG answering, or thinking out loud? -----------
def test_reasoning_leakage_is_detected():
    from rag_eval.metrics.generation import looks_like_reasoning
    # A reasoning model with thinking left on emits its chain-of-thought as the
    # answer. Judges score that near-perfect on faithfulness because it
    # contradicts nothing, so nothing downstream catches it.
    assert looks_like_reasoning(
        "We need answer user's query using context only. Need follow instructions")
    assert looks_like_reasoning("Let me check the context for the resignation date.")
    assert looks_like_reasoning("Okay, the user is asking about PLKN.")


def test_a_real_answer_is_not_flagged():
    from rag_eval.metrics.generation import looks_like_reasoning
    # A false positive would wrongly discredit a working RAG, so detection is
    # deliberately conservative and only looks at how the answer opens.
    assert not looks_like_reasoning("Yang Berhormat bagi kawasan Pandan dan Setiawangsa.")
    assert not looks_like_reasoning("Jumlah keseluruhan setakat ini adalah 3,404 orang.")
    assert not looks_like_reasoning("")
    # mentions the phrase late, after genuinely answering
    assert not looks_like_reasoning(
        "Sebanyak 108 ladang beroperasi. " + "x" * 400 + " we need to note that")


def test_answer_quality_reports_the_leak_rate():
    from rag_eval.metrics.generation import answer_quality
    traces = [
        trace("q1", [], answer="Jumlah keseluruhan adalah 3,404 orang."),
        trace("q2", [], answer="We need answer user's query using context only."),
        trace("q3", [], answer="Let me check the retrieved context first."),
        trace("q4", [], error="HTTP 503"),
    ]
    q = answer_quality(traces)

    assert q["traces"] == 4
    assert q["answered"] == 3          # the errored trace is not an answer
    assert q["reasoning_leaked"] == 2
    assert q["reasoning_leak_rate"] == round(2 / 3, 4)
    assert q["examples"] == ["q2", "q3"]


def test_answer_quality_on_a_healthy_run_reports_zero():
    from rag_eval.metrics.generation import answer_quality
    q = answer_quality([trace("q1", [], answer="Tidak. MPOB tidak mengawal harga.")])
    assert q["reasoning_leaked"] == 0
    assert q["reasoning_leak_rate"] == 0.0
