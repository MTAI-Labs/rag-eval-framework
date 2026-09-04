"""Regression diff direction, and that the HTML report is self-contained."""

from rag_eval.reporting import compare, render
from rag_eval.reporting.diff import per_question_changes


def scorecard(run_id: str, **overrides) -> dict:
    card = {
        "run_id": run_id,
        "adapter": "nvidia",
        "retrieval": {"hit_rate@5": 0.80, "recall@5": 0.60, "mrr": 0.70,
                      "page_citation_accuracy": 0.50},
        "generation": {"faithfulness": 4.0, "correctness": 4.0, "completeness": 4.0,
                       "hallucination_rate": 0.10},
        "ops": {"error_rate": 0.02},
        "headline": [],
        "per_question": [],
    }
    for block, values in overrides.items():
        card[block] = {**card[block], **values}
    return card


def test_higher_is_better_metrics():
    diff = compare(scorecard("b", retrieval={"hit_rate@5": 0.90}), scorecard("a"))
    row = next(r for r in diff["metrics"] if r["metric"] == "hit_rate@5")

    assert row["verdict"] == "improved"
    assert round(row["delta"], 4) == 0.10
    assert row["pct_change"] == 12.5


def test_lower_is_better_metrics_invert_the_verdict():
    # More hallucination is a regression even though the number went up.
    diff = compare(scorecard("b", generation={"hallucination_rate": 0.25}), scorecard("a"))
    row = next(r for r in diff["metrics"] if r["metric"] == "hallucination_rate")

    assert row["verdict"] == "regressed"
    assert row["delta"] > 0


def test_identical_runs_are_flat():
    diff = compare(scorecard("b"), scorecard("a"))
    assert diff["regressed"] == []
    assert diff["improved"] == []
    assert diff["summary"].get("flat")


def test_a_first_run_has_no_baseline():
    diff = compare(scorecard("a"), None)
    assert diff["has_baseline"] is False
    assert all(r["verdict"] in ("new", "missing") for r in diff["metrics"])


def test_missing_metrics_are_reported_not_treated_as_zero():
    current = scorecard("b")
    current["generation"] = {}
    diff = compare(current, scorecard("a"))
    row = next(r for r in diff["metrics"] if r["metric"] == "faithfulness")
    assert row["verdict"] == "missing"


def test_per_question_changes_point_at_the_rows_that_moved():
    baseline = scorecard("a")
    baseline["per_question"] = [
        {"question_id": "q1", "retrieval": {"hit_at_k": {"5": True}}, "scores": {"correctness": 5}},
        {"question_id": "q2", "retrieval": {"hit_at_k": {"5": True}}, "scores": {"correctness": 4}},
    ]
    current = scorecard("b")
    current["per_question"] = [
        {"question_id": "q1", "question": "Q1?", "retrieval": {"hit_at_k": {"5": False}},
         "scores": {"correctness": 2}},
        {"question_id": "q2", "question": "Q2?", "retrieval": {"hit_at_k": {"5": True}},
         "scores": {"correctness": 4}},
    ]

    changes = per_question_changes(current, baseline)
    assert [c["question_id"] for c in changes] == ["q1"]
    assert changes[0]["direction"] == "regressed"
    assert changes[0]["hit@5"] == {"baseline": True, "current": False}


def test_html_report_is_self_contained():
    card = scorecard("b")
    card["headline"] = [
        {"block": "retrieval", "metric": "hit_rate@5", "value": 0.8, "better": "higher"}
    ]
    card["warnings"] = ["7 questions await human adjudication"]
    card["per_question"] = [
        {"question_id": "q1", "question": "Who resigned?", "gold_sitting": "dr_2026-06-22",
         "gold_pages": [3], "generated_answer": "Two members.",
         "retrieval": {"hit_at_k": {"5": True}, "citation_page_match": True},
         "scores": {"faithfulness": 5}, "flagged": [], "agreement": {}, "rationales": {}}
    ]
    html = render(card, compare(card, scorecard("a")), {"framework_version": "0.1.0"})

    assert "<!doctype html>" in html
    assert "http://" not in html and "https://" not in html  # no CDN, opens offline
    assert "hit_rate@5" in html
    assert "7 questions await human adjudication" in html
    assert "Who resigned?" in html


def test_html_escapes_content_from_the_rag_service():
    card = scorecard("b")
    card["per_question"] = [
        {"question_id": "q1", "question": "<script>alert(1)</script>",
         "gold_sitting": "dr_2026-06-22", "gold_pages": [3],
         "generated_answer": "<img onerror=x>", "retrieval": {}, "scores": {},
         "flagged": [], "agreement": {}, "rationales": {}}
    ]
    html = render(card)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


# -- environment drift: are these two runs even the same system? -----------
def env_card(run_id: str, **env) -> dict:
    card = scorecard(run_id)
    card["rag_environment"] = {
        "retrieval_mode": "dense",
        "embedding_model": "nvidia/llama-nemotron-embed-1b-v2",
        "collection": "parliament-hansard-eval",
        **env,
    }
    return card


def test_a_dense_to_hybrid_change_makes_the_diff_incomparable():
    # Same vectors, different retrieval path. Reporting the delta as an
    # improvement is the easiest way for this framework to mislead someone.
    diff = compare(env_card("b", retrieval_mode="hybrid"), env_card("a"))

    assert diff["comparable"] is False
    assert diff["environment_drift"]["differences"] == [
        {"field": "retrieval_mode", "baseline": "dense", "current": "hybrid"}
    ]


def test_a_changed_embedding_model_makes_the_diff_incomparable():
    diff = compare(env_card("b", embedding_model="nvidia/llama-3.2-nv-embedqa-1b-v2"),
                   env_card("a"))
    assert diff["comparable"] is False
    assert diff["environment_drift"]["differences"][0]["field"] == "embedding_model"


def test_the_deltas_are_still_computed_when_incomparable():
    # You want to see the numbers; you must not read them as a verdict.
    current = env_card("b", retrieval_mode="hybrid")
    current["retrieval"]["hit_rate@5"] = 0.95
    diff = compare(current, env_card("a"))

    row = next(r for r in diff["metrics"] if r["metric"] == "hit_rate@5")
    assert row["delta"] is not None
    assert diff["comparable"] is False


def test_the_same_stack_stays_comparable():
    diff = compare(env_card("b"), env_card("a"))
    assert diff["comparable"] is True
    assert diff["environment_drift"]["differences"] == []


def test_a_moved_host_is_noted_but_not_invalidating():
    # Same collection, model and retrieval path served from another box is
    # still the same system.
    diff = compare(env_card("b", base_url="http://rpgpu127:8081"),
                   env_card("a", base_url="http://hgpu122:8081"))
    assert diff["comparable"] is True
    assert diff["environment_drift"]["notes"][0]["field"] == "base_url"


def test_runs_without_environment_data_are_not_falsely_flagged():
    # Older runs predate the probe; absence of evidence is not drift.
    diff = compare(scorecard("b"), scorecard("a"))
    assert diff["comparable"] is True
    assert diff["environment_drift"]["checked"] is False


def test_the_report_leads_with_incomparability():
    current = env_card("b", retrieval_mode="hybrid")
    current["headline"] = [
        {"block": "retrieval", "metric": "hit_rate@5", "value": 0.95, "better": "higher"}
    ]
    html = render(current, compare(current, env_card("a")))

    assert "not the same system" in html
    # The banner must precede the headline numbers a reader sees first.
    assert html.index("not the same system") < html.index("hit_rate@5")
