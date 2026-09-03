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
