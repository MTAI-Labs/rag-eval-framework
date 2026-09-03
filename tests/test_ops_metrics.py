"""Ops metrics: latency percentiles, error rate and cost estimation."""

from conftest import trace

from rag_eval.judges.panel import PanelUsage
from rag_eval.metrics import ops
from rag_eval.types import RagTrace


def timed(qid: str, latency: float) -> RagTrace:
    return trace(qid, [], latency_ms=latency)


def test_percentiles_and_error_rate():
    traces = [timed(f"q{i}", latency) for i, latency in enumerate([100, 200, 300, 400])]
    traces.append(trace("q5", [], error="timeout"))

    metrics = ops.aggregate(traces)
    assert metrics["questions"] == 5
    assert metrics["succeeded"] == 4
    assert metrics["error_rate"] == 0.2
    assert metrics["latency_ms"]["p50"] == 250.0
    assert metrics["latency_ms"]["max"] == 400.0


def test_failed_traces_do_not_pollute_latency():
    traces = [timed("q1", 100.0), trace("q2", [], latency_ms=99999.0, error="boom")]
    assert ops.aggregate(traces)["latency_ms"]["mean"] == 100.0
    assert ops.aggregate(traces)["latency_ms"]["measured"] == 1


def test_cost_is_none_when_the_service_reports_no_tokens():
    # Reporting an unknown cost as $0.00 would make a scorecard lie about spend.
    t = RagTrace(question_id="q1", question="?", adapter="mock", latency_ms=10.0)
    metrics = ops.aggregate([t], input_cost_per_1k=1.0, output_cost_per_1k=2.0)
    assert metrics["cost_usd"]["rag_total"] is None


def test_cost_uses_the_configured_rates():
    traces = [timed("q1", 100.0), timed("q2", 100.0)]  # 100 prompt + 20 completion each
    metrics = ops.aggregate(traces, input_cost_per_1k=1.0, output_cost_per_1k=2.0)
    # (200/1000 * 1.0) + (40/1000 * 2.0) = 0.28
    assert metrics["cost_usd"]["rag_total"] == 0.28
    assert metrics["cost_usd"]["rag_per_query"] == 0.14


def test_judge_usage_is_folded_into_the_total():
    usage = PanelUsage(calls=6, prompt_tokens=1000, completion_tokens=100, cost_usd=0.5)
    metrics = ops.aggregate([timed("q1", 100.0)], input_cost_per_1k=1.0,
                            output_cost_per_1k=1.0, judge_usage=usage)
    assert metrics["judge"]["calls"] == 6
    assert metrics["cost_usd"]["total_with_judge"] == round(0.12 + 0.5, 6)


def test_error_breakdown_groups_by_kind():
    traces = [
        trace("q1", [], error="HTTP 503: service unavailable"),
        trace("q2", [], error="HTTP 503: service unavailable"),
        trace("q3", [], error="TimeoutError: read timed out"),
    ]
    breakdown = ops.aggregate(traces)["error_breakdown"]
    assert breakdown["HTTP 503"] == 2
    assert breakdown["TimeoutError"] == 1


def test_empty_run_reports_none_rather_than_dividing_by_zero():
    metrics = ops.aggregate([])
    assert metrics["error_rate"] is None
    assert metrics["latency_ms"]["p95"] is None
