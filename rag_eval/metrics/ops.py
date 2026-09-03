"""Ops metrics: latency, tokens, estimated cost per query, error rate.

Cost is *estimated*: the per-1k-token rates live in config, so a scorecard says
what it assumed. When a service reports no token usage the cost is reported as
``None`` rather than 0 -- an unknown cost that renders as free is worse than no
number at all.
"""

from __future__ import annotations

from typing import Any, Sequence

from rag_eval.config import Config
from rag_eval.judges.panel import PanelUsage
from rag_eval.types import RagTrace, percentile


def aggregate(
    traces: Sequence[RagTrace],
    *,
    input_cost_per_1k: float = 0.0,
    output_cost_per_1k: float = 0.0,
    judge_usage: PanelUsage | None = None,
) -> dict[str, Any]:
    total = len(traces)
    ok = [t for t in traces if t.ok]
    latencies = [t.latency_ms for t in ok if t.latency_ms is not None]

    prompt_tokens = [t.prompt_tokens for t in ok if t.prompt_tokens is not None]
    completion_tokens = [t.completion_tokens for t in ok if t.completion_tokens is not None]
    has_tokens = bool(prompt_tokens or completion_tokens)

    cost_total = (
        sum(prompt_tokens) / 1000.0 * input_cost_per_1k
        + sum(completion_tokens) / 1000.0 * output_cost_per_1k
    ) if has_tokens else None

    metrics: dict[str, Any] = {
        "questions": total,
        "succeeded": len(ok),
        "errors": total - len(ok),
        "error_rate": round((total - len(ok)) / total, 4) if total else None,
        "latency_ms": {
            "p50": _round(percentile(latencies, 0.50)),
            "p95": _round(percentile(latencies, 0.95)),
            "p99": _round(percentile(latencies, 0.99)),
            "mean": _round(sum(latencies) / len(latencies)) if latencies else None,
            "max": _round(max(latencies)) if latencies else None,
            "measured": len(latencies),
        },
        "tokens": {
            "prompt_total": sum(prompt_tokens) if prompt_tokens else None,
            "completion_total": sum(completion_tokens) if completion_tokens else None,
            "mean_per_query": (
                round((sum(prompt_tokens) + sum(completion_tokens)) / len(ok), 1)
                if has_tokens and ok
                else None
            ),
        },
        "cost_usd": {
            "rag_total": _round(cost_total, 6),
            "rag_per_query": _round(cost_total / len(ok), 6) if cost_total is not None and ok else None,
            "input_cost_per_1k": input_cost_per_1k,
            "output_cost_per_1k": output_cost_per_1k,
        },
    }

    if judge_usage is not None:
        metrics["judge"] = {
            **judge_usage.to_dict(),
            "cost_per_query": (
                round(judge_usage.cost_usd / total, 6) if total else None
            ),
        }
        if cost_total is not None:
            metrics["cost_usd"]["total_with_judge"] = _round(
                cost_total + judge_usage.cost_usd, 6
            )

    errors = [t.error for t in traces if t.error]
    if errors:
        counts: dict[str, int] = {}
        for e in errors:
            key = e.split(":")[0][:80]
            counts[key] = counts.get(key, 0) + 1
        metrics["error_breakdown"] = dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    return metrics


def costs_for_adapter(config: Config, adapter: str) -> tuple[float, float]:
    """Per-1k input/output token rates configured for a RAG service."""
    options = config.adapter_options(adapter)
    return (
        float(options.get("input_cost_per_1k", 0.0)),
        float(options.get("output_cost_per_1k", 0.0)),
    )


def _round(value: float | None, digits: int = 2) -> float | None:
    return None if value is None else round(value, digits)
