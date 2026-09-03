"""Run-over-run regression diff.

This is the point of the framework: the diff between the run before a RAG
change and the run after it is the go/no-go evidence (design spec §3.1 step 5).

Direction matters -- ``hallucination_rate`` going up is a regression while
``recall@5`` going up is an improvement -- so every compared metric declares
which way is better.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from rag_eval.metrics.scorecard import HEADLINE

#: Metrics compared beyond the headline set, as (block, key, direction).
EXTRA_COMPARISONS = (
    ("retrieval", "hit_rate@1", "higher"),
    ("retrieval", "hit_rate@3", "higher"),
    ("retrieval", "hit_rate@10", "higher"),
    ("retrieval", "recall@1", "higher"),
    ("retrieval", "recall@3", "higher"),
    ("retrieval", "recall@10", "higher"),
    ("retrieval", "sitting_citation_accuracy", "higher"),
    ("generation", "citation_accuracy", "higher"),
    ("generation", "faithfulness_pass_rate", "higher"),
    ("generation", "correctness_pass_rate", "higher"),
    ("generation", "empty_answer_rate", "lower"),
)

#: A change smaller than this is reported as "flat" rather than as movement.
DEFAULT_EPSILON = 1e-9


@dataclass
class MetricDelta:
    block: str
    metric: str
    better: str
    current: float | None
    baseline: float | None

    @property
    def delta(self) -> float | None:
        if self.current is None or self.baseline is None:
            return None
        return self.current - self.baseline

    @property
    def pct_change(self) -> float | None:
        d = self.delta
        if d is None or not self.baseline:
            return None
        return d / abs(self.baseline) * 100.0

    def verdict(self, epsilon: float = DEFAULT_EPSILON) -> str:
        """``improved`` | ``regressed`` | ``flat`` | ``new`` | ``missing``."""
        if self.current is None and self.baseline is None:
            return "missing"
        if self.baseline is None:
            return "new"
        if self.current is None:
            return "missing"
        d = self.delta or 0.0
        if abs(d) <= epsilon:
            return "flat"
        improved = d > 0 if self.better == "higher" else d < 0
        return "improved" if improved else "regressed"

    def to_dict(self, epsilon: float = DEFAULT_EPSILON) -> dict[str, Any]:
        return {
            "block": self.block,
            "metric": self.metric,
            "better": self.better,
            "current": self.current,
            "baseline": self.baseline,
            "delta": None if self.delta is None else round(self.delta, 6),
            "pct_change": None if self.pct_change is None else round(self.pct_change, 2),
            "verdict": self.verdict(epsilon),
        }


def _get(scorecard: dict[str, Any], block: str, key: str) -> float | None:
    value = (scorecard.get(block) or {}).get(key)
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def compare(
    current: dict[str, Any],
    baseline: dict[str, Any] | None,
    *,
    epsilon: float = DEFAULT_EPSILON,
    comparisons: Sequence[tuple[str, str, str]] | None = None,
) -> dict[str, Any]:
    """Diff two scorecards. A missing baseline yields a first-run report."""
    specs = list(comparisons or (tuple(HEADLINE) + EXTRA_COMPARISONS))
    deltas = [
        MetricDelta(
            block=block,
            metric=metric,
            better=better,
            current=_get(current, block, metric),
            baseline=_get(baseline, block, metric) if baseline else None,
        )
        for block, metric, better in specs
    ]
    rows = [d.to_dict(epsilon) for d in deltas]
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1

    return {
        "baseline_run_id": (baseline or {}).get("run_id"),
        "current_run_id": current.get("run_id"),
        "adapter": current.get("adapter"),
        "has_baseline": baseline is not None,
        "summary": counts,
        "regressed": [r for r in rows if r["verdict"] == "regressed"],
        "improved": [r for r in rows if r["verdict"] == "improved"],
        "metrics": rows,
        "per_question": per_question_changes(current, baseline) if baseline else [],
    }


def per_question_changes(
    current: dict[str, Any], baseline: dict[str, Any], *, dimension: str = "correctness"
) -> list[dict[str, Any]]:
    """Questions whose outcome moved, so a regression can be traced to rows.

    An aggregate that drops 2 points is unactionable; the list of questions that
    stopped retrieving their sitting is what someone actually debugs.
    """
    base_by_id = {q["question_id"]: q for q in baseline.get("per_question", [])}
    changes: list[dict[str, Any]] = []

    for row in current.get("per_question", []):
        prior = base_by_id.get(row["question_id"])
        if not prior:
            continue

        now_hit = (row.get("retrieval") or {}).get("hit_at_k", {}).get("5")
        was_hit = (prior.get("retrieval") or {}).get("hit_at_k", {}).get("5")
        now_score = (row.get("scores") or {}).get(dimension)
        was_score = (prior.get("scores") or {}).get(dimension)

        retrieval_moved = now_hit is not None and was_hit is not None and now_hit != was_hit
        score_moved = (
            isinstance(now_score, int) and isinstance(was_score, int) and now_score != was_score
        )
        if not (retrieval_moved or score_moved):
            continue

        changes.append(
            {
                "question_id": row["question_id"],
                "question": row.get("question", "")[:200],
                "gold_sitting": row.get("gold_sitting"),
                "hit@5": {"baseline": was_hit, "current": now_hit},
                dimension: {"baseline": was_score, "current": now_score},
                "direction": _row_direction(was_hit, now_hit, was_score, now_score),
            }
        )

    order = {"regressed": 0, "mixed": 1, "improved": 2}
    changes.sort(key=lambda c: (order.get(c["direction"], 3), c["question_id"]))
    return changes


def _row_direction(was_hit: Any, now_hit: Any, was_score: Any, now_score: Any) -> str:
    signals = []
    if was_hit is not None and now_hit is not None and was_hit != now_hit:
        signals.append("improved" if now_hit else "regressed")
    if isinstance(was_score, int) and isinstance(now_score, int) and was_score != now_score:
        signals.append("improved" if now_score > was_score else "regressed")
    if not signals:
        return "flat"
    return signals[0] if len(set(signals)) == 1 else "mixed"
