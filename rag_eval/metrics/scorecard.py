"""Assemble one run's metrics into a single scorecard document.

The scorecard is the artifact a human reads to make a go/no-go call, so it
carries the numbers *and* the caveats: how many questions were unscorable, how
many judge rows are still waiting on a human, and whether the retrieved chunks
even carried the metadata the retrieval metrics need.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from rag_eval.config import Config
from rag_eval.judges.panel import PanelUsage
from rag_eval.judges.rubric import PanelVerdict
from rag_eval.metrics import generation, ops, retrieval
from rag_eval.types import GoldenItem, RagTrace

#: Metrics promoted to the scorecard headline, with the direction that is better.
HEADLINE = (
    ("retrieval", "hit_rate@5", "higher"),
    ("retrieval", "recall@5", "higher"),
    ("retrieval", "mrr", "higher"),
    ("retrieval", "page_citation_accuracy", "higher"),
    ("generation", "faithfulness", "higher"),
    ("generation", "correctness", "higher"),
    ("generation", "completeness", "higher"),
    ("generation", "hallucination_rate", "lower"),
    ("ops", "error_rate", "lower"),
)


@dataclass
class Scorecard:
    run_id: str
    adapter: str
    dataset: str = ""
    created_at: str = ""
    retrieval: dict[str, Any] = field(default_factory=dict)
    generation: dict[str, Any] = field(default_factory=dict)
    ops: dict[str, Any] = field(default_factory=dict)
    judges: dict[str, Any] = field(default_factory=dict)
    per_question: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def headline(self) -> list[dict[str, Any]]:
        rows = []
        for block, key, direction in HEADLINE:
            value = getattr(self, block).get(key)
            rows.append({"block": block, "metric": key, "value": value, "better": direction})
        return rows

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "adapter": self.adapter,
            "dataset": self.dataset,
            "created_at": self.created_at,
            "retrieval": self.retrieval,
            "generation": self.generation,
            "ops": self.ops,
            "judges": self.judges,
            "warnings": self.warnings,
            "headline": self.headline(),
            "per_question": self.per_question,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Scorecard":
        return cls(
            run_id=d.get("run_id", ""),
            adapter=d.get("adapter", ""),
            dataset=d.get("dataset", ""),
            created_at=d.get("created_at", ""),
            retrieval=d.get("retrieval", {}),
            generation=d.get("generation", {}),
            ops=d.get("ops", {}),
            judges=d.get("judges", {}),
            per_question=d.get("per_question", []),
            warnings=list(d.get("warnings", [])),
        )


def build_scorecard(
    run_id: str,
    adapter: str,
    items: Sequence[GoldenItem],
    traces: Sequence[RagTrace],
    verdicts: Sequence[PanelVerdict] | None = None,
    *,
    config: Config | None = None,
    dataset: str = "",
    created_at: str = "",
    judge_usage: PanelUsage | None = None,
) -> Scorecard:
    config = config or Config()
    verdicts = list(verdicts or [])

    by_id = {t.question_id: t for t in traces}
    pairs = [(item, by_id[item.id]) for item in items if item.id in by_id]

    retrieval_results = [
        retrieval.score_question(item, trace, config.metrics.k_values) for item, trace in pairs
    ]
    input_cost, output_cost = ops.costs_for_adapter(config, adapter)

    card = Scorecard(
        run_id=run_id,
        adapter=adapter,
        dataset=dataset,
        created_at=created_at,
        retrieval=retrieval.aggregate(retrieval_results, config.metrics.k_values),
        generation=generation.aggregate(
            verdicts,
            [t for _, t in pairs],
            hallucination_threshold=config.metrics.hallucination_threshold,
            pass_threshold=config.metrics.pass_threshold,
        ),
        ops=ops.aggregate(
            [t for _, t in pairs],
            input_cost_per_1k=input_cost,
            output_cost_per_1k=output_cost,
            judge_usage=judge_usage,
        ),
        judges=generation.per_judge_scores(verdicts),
    )

    verdict_by_id = {v.question_id: v for v in verdicts}
    item_by_id = {i.id: i for i in items}
    for result in retrieval_results:
        verdict = verdict_by_id.get(result.question_id)
        item = item_by_id[result.question_id]
        trace = by_id[result.question_id]
        card.per_question.append(
            {
                "question_id": result.question_id,
                "question": item.question,
                "gold_sitting": result.gold_sitting,
                "gold_pages": list(result.gold_pages),
                "expected_answer": item.expected_answer,
                "generated_answer": trace.generated_answer,
                "error": trace.error,
                "latency_ms": trace.latency_ms,
                "retrieval": result.to_dict(),
                "scores": verdict.final_scores() if verdict else {},
                "flagged": list(verdict.flagged_dimensions) if verdict else [],
                "agreement": dict(verdict.agreement) if verdict else {},
                "rationales": (
                    {v.model: v.rationale for v in verdict.verdicts if v.rationale}
                    if verdict
                    else {}
                ),
            }
        )

    card.warnings = _warnings(card)
    return card


def _warnings(card: Scorecard) -> list[str]:
    """Caveats that change how the headline numbers should be read."""
    out: list[str] = []

    health = card.retrieval.get("metadata_health", {})
    pct = health.get("chunks_with_sitting_id_pct")
    if pct is not None and pct < 95:
        out.append(
            f"Only {pct}% of retrieved chunks carry a sitting id — retrieval and "
            f"page-citation metrics measure the ingestion path as much as the retriever. "
            f"Run 'rag-eval ingest-check' before trusting them."
        )

    unscorable = card.retrieval.get("unscorable_no_reference", 0)
    if unscorable:
        out.append(
            f"{unscorable} question(s) have no golden reference and are excluded from "
            f"retrieval metrics (not counted as misses)."
        )

    awaiting = card.generation.get("panel_disagreement", {}).get("awaiting_human_review", 0)
    if awaiting:
        out.append(
            f"{awaiting} judged question(s) split the panel and are still awaiting human "
            f"adjudication; generation means include provisional median scores."
        )

    unjudged = card.generation.get("unjudged", 0)
    if unjudged and card.generation.get("questions"):
        out.append(f"{unjudged} question(s) were not judged (adapter error or judge failure).")

    error_rate = card.ops.get("error_rate")
    if error_rate:
        out.append(f"Adapter error rate {error_rate:.1%} — failed questions are excluded from judging.")

    for model, stats in card.judges.items():
        if stats.get("failure_rate") and stats["failure_rate"] > 0.05:
            out.append(
                f"Judge {model} failed on {stats['failure_rate']:.1%} of calls; its votes are "
                f"missing from those rows."
            )
    return out
