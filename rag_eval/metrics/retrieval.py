"""Retrieval metrics: does the RAG stack find the right evidence?

Definitions follow design spec §4.2 exactly:

``hit_rate@k``              >=1 of the top-k chunks comes from the correct sitting
``recall@k``                a top-k chunk carries a gold page of the correct sitting
``mrr``                     mean reciprocal rank of the first correct-sitting chunk
``page_citation_accuracy``  the RAG's cited ``ms.`` matches the Excel page (deterministic)

Two rules apply throughout, and both matter when reading a scorecard:

1. Questions with no golden reference are *excluded* from the denominator, not
   scored as misses. Every metric reports its own ``scorable`` count.
2. A chunk whose metadata lost ``sitting_id``/``page`` can never match. That is
   an ingestion defect, not a retrieval failure, so ``metadata_health`` reports
   it separately -- check it before believing a bad hit-rate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from rag_eval.types import GoldenItem, RagTrace, RetrievedChunk, SourceRef


@dataclass
class RetrievalResult:
    """Per-question retrieval outcome, kept so a scorecard row can be explained."""

    question_id: str
    scorable: bool
    gold_sitting: str | None = None
    gold_pages: tuple[int, ...] = ()
    hit_at_k: dict[int, bool] = field(default_factory=dict)
    recall_at_k: dict[int, bool] = field(default_factory=dict)
    first_correct_rank: int | None = None
    reciprocal_rank: float = 0.0
    cited_sittings: list[str] = field(default_factory=list)
    citation_sitting_match: bool = False
    citation_page_match: bool = False
    retrieved_chunks: int = 0
    chunks_missing_metadata: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "scorable": self.scorable,
            "gold_sitting": self.gold_sitting,
            "gold_pages": list(self.gold_pages),
            "hit_at_k": {str(k): v for k, v in self.hit_at_k.items()},
            "recall_at_k": {str(k): v for k, v in self.recall_at_k.items()},
            "first_correct_rank": self.first_correct_rank,
            "reciprocal_rank": round(self.reciprocal_rank, 6),
            "cited_sittings": list(self.cited_sittings),
            "citation_sitting_match": self.citation_sitting_match,
            "citation_page_match": self.citation_page_match,
            "retrieved_chunks": self.retrieved_chunks,
            "chunks_missing_metadata": self.chunks_missing_metadata,
        }


def _chunk_matches_sitting(chunk: RetrievedChunk, sitting_id: str) -> bool:
    return bool(chunk.sitting_id) and chunk.sitting_id == sitting_id


def _chunk_matches_page(chunk: RetrievedChunk, sitting_id: str, pages: Sequence[int]) -> bool:
    if not _chunk_matches_sitting(chunk, sitting_id) or chunk.page is None:
        return False
    # No gold page recorded -> the correct sitting is the strongest claim we can make.
    return chunk.page in pages if pages else True


def citation_matches(
    cited: Iterable[SourceRef], gold: SourceRef
) -> tuple[bool, bool]:
    """``(sitting matched, page matched)`` for the answer's citations.

    Page matching is an *intersection* test, not set equality: the Excel
    reference is often a range (``ms. 15-16``) where the answer legitimately
    cites one page of it, and demanding the full range would score correct
    behaviour as a miss.
    """
    sitting_ok = False
    page_ok = False
    for ref in cited:
        if not ref.matches_sitting(gold):
            continue
        sitting_ok = True
        if not gold.pages:
            page_ok = True
        elif set(ref.pages) & set(gold.pages):
            page_ok = True
    return sitting_ok, page_ok


def score_question(
    item: GoldenItem, trace: RagTrace, k_values: Sequence[int] = (1, 3, 5, 10)
) -> RetrievalResult:
    """Score one question's retrieval against its golden reference."""
    result = RetrievalResult(
        question_id=item.id,
        scorable=item.is_scorable_for_retrieval,
        retrieved_chunks=len(trace.retrieved_chunks),
        chunks_missing_metadata=sum(1 for c in trace.retrieved_chunks if not c.sitting_id),
    )
    if not item.reference:
        return result

    gold = item.reference
    result.gold_sitting = gold.sitting_id
    result.gold_pages = gold.pages

    ordered = sorted(trace.retrieved_chunks, key=lambda c: c.rank)
    for k in k_values:
        window = ordered[:k]
        result.hit_at_k[k] = any(_chunk_matches_sitting(c, gold.sitting_id) for c in window)
        result.recall_at_k[k] = any(
            _chunk_matches_page(c, gold.sitting_id, gold.pages) for c in window
        )

    for position, chunk in enumerate(ordered, 1):
        if _chunk_matches_sitting(chunk, gold.sitting_id):
            result.first_correct_rank = position
            result.reciprocal_rank = 1.0 / position
            break

    result.cited_sittings = [ref.sitting_id for ref in trace.cited_sources]
    result.citation_sitting_match, result.citation_page_match = citation_matches(
        trace.cited_sources, gold
    )
    return result


def _rate(hits: int, total: int) -> float | None:
    return round(hits / total, 4) if total else None


def aggregate(
    results: Sequence[RetrievalResult], k_values: Sequence[int] = (1, 3, 5, 10)
) -> dict[str, Any]:
    """Aggregate per-question retrieval results into the scorecard block."""
    scorable = [r for r in results if r.scorable]
    n = len(scorable)

    metrics: dict[str, Any] = {
        "questions": len(results),
        "scorable": n,
        "unscorable_no_reference": len(results) - n,
    }
    for k in k_values:
        metrics[f"hit_rate@{k}"] = _rate(sum(1 for r in scorable if r.hit_at_k.get(k)), n)
        metrics[f"recall@{k}"] = _rate(sum(1 for r in scorable if r.recall_at_k.get(k)), n)

    metrics["mrr"] = (
        round(sum(r.reciprocal_rank for r in scorable) / n, 4) if n else None
    )
    metrics["page_citation_accuracy"] = _rate(
        sum(1 for r in scorable if r.citation_page_match), n
    )
    metrics["sitting_citation_accuracy"] = _rate(
        sum(1 for r in scorable if r.citation_sitting_match), n
    )
    metrics["metadata_health"] = metadata_health(results)
    return metrics


def metadata_health(results: Sequence[RetrievalResult]) -> dict[str, Any]:
    """Whether the ingestion preserved the metadata these metrics depend on.

    A near-zero ``chunks_with_sitting_id`` means the retrieval numbers above are
    measuring the ingestion path, not the retriever.
    """
    total_chunks = sum(r.retrieved_chunks for r in results)
    missing = sum(r.chunks_missing_metadata for r in results)
    return {
        "retrieved_chunks": total_chunks,
        "chunks_missing_sitting_id": missing,
        "chunks_with_sitting_id_pct": (
            round((total_chunks - missing) / total_chunks * 100, 2) if total_chunks else None
        ),
        "questions_with_no_usable_metadata": sum(
            1 for r in results if r.retrieved_chunks and r.chunks_missing_metadata == r.retrieved_chunks
        ),
    }
