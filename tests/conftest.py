"""Shared fixtures: a synthetic golden set and traces with known-good scores.

Every metric test works from traces whose correct answer is computable by hand,
so a failing assertion points at the metric rather than at the fixture.
"""

from __future__ import annotations

import json

import pytest

from rag_eval.dataset.loader import write_golden_set
from rag_eval.types import GoldenItem, RagTrace, RetrievedChunk, SourceRef


def ref(doc: str = "dr", date: str = "2026-06-22", *pages: int) -> SourceRef:
    return SourceRef(doc_type=doc, sitting_date=date, pages=tuple(pages),
                     raw=f"{doc}_{date}, ms. {','.join(map(str, pages))}")


def chunk(rank: int, sitting: str | None, page: int | None, text: str = "context") -> RetrievedChunk:
    return RetrievedChunk(rank=rank, text=text, chunk_id=f"c{rank}",
                          score=1.0 / rank, sitting_id=sitting, page=page)


def item(qid: str, reference: SourceRef | None = None) -> GoldenItem:
    return GoldenItem(
        id=qid,
        question=f"Question {qid}?",
        expected_answer=f"Golden answer for {qid}.",
        reference=reference,
    )


def trace(qid: str, chunks: list[RetrievedChunk], *, cited: list[SourceRef] | None = None,
          answer: str = "an answer", latency_ms: float = 100.0,
          error: str | None = None) -> RagTrace:
    return RagTrace(
        question_id=qid,
        question=f"Question {qid}?",
        adapter="mock",
        generated_answer="" if error else answer,
        retrieved_chunks=chunks,
        cited_sources=cited or [],
        latency_ms=latency_ms,
        prompt_tokens=100,
        completion_tokens=20,
        model="mock-1",
        error=error,
    )


@pytest.fixture
def golden_items() -> list[GoldenItem]:
    return [
        item("q1", ref("dr", "2026-06-22", 3)),
        item("q2", ref("dn", "2026-08-04", 15, 16)),
        item("q3", ref("kkdr", "2026-07-14", 1, 2, 5)),
        item("q4", None),  # no reference -> excluded from retrieval metrics
    ]


@pytest.fixture
def golden_file(tmp_path, golden_items):
    path = tmp_path / "golden_v1.jsonl"
    write_golden_set(path, golden_items, source="synthetic")
    return path


@pytest.fixture
def perfect_verdict_json() -> str:
    return json.dumps(
        {
            "scores": {
                "faithfulness": 5,
                "correctness": 5,
                "completeness": 4,
                "citation_accuracy": 5,
            },
            "rationale": "Every claim is supported by the retrieved context.",
        }
    )
