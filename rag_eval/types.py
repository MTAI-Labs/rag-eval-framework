"""Core record types shared by every stage of the pipeline.

Everything here is a plain dataclass with explicit ``to_dict``/``from_dict`` so
run artifacts stay readable JSON that a human can diff in a PR.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any, Iterable, Sequence

# Hansard document-type prefixes used in the Excel reference field.
DOC_TYPES = {
    "dr": "Dewan Rakyat",
    "dn": "Dewan Negara",
    "kkdr": "Kamar Khas Dewan Rakyat",
}


@dataclass(frozen=True)
class SourceRef:
    """A citation into the Hansard corpus: one sitting plus one or more pages.

    ``pages`` is a tuple because the Excel reference field carries ranges
    (``ms. 15-16``) and lists (``ms 1-2, 5``) as well as single pages.
    """

    doc_type: str
    sitting_date: str  # ISO yyyy-mm-dd
    pages: tuple[int, ...] = ()
    raw: str = ""

    @property
    def sitting_id(self) -> str:
        return f"{self.doc_type}_{self.sitting_date}"

    @property
    def chamber(self) -> str:
        return DOC_TYPES.get(self.doc_type, "Unknown")

    def matches_sitting(self, other: "SourceRef | str | None") -> bool:
        if other is None:
            return False
        other_id = other if isinstance(other, str) else other.sitting_id
        return self.sitting_id == other_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_type": self.doc_type,
            "sitting_date": self.sitting_date,
            "pages": list(self.pages),
            "raw": self.raw,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SourceRef":
        return cls(
            doc_type=d["doc_type"],
            sitting_date=d["sitting_date"],
            pages=tuple(int(p) for p in d.get("pages", ())),
            raw=d.get("raw", ""),
        )

    def __str__(self) -> str:  # pragma: no cover - display only
        pages = ", ".join(str(p) for p in self.pages)
        return f"{self.sitting_id}, ms. {pages}" if pages else self.sitting_id


@dataclass
class GoldenItem:
    """One golden Q&A pair from ``TanyaParlimen QnA.xlsx``."""

    id: str
    question: str
    expected_answer: str
    reference: SourceRef | None = None
    owner: str = ""
    source_row: int | None = None
    tags: list[str] = field(default_factory=list)

    @property
    def is_scorable_for_retrieval(self) -> bool:
        """Retrieval metrics need a gold sitting; a few rows have no reference."""
        return self.reference is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "expected_answer": self.expected_answer,
            "reference": self.reference.to_dict() if self.reference else None,
            "owner": self.owner,
            "source_row": self.source_row,
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GoldenItem":
        ref = d.get("reference")
        return cls(
            id=str(d["id"]),
            question=d["question"],
            expected_answer=d["expected_answer"],
            reference=SourceRef.from_dict(ref) if ref else None,
            owner=d.get("owner", ""),
            source_row=d.get("source_row"),
            tags=list(d.get("tags", [])),
        )


@dataclass
class RetrievedChunk:
    """A chunk returned by a RAG service's retriever.

    ``sitting_id`` and ``page`` are the metadata the ingestion path must
    preserve; without them retrieval and page-citation metrics are not
    computable at all (see ``rag-eval ingest-check``).
    """

    rank: int
    text: str = ""
    chunk_id: str = ""
    score: float | None = None
    sitting_id: str | None = None
    page: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RetrievedChunk":
        return cls(
            rank=int(d["rank"]),
            text=d.get("text", ""),
            chunk_id=d.get("chunk_id", ""),
            score=d.get("score"),
            sitting_id=d.get("sitting_id"),
            page=d.get("page"),
            metadata=dict(d.get("metadata", {})),
        )


@dataclass
class RagTrace:
    """The full trace of one golden question through one RAG service."""

    question_id: str
    question: str
    adapter: str
    generated_answer: str = ""
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
    cited_sources: list[SourceRef] = field(default_factory=list)
    latency_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    model: str = ""
    error: str | None = None
    raw_response: dict[str, Any] | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def total_tokens(self) -> int | None:
        if self.prompt_tokens is None and self.completion_tokens is None:
            return None
        return (self.prompt_tokens or 0) + (self.completion_tokens or 0)

    def context_text(self, k: int | None = None) -> str:
        """The retrieved context as the judge sees it."""
        chunks = self.top_k(k) if k else self.retrieved_chunks
        parts = []
        for c in chunks:
            label = c.sitting_id or "unknown-sitting"
            page = f", ms. {c.page}" if c.page is not None else ""
            parts.append(f"[{c.rank}] ({label}{page})\n{c.text}")
        return "\n\n".join(parts)

    def top_k(self, k: int) -> list[RetrievedChunk]:
        ordered = sorted(self.retrieved_chunks, key=lambda c: c.rank)
        return ordered[:k]

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "adapter": self.adapter,
            "generated_answer": self.generated_answer,
            "retrieved_chunks": [c.to_dict() for c in self.retrieved_chunks],
            "cited_sources": [s.to_dict() for s in self.cited_sources],
            "latency_ms": self.latency_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "model": self.model,
            "error": self.error,
            "raw_response": self.raw_response,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RagTrace":
        return cls(
            question_id=str(d["question_id"]),
            question=d["question"],
            adapter=d.get("adapter", ""),
            generated_answer=d.get("generated_answer", ""),
            retrieved_chunks=[RetrievedChunk.from_dict(c) for c in d.get("retrieved_chunks", [])],
            cited_sources=[SourceRef.from_dict(s) for s in d.get("cited_sources", [])],
            latency_ms=d.get("latency_ms"),
            prompt_tokens=d.get("prompt_tokens"),
            completion_tokens=d.get("completion_tokens"),
            model=d.get("model", ""),
            error=d.get("error"),
            raw_response=d.get("raw_response"),
        )


class Stopwatch:
    """Wall-clock timer for adapter latency, in milliseconds."""

    def __enter__(self) -> "Stopwatch":
        self._start = time.perf_counter()
        self.elapsed_ms = 0.0
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0


def dedupe_sittings(chunks: Iterable[RetrievedChunk]) -> list[str]:
    """Distinct sitting ids in retrieval order, ignoring chunks with no metadata."""
    seen: list[str] = []
    for c in sorted(chunks, key=lambda c: c.rank):
        if c.sitting_id and c.sitting_id not in seen:
            seen.append(c.sitting_id)
    return seen


def percentile(values: Sequence[float], q: float) -> float | None:
    """Linear-interpolated percentile; ``q`` in [0, 1]. ``None`` for empty input."""
    xs = sorted(values)
    if not xs:
        return None
    if len(xs) == 1:
        return float(xs[0])
    pos = q * (len(xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return float(xs[lo] * (1 - frac) + xs[hi] * frac)
