"""The adapter contract every RAG service plugs in through.

An adapter's whole job is to turn one question into a :class:`RagTrace` with
*populated retrieval metadata*. Everything downstream -- retrieval metrics,
judge context, citation accuracy -- depends on ``sitting_id`` and ``page``
surviving the trip out of the RAG service, which is why the base class provides
the normalisation helpers rather than leaving each adapter to invent them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable

from rag_eval.dataset.refs import ReferenceParseError, normalise_sitting_id, parse_reference
from rag_eval.types import RagTrace, RetrievedChunk, SourceRef, Stopwatch


class AdapterError(RuntimeError):
    """The RAG service failed in a way the harness should record, not crash on."""


class RagAdapter(ABC):
    """Common interface: ``answer(question) -> RagTrace``."""

    #: Registry key used by ``--adapter``.
    name: str = "base"

    def __init__(self, **options: Any) -> None:
        self.options = options
        self.top_k: int = int(options.get("top_k", 5))

    # -- contract ---------------------------------------------------------
    @abstractmethod
    def answer(self, question: str, question_id: str = "") -> RagTrace:
        """Run one question end to end and return the full trace.

        ``question_id`` is carried through for traceability only; an adapter
        must not use it to look up an answer (that would be cheating on the
        eval). The mock adapter is the deliberate exception.

        Implementations should record a transport/service failure as
        ``RagTrace.error`` rather than raising, so that one bad question does
        not abort a 371-question run. Raise only for misconfiguration.
        """

    def health(self) -> tuple[bool, str]:
        """Cheap reachability probe used by ``rag-eval ingest-check``."""
        return True, "no health check implemented"

    def probe_environment(self) -> dict[str, Any]:
        """What the RAG stack *actually is*, read from the service at run time.

        ``describe()`` reports what we configured; this reports what is really
        serving. The two diverge in ways that silently invalidate a comparison
        -- a collection built dense-only vs. hybrid, or a stack whose recorded
        embedding model is a stale label for the one it actually loads. Those
        differences move scores without anything in the RAG changing, so they
        belong in the manifest and in the run-over-run comparability check.

        Best-effort by contract: an adapter that cannot determine a value
        reports ``"unknown"`` rather than guessing, and a probe that fails must
        never abort a run.
        """
        return {}

    def describe(self) -> dict[str, Any]:
        """Config captured into the run manifest (must not contain secrets)."""
        return {
            "adapter": self.name,
            "top_k": self.top_k,
            "options": {k: v for k, v in self.options.items() if "key" not in k.lower()},
        }

    def close(self) -> None:
        """Release connections. Adapters that hold none need not override."""

    def __enter__(self) -> "RagAdapter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- helpers for implementations --------------------------------------
    def _trace(self, question: str, question_id: str = "") -> RagTrace:
        return RagTrace(question_id=question_id, question=question, adapter=self.name)

    @staticmethod
    def timer() -> Stopwatch:
        return Stopwatch()

    @staticmethod
    def make_chunk(
        rank: int,
        text: str,
        *,
        chunk_id: str = "",
        score: float | None = None,
        sitting_id: Any = None,
        page: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> RetrievedChunk:
        """Build a chunk with sitting id and page coerced into canonical form."""
        return RetrievedChunk(
            rank=rank,
            text=text or "",
            chunk_id=str(chunk_id or ""),
            score=float(score) if score is not None else None,
            sitting_id=normalise_sitting_id(sitting_id),
            page=_coerce_page(page),
            metadata=dict(metadata or {}),
        )

    @staticmethod
    def parse_citations(values: Iterable[Any]) -> list[SourceRef]:
        """Parse whatever the service calls a citation into ``SourceRef``s.

        Unparseable citations are dropped rather than fatal: citation accuracy
        should score them as a miss, not kill the run.
        """
        refs: list[SourceRef] = []
        for value in values or ():
            if isinstance(value, SourceRef):
                refs.append(value)
                continue
            try:
                ref = parse_reference(str(value))
            except ReferenceParseError:
                continue
            if ref is not None:
                refs.append(ref)
        return refs

    @staticmethod
    def citations_from_chunks(chunks: Iterable[RetrievedChunk]) -> list[SourceRef]:
        """Fallback citations for services that cite by returning chunks.

        One ``SourceRef`` per (sitting, page) pair present in the retrieved set,
        in retrieval order.
        """
        refs: list[SourceRef] = []
        seen: set[tuple[str, int | None]] = set()
        for c in sorted(chunks, key=lambda c: c.rank):
            if not c.sitting_id:
                continue
            key = (c.sitting_id, c.page)
            if key in seen:
                continue
            seen.add(key)
            doc_type, _, date = c.sitting_id.partition("_")
            refs.append(
                SourceRef(
                    doc_type=doc_type,
                    sitting_date=date,
                    pages=(c.page,) if c.page is not None else (),
                    raw=f"{c.sitting_id}, ms. {c.page}" if c.page is not None else c.sitting_id,
                )
            )
        return refs


def _coerce_page(value: Any) -> int | None:
    """``'ms. 12'``, ``'12'``, ``12.0`` -> ``12``; anything else -> ``None``."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    import re

    m = re.search(r"\d+", str(value))
    return int(m.group()) if m else None


def dig(data: Any, path: str, default: Any = None) -> Any:
    """Read a dotted path out of nested dicts/lists: ``'choices.0.message.content'``.

    Adapters use this so the response field layout of a RAG service can be
    adjusted in config when the service's API shifts, without a code change.
    """
    current = data
    for part in path.split("."):
        if current is None:
            return default
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, (list, tuple)):
            if not part.lstrip("-").isdigit():
                return default
            idx = int(part)
            current = current[idx] if -len(current) <= idx < len(current) else None
        else:
            return default
    return default if current is None else current
