"""Adapter for the hosted NVIDIA RAG stack (Ingestor + RAG Server, RTX6000).

Field paths into the service's JSON are configurable rather than hard-coded:
the RAG Blueprint's response shape has moved between releases, and a shifted
key should be a config edit, not a patch release of this framework.

Chunk metadata is where this adapter earns its keep. ``page_citation_accuracy``
is only computable if the ingestion path preserved sitting id + page (design
spec §6); ``rag-eval ingest-check`` uses ``metadata_coverage`` below to prove
it did before anyone trusts a score.
"""

from __future__ import annotations

import json
import os
from typing import Any

from rag_eval.adapters.base import RagAdapter, dig
from rag_eval.adapters.http import HttpError, get_json, post_json
from rag_eval.types import RagTrace

DEFAULTS = {
    "generate_path": "/v1/generate",
    "search_path": "/v1/search",
    "health_path": "/v1/health",
    "answer_field": "choices.0.message.content",
    "citations_field": "citations.results",
    "chunks_field": "citations.results",
    "chunk_text_field": "content",
    "chunk_score_field": "score",
    "chunk_id_field": "document_id",
    # Where the ingestor put the Hansard metadata. Several candidates are tried
    # in order, so one adapter build works across ingestion revisions.
    "sitting_fields": ["metadata.sitting_id", "metadata.source_id", "document_name", "source"],
    "page_fields": ["metadata.page_number", "metadata.page", "metadata.ms"],
    "prompt_tokens_field": "usage.prompt_tokens",
    "completion_tokens_field": "usage.completion_tokens",
    "model_field": "model",
    # Environment probing. The NIM's own /v1/models is authoritative for the
    # embedding model; a collection's recorded metadata is a label that can go
    # stale against what the service actually loaded.
    "models_path": "/v1/models",
    "collection_info_path": "",          # e.g. "/v1/collections/{collection}"
    "sparse_markers": ["sparse", "bm25", "BM25"],
}


class NvidiaRagAdapter(RagAdapter):
    """Options: ``base_url``, ``collection``, ``top_k``, ``api_key_env``,
    ``model``, ``temperature``, plus any field-path override from ``DEFAULTS``."""

    name = "nvidia"

    def __init__(self, **options: Any) -> None:
        super().__init__(**options)
        self.settings = {**DEFAULTS, **options}
        self.base_url = str(
            options.get("base_url") or os.environ.get("NVIDIA_RAG_BASE_URL", "")
        ).rstrip("/")
        if not self.base_url:
            raise ValueError(
                "NvidiaRagAdapter needs base_url (option) or NVIDIA_RAG_BASE_URL (env)"
            )
        self.collection = str(
            options.get("collection") or os.environ.get("NVIDIA_RAG_COLLECTION", "parliament-hansard-eval")
        )
        self.api_key = os.environ.get(str(options.get("api_key_env", "NVIDIA_RAG_API_KEY")), "")
        self.timeout = float(options.get("timeout_s", 120.0))
        self.model = str(options.get("model", ""))

    # -- contract ---------------------------------------------------------
    def answer(self, question: str, question_id: str = "") -> RagTrace:
        trace = self._trace(question, question_id)
        payload: dict[str, Any] = {
            "messages": [{"role": "user", "content": question}],
            "use_knowledge_base": True,
            "collection_names": [self.collection],
            "top_k": self.top_k,
            "temperature": float(self.settings.get("temperature", 0.0)),
            "stream": False,
        }
        if self.model:
            payload["model"] = self.model

        url = f"{self.base_url}{self.settings['generate_path']}"
        try:
            with self.timer() as sw:
                body = post_json(url, payload, headers=self._headers(), timeout=self.timeout)
        except (HttpError, RuntimeError) as exc:
            trace.error = str(exc)
            return trace

        trace.latency_ms = sw.elapsed_ms
        return self._populate(trace, body)

    def health(self) -> tuple[bool, str]:
        url = f"{self.base_url}{self.settings['health_path']}"
        try:
            body = get_json(url, headers=self._headers(), timeout=15.0)
        except Exception as exc:  # pragma: no cover - network path
            return False, f"{url}: {exc}"
        return True, f"{url}: {body if isinstance(body, str) else 'ok'}"

    def describe(self) -> dict[str, Any]:
        return {
            "adapter": self.name,
            "base_url": self.base_url,
            "collection": self.collection,
            "top_k": self.top_k,
            "model": self.model or "(server default)",
        }

    # -- internals --------------------------------------------------------
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def _populate(self, trace: RagTrace, body: dict[str, Any]) -> RagTrace:
        s = self.settings
        trace.raw_response = body if self.options.get("keep_raw") else None
        trace.generated_answer = str(dig(body, s["answer_field"], "") or "")
        trace.model = str(dig(body, s["model_field"], self.model) or self.model)
        trace.prompt_tokens = _as_int(dig(body, s["prompt_tokens_field"]))
        trace.completion_tokens = _as_int(dig(body, s["completion_tokens_field"]))

        raw_chunks = dig(body, s["chunks_field"], []) or []
        if not isinstance(raw_chunks, list):
            raw_chunks = []
        trace.retrieved_chunks = [
            self.make_chunk(
                rank=i + 1,
                text=str(dig(c, s["chunk_text_field"], "") or ""),
                chunk_id=str(dig(c, s["chunk_id_field"], "") or ""),
                score=_as_float(dig(c, s["chunk_score_field"])),
                sitting_id=_first(c, s["sitting_fields"]),
                page=_first(c, s["page_fields"]),
                metadata=c.get("metadata", {}) if isinstance(c, dict) else {},
            )
            for i, c in enumerate(raw_chunks)
        ]

        cited = dig(body, s["citations_field"], None)
        if isinstance(cited, list) and cited and isinstance(cited[0], str):
            trace.cited_sources = self.parse_citations(cited)
        else:
            trace.cited_sources = self.citations_from_chunks(trace.retrieved_chunks)

        if not trace.generated_answer and not trace.retrieved_chunks:
            trace.error = "empty response from RAG server"
        return trace

    def probe_environment(self) -> dict[str, Any]:
        """Read the served embedding model and the collection's retrieval mode.

        Both are recorded with *how* they were determined, because "the NIM told
        us" and "a collection metadata field claimed it" are not the same
        evidence and a manifest that blurs them is not provenance.
        """
        env: dict[str, Any] = {
            "base_url": self.base_url,
            "collection": self.collection,
            "embedding_model": "unknown",
            "embedding_model_source": "unknown",
            "retrieval_mode": "unknown",
            "retrieval_mode_source": "unknown",
        }

        models_path = str(self.settings.get("models_path") or "")
        if models_path:
            try:
                body = get_json(
                    f"{self.base_url}{models_path}", headers=self._headers(), timeout=15.0
                )
                ids = [
                    m.get("id")
                    for m in (body or {}).get("data", [])
                    if isinstance(m, dict) and m.get("id")
                ]
                if ids:
                    env["embedding_model"] = ids[0] if len(ids) == 1 else ", ".join(sorted(ids))
                    env["embedding_model_source"] = f"served: {models_path}"
            except Exception as exc:  # noqa: BLE001 - a probe never fails a run
                env["embedding_model_source"] = f"probe failed: {exc}"[:200]

        info_path = str(self.settings.get("collection_info_path") or "")
        if info_path:
            try:
                body = get_json(
                    f"{self.base_url}{info_path.format(collection=self.collection)}",
                    headers=self._headers(), timeout=15.0,
                )
                blob = json.dumps(body).lower()
                markers = [str(m).lower() for m in self.settings.get("sparse_markers", [])]
                env["retrieval_mode"] = "hybrid" if any(m in blob for m in markers) else "dense"
                env["retrieval_mode_source"] = f"probed: {info_path}"
            except Exception as exc:  # noqa: BLE001
                env["retrieval_mode_source"] = f"probe failed: {exc}"[:200]

        # Let an operator state what a probe cannot reach, clearly marked as such.
        for key in ("retrieval_mode", "embedding_model"):
            declared = self.options.get(key)
            if declared and env[key] == "unknown":
                env[key] = str(declared)
                env[f"{key}_source"] = "declared in config (unverified)"
        return env

    def metadata_coverage(self, trace: RagTrace) -> dict[str, Any]:
        """How much of the retrieved set carries scoreable Hansard metadata."""
        chunks = trace.retrieved_chunks
        total = len(chunks)
        return {
            "chunks": total,
            "with_sitting_id": sum(1 for c in chunks if c.sitting_id),
            "with_page": sum(1 for c in chunks if c.page is not None),
            "sittings": sorted({c.sitting_id for c in chunks if c.sitting_id}),
        }


def _first(record: Any, paths: list[str]) -> Any:
    for path in paths:
        value = dig(record, path)
        if value not in (None, ""):
            return value
    return None


def _as_int(value: Any) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
