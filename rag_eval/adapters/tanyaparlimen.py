"""Adapter for TanyaParlimen's production RAG.

Same contract as the NVIDIA adapter, different wire format: TanyaParlimen
answers with ``{answer, sources: [...]}``, where a source carries the Hansard
document id and page. Field paths stay configurable for the same reason.
"""

from __future__ import annotations

import os
from typing import Any

from rag_eval.adapters.base import RagAdapter, dig
from rag_eval.adapters.http import HttpError, get_json, post_json
from rag_eval.types import RagTrace

DEFAULTS = {
    "ask_path": "/api/ask",
    "health_path": "/health",
    "answer_field": "answer",
    "chunks_field": "sources",
    "chunk_text_field": "text",
    "chunk_score_field": "score",
    "chunk_id_field": "id",
    "sitting_fields": ["sitting_id", "document_id", "metadata.sitting_id", "document"],
    "page_fields": ["page", "ms", "metadata.page"],
    "citations_field": "citations",
    "prompt_tokens_field": "usage.prompt_tokens",
    "completion_tokens_field": "usage.completion_tokens",
    "model_field": "model",
}


class TanyaParlimenAdapter(RagAdapter):
    """Options: ``base_url``, ``top_k``, ``api_key_env``, ``session_id``,
    ``language``, plus any field-path override from ``DEFAULTS``."""

    name = "tanyaparlimen"

    def __init__(self, **options: Any) -> None:
        super().__init__(**options)
        self.settings = {**DEFAULTS, **options}
        self.base_url = str(
            options.get("base_url") or os.environ.get("TANYAPARLIMEN_BASE_URL", "")
        ).rstrip("/")
        if not self.base_url:
            raise ValueError(
                "TanyaParlimenAdapter needs base_url (option) or TANYAPARLIMEN_BASE_URL (env)"
            )
        self.api_key = os.environ.get(str(options.get("api_key_env", "TANYAPARLIMEN_API_KEY")), "")
        self.timeout = float(options.get("timeout_s", 120.0))
        self.language = str(options.get("language", "")) or None

    def answer(self, question: str, question_id: str = "") -> RagTrace:
        trace = self._trace(question, question_id)
        payload: dict[str, Any] = {"question": question, "top_k": self.top_k}
        if self.language:
            payload["language"] = self.language
        if self.options.get("session_id"):
            payload["session_id"] = self.options["session_id"]

        url = f"{self.base_url}{self.settings['ask_path']}"
        try:
            with self.timer() as sw:
                body = post_json(url, payload, headers=self._headers(), timeout=self.timeout)
        except (HttpError, RuntimeError, ValueError) as exc:
            # ValueError covers a 200 whose body is not JSON — a broken service
            # must produce an error trace, never take down the run.
            trace.error = str(exc)
            return trace

        trace.latency_ms = sw.elapsed_ms
        s = self.settings
        trace.raw_response = body if self.options.get("keep_raw") else None
        trace.generated_answer = str(dig(body, s["answer_field"], "") or "")
        trace.model = str(dig(body, s["model_field"], "") or "")
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
                metadata=c if isinstance(c, dict) else {},
            )
            for i, c in enumerate(raw_chunks)
        ]

        cited = dig(body, s["citations_field"], None)
        if isinstance(cited, list) and cited:
            trace.cited_sources = self.parse_citations(cited)
        else:
            trace.cited_sources = self.citations_from_chunks(trace.retrieved_chunks)

        if not trace.generated_answer and not trace.retrieved_chunks:
            trace.error = "empty response from TanyaParlimen"
        return trace

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
            "top_k": self.top_k,
            "language": self.language or "(service default)",
        }

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}


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
