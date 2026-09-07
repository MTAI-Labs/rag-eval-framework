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
import re
from pathlib import Path
from typing import Any, Sequence

from rag_eval.adapters.base import RagAdapter, dig
from rag_eval.adapters.http import HttpError, get_json, post_json, post_multipart
from rag_eval.adapters.multipart import encode as encode_multipart
from rag_eval.types import RagTrace

# Verified against the live stack on rpgpu127 (query 8081 / ingest 8082) via
# each server's own openapi.json -- not guessed. The query server exposes
# /v1/{generate,search,health,configuration,embedding-profiles}; the ingest
# server exposes /collections, /documents, /status, /health.
DEFAULTS = {
    "generate_path": "/v1/generate",
    "search_path": "/v1/search",
    "health_path": "/v1/health",
    "config_path": "/v1/configuration",
    "profiles_path": "/v1/embedding-profiles",
    "collections_path": "/v1/collections",
    "create_collection_path": "/v1/collection",
    "documents_path": "/v1/documents",
    "status_path": "/v1/status",
    "answer_field": "choices.0.message.content",
    "citations_field": "citations.results",
    "chunks_field": "citations.results",
    "chunk_text_field": "content",
    "chunk_score_field": "score",
    "chunk_id_field": "document_id",
    # Where the ingestor puts the Hansard metadata. The live collections carry
    # session_date/dewan/parliament/penggal/mesyuarat/filename, plus page_number
    # in custom_metadata -- there is no ready-made sitting id, so it is derived
    # from dewan + session_date (see _derive_sitting). Several candidates are
    # tried in order so one build works across ingestion revisions.
    "sitting_fields": ["metadata.sitting_id", "metadata.source_id", "document_name", "source"],
    "page_fields": ["metadata.page_number", "content_metadata.page_number",
                    "metadata.page", "metadata.ms", "page_number"],
    "dewan_fields": ["metadata.dewan", "content_metadata.dewan", "dewan"],
    "session_date_fields": ["metadata.session_date", "content_metadata.session_date",
                            "session_date"],
    "filename_fields": ["metadata.filename", "content_metadata.filename", "filename"],
    "prompt_tokens_field": "usage.prompt_tokens",
    "completion_tokens_field": "usage.completion_tokens",
    "model_field": "model",
    # The query server's /v1/configuration is authoritative for what is actually
    # loaded; a collection's recorded metadata is a label that can go stale.
    "sparse_markers": ["sparse", "bm25"],
}

#: Chamber names as the ingestor records them, mapped to golden-set prefixes.
DEWAN_PREFIXES = {
    "dewan rakyat": "dr",
    "dewan negara": "dn",
    "kamar khas dewan rakyat": "kkdr",
    "kamar khas": "kkdr",
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
            options.get("collection") or os.environ.get("NVIDIA_RAG_COLLECTION", "parliament_hansard_eval")
        )
        # The ingest server (8082) is a separate service from the query server
        # (8081): collections live there, so collection existence and metadata
        # checks must not be aimed at the query server.
        self.ingest_base_url = str(
            options.get("ingest_base_url") or os.environ.get("NVIDIA_INGEST_BASE_URL", "")
        ).rstrip("/")
        self.api_key = os.environ.get(str(options.get("api_key_env", "NVIDIA_RAG_API_KEY")), "")
        self.timeout = float(options.get("timeout_s", 120.0))
        self.model = str(options.get("model", ""))
        # Which embedding profile this collection was built with. It is not
        # discoverable from the collection listing, so it is configuration --
        # but it changes the embedding model AND what the stack can do with the
        # collection, so it is recorded in the run environment either way.
        self.embedding_profile = str(
            options.get("embedding_profile")
            or os.environ.get("NVIDIA_RAG_EMBEDDING_PROFILE", "text")
        )

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
            "ingest_base_url": self.ingest_base_url,
            "collection": self.collection,
            "embedding_profile": self.embedding_profile,
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
                sitting_id=self._sitting_of(c),
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
        """Read what the stack is actually running, from /v1/configuration.

        That endpoint reports the models the server has loaded, which is the
        authoritative answer -- a collection's recorded ``embedding_model`` is a
        label written at ingest time and can be stale against what is serving.

        The LLM and the reranker are captured too, not just the embedder: a
        different generator moves every judge score, and toggling the reranker
        reorders retrieval. Both would otherwise surface as an unexplained
        run-over-run difference.
        """
        env: dict[str, Any] = {
            "base_url": self.base_url,
            "ingest_base_url": self.ingest_base_url,
            "collection": self.collection,
            "embedding_model": "unknown",
            "embedding_model_source": "unknown",
            "retrieval_mode": "unknown",
            "retrieval_mode_source": "not exposed by this API",
        }

        env["embedding_profile"] = self.embedding_profile
        profile = self._profile_info(self.embedding_profile)
        if profile:
            caps = profile.get("capabilities") or {}
            env["embedding_model"] = _model_from_label(profile.get("label", "")) or "unknown"
            env["embedding_model_source"] = f"profile '{self.embedding_profile}': {self.settings['profiles_path']}"
            env["embedding_dimensions"] = profile.get("dimensions")
            env["profile_supports_reranker"] = caps.get("reranker")
            env["profile_multimodal"] = caps.get("multimodal")

        path = str(self.settings.get("config_path") or "")
        if path:
            try:
                body = get_json(
                    f"{self.base_url}{path}", headers=self._headers(), timeout=15.0
                ) or {}
                models = body.get("models") or {}
                toggles = body.get("feature_toggles") or {}
                rag_cfg = body.get("rag_configuration") or {}
                if models.get("embedding_model") and env["embedding_model"] == "unknown":
                    env["embedding_model"] = models["embedding_model"]
                    env["embedding_model_source"] = f"served: {path}"
                env.update(
                    {
                        "llm_model": models.get("llm_model", "unknown"),
                        "reranker_model": models.get("reranker_model", "unknown"),
                        # Effective, not merely configured: the server may have
                        # the reranker on while this collection's embedding
                        # profile cannot use it, and a manifest claiming a
                        # reranker that never ran is a false provenance record.
                        "reranker_enabled": _effective_reranker(
                            toggles.get("enable_reranker"),
                            env.get("profile_supports_reranker"),
                        ),
                        "reranker_configured": toggles.get("enable_reranker"),
                        "query_rewriting": toggles.get("enable_query_rewriting"),
                        "vdb_top_k": rag_cfg.get("vdb_top_k"),
                        "reranker_top_k": rag_cfg.get("reranker_top_k"),
                    }
                )
            except Exception as exc:  # noqa: BLE001 - a probe never fails a run
                env["embedding_model_source"] = f"probe failed: {exc}"[:200]

        for key in ("retrieval_mode", "embedding_model"):
            declared = self.options.get(key)
            if declared and env[key] == "unknown":
                env[key] = str(declared)
                env[f"{key}_source"] = "declared in config (unverified)"
        return env

    def embedding_profiles(self) -> list[dict[str, Any]]:
        """Profiles the stack offers, with their capabilities."""
        try:
            body = get_json(
                f"{self.base_url}{self.settings['profiles_path']}",
                headers=self._headers(), timeout=15.0,
            ) or {}
        except Exception:  # noqa: BLE001
            return []
        return body.get("profiles", []) if isinstance(body, dict) else []

    def _profile_info(self, name: str) -> dict[str, Any] | None:
        return next((p for p in self.embedding_profiles() if p.get("name") == name), None)

    def list_collections(self) -> list[dict[str, Any]]:
        """Collections on the ingest server. Empty list if it cannot be reached."""
        if not self.ingest_base_url:
            return []
        try:
            body = get_json(
                f"{self.ingest_base_url}{self.settings['collections_path']}",
                headers=self._headers(), timeout=30.0,
            ) or {}
        except Exception:  # noqa: BLE001
            return []
        return body.get("collections", []) if isinstance(body, dict) else []

    # -- ingestion (writes) ------------------------------------------------
    # The vector DB will not accept documents for a collection that does not
    # exist: on this stack you POST /collection first -- declaring the metadata
    # schema -- and only then POST /documents. The schema is the important half.
    # Fields not declared here cannot be filtered on or returned as chunk
    # metadata later, which is precisely how page-citation accuracy stops being
    # computable (design spec §6).
    def create_collection(
        self,
        metadata_schema: list[dict[str, Any]],
        *,
        embedding_profile: str = "text",
        description: str = "",
        tags: Sequence[str] = (),
        created_by: str = "",
    ) -> dict[str, Any]:
        """Create the eval collection. Fails loudly if it already exists."""
        if not self.ingest_base_url:
            raise ValueError("NVIDIA_INGEST_BASE_URL is not set; cannot create a collection")
        payload = {
            "collection_name": self.collection,
            "metadata_schema": metadata_schema,
            "embedding_profile": embedding_profile,
            "description": description,
            "tags": list(tags),
            "created_by": created_by,
            "status": "Active",
        }
        return post_json(
            f"{self.ingest_base_url}{self.settings['create_collection_path']}",
            payload, headers=self._headers(), timeout=120.0,
        )

    def upload_documents(
        self,
        files: Sequence[tuple[Path, dict[str, Any]]],
        *,
        blocking: bool = False,
        timeout: float = 1800.0,
    ) -> dict[str, Any]:
        """Upload PDFs with per-document custom metadata.

        ``files`` is ``[(path, metadata), …]``; the metadata keys must match the
        collection's declared schema or the server rejects them.
        """
        if not self.ingest_base_url:
            raise ValueError("NVIDIA_INGEST_BASE_URL is not set; cannot upload documents")
        data = {
            "collection_name": self.collection,
            "blocking": blocking,
            "custom_metadata": [
                {"filename": path.name, "metadata": meta} for path, meta in files
            ],
        }
        body, ctype = encode_multipart(
            [("data", json.dumps(data))],
            [("documents", path) for path, _ in files],
        )
        return post_multipart(
            f"{self.ingest_base_url}{self.settings['documents_path']}",
            body, ctype, headers=self._headers(), timeout=timeout,
        )

    def list_documents(self) -> list[str]:
        """Document names already in the collection. Empty if it cannot be read.

        Uploading a PDF that is already there duplicates its chunks, which
        quietly inflates retrieval and makes a re-run non-reproducible.
        """
        if not self.ingest_base_url:
            return []
        try:
            body = get_json(
                f"{self.ingest_base_url}{self.settings['documents_path']}"
                f"?collection_name={self.collection}",
                headers=self._headers(), timeout=60.0,
            ) or {}
        except Exception:  # noqa: BLE001
            return []
        docs = body.get("documents", []) if isinstance(body, dict) else []
        return [
            d.get("document_name", "") if isinstance(d, dict) else str(d)
            for d in docs
        ]

    def ingestion_status(self, task_id: str = "") -> dict[str, Any]:
        """Poll a running ingestion task."""
        if not self.ingest_base_url:
            return {}
        url = f"{self.ingest_base_url}{self.settings['status_path']}"
        if task_id:
            url += f"?task_id={task_id}"
        try:
            return get_json(url, headers=self._headers(), timeout=30.0) or {}
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)[:200]}

    def collection_status(self) -> dict[str, Any]:
        """Whether the eval collection exists, and what it holds.

        ``ingest-check`` leans on this: a scorecard produced against a
        collection that was never created is not a bad score, it is no score.
        """
        if not self.ingest_base_url:
            return {"checked": False, "reason": "NVIDIA_INGEST_BASE_URL is not set"}
        collections = self.list_collections()
        if not collections:
            return {"checked": False, "reason": f"no collections listed at {self.ingest_base_url}"}
        found = next(
            (c for c in collections if c.get("collection_name") == self.collection), None
        )
        return {
            "checked": True,
            "exists": found is not None,
            "collection": self.collection,
            "entities": (found or {}).get("num_entities"),
            "metadata_fields": [f.get("name") for f in (found or {}).get("metadata_schema", [])],
            "total_collections": len(collections),
        }

    def _sitting_of(self, chunk: Any) -> Any:
        """The chunk's sitting id, declared or derived.

        Prefers an explicit ``sitting_id``. Failing that, rebuilds it from the
        chamber and the session date, which is how the existing Hansard
        collections on this stack record it -- ``dewan: "dewan rakyat"`` plus
        ``session_date: "2026-06-22"`` is the same fact as ``dr_2026-06-22``.
        Without this, a collection that carries chamber and date but no
        ``sitting_id`` would score zero on every retrieval metric despite
        holding everything needed to score properly.
        """
        explicit = _first(chunk, self.settings["sitting_fields"])
        if explicit:
            return explicit

        dewan = _first(chunk, self.settings.get("dewan_fields", []))
        date = _first(chunk, self.settings.get("session_date_fields", []))
        if not (dewan and date):
            return None
        prefix = DEWAN_PREFIXES.get(str(dewan).strip().lower())
        if not prefix:
            return None
        return f"{prefix}_{str(date).strip()}"

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


def _model_from_label(label: str) -> str:
    """``"VL (llama-nemotron-embed-vl-1b-v2)"`` -> the model id inside."""
    m = re.search(r"\(([^)]+)\)", label or "")
    return m.group(1).strip() if m else ""


def _effective_reranker(configured: Any, profile_supports: Any) -> Any:
    """The reranker only runs if the server enables it *and* the profile allows it."""
    if configured is None:
        return None
    if profile_supports is None:
        return configured
    return bool(configured) and bool(profile_supports)


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
