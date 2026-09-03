"""Adapter registry: one entry per RAG service, selected by ``--adapter``."""

from __future__ import annotations

from typing import Any

from rag_eval.adapters.base import AdapterError, RagAdapter
from rag_eval.adapters.mock import MockAdapter
from rag_eval.adapters.nvidia import NvidiaRagAdapter
from rag_eval.adapters.tanyaparlimen import TanyaParlimenAdapter

_REGISTRY: dict[str, type[RagAdapter]] = {
    MockAdapter.name: MockAdapter,
    NvidiaRagAdapter.name: NvidiaRagAdapter,
    TanyaParlimenAdapter.name: TanyaParlimenAdapter,
}


def register(adapter_cls: type[RagAdapter]) -> type[RagAdapter]:
    """Register a new RAG service. Usable as a class decorator."""
    _REGISTRY[adapter_cls.name] = adapter_cls
    return adapter_cls


def available() -> list[str]:
    return sorted(_REGISTRY)


def get_adapter(name: str, **options: Any) -> RagAdapter:
    try:
        adapter_cls = _REGISTRY[name]
    except KeyError:
        raise AdapterError(
            f"unknown adapter {name!r}; available: {', '.join(available())}"
        ) from None
    return adapter_cls(**options)


__all__ = [
    "AdapterError",
    "RagAdapter",
    "MockAdapter",
    "NvidiaRagAdapter",
    "TanyaParlimenAdapter",
    "register",
    "available",
    "get_adapter",
]
