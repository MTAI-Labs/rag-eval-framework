"""Adapter contract tests, run against a mock RAG service.

These are the tests a new adapter must pass; they check the contract the rest
of the framework relies on rather than any one service's wire format.
"""

import json

import pytest

from rag_eval.adapters import available, get_adapter
from rag_eval.adapters.base import AdapterError, RagAdapter, dig
from rag_eval.adapters.nvidia import NvidiaRagAdapter
from rag_eval.adapters.tanyaparlimen import TanyaParlimenAdapter
from rag_eval.types import RagTrace


def test_registry_exposes_every_shipped_adapter():
    assert {"mock", "nvidia", "tanyaparlimen"} <= set(available())
    with pytest.raises(AdapterError, match="unknown adapter"):
        get_adapter("does-not-exist")


def test_mock_adapter_satisfies_the_contract():
    adapter = get_adapter("mock", top_k=3)
    trace = adapter.answer("Who resigned?", "q1")

    assert isinstance(trace, RagTrace)
    assert trace.question_id == "q1"
    assert trace.adapter == "mock"
    assert len(trace.retrieved_chunks) == 3
    assert [c.rank for c in trace.retrieved_chunks] == [1, 2, 3]
    assert trace.ok


def test_adapters_replay_fixture_traces(tmp_path):
    fixture = tmp_path / "traces.jsonl"
    fixture.write_text(json.dumps({
        "question_id": "q1",
        "question": "Who resigned?",
        "adapter": "mock",
        "generated_answer": "Two members did.",
        "retrieved_chunks": [
            {"rank": 1, "text": "…", "sitting_id": "dr_2026-06-22", "page": 3}
        ],
        "cited_sources": [
            {"doc_type": "dr", "sitting_date": "2026-06-22", "pages": [3], "raw": ""}
        ],
        "latency_ms": 42.0,
    }) + "\n", encoding="utf-8")

    trace = get_adapter("mock", fixture=str(fixture)).answer("Who resigned?", "q1")
    assert trace.generated_answer == "Two members did."
    assert trace.retrieved_chunks[0].sitting_id == "dr_2026-06-22"
    assert trace.cited_sources[0].pages == (3,)


def test_scripted_failures_are_errors_not_exceptions():
    trace = get_adapter("mock", fail_ids=["q1"]).answer("anything", "q1")
    assert not trace.ok
    assert trace.error


def test_chunk_metadata_is_normalised_by_the_base_class():
    # An ingestor that emits "KKDR_2026-7-14.pdf" and "ms. 12" must still score.
    chunk = RagAdapter.make_chunk(1, "text", sitting_id="KKDR_2026-7-14.pdf", page="ms. 12")
    assert chunk.sitting_id == "kkdr_2026-07-14"
    assert chunk.page == 12

    assert RagAdapter.make_chunk(1, "t", sitting_id=None, page=None).page is None
    assert RagAdapter.make_chunk(1, "t", page="no page here").page is None


def test_citations_fall_back_to_the_retrieved_chunks():
    chunks = [
        RagAdapter.make_chunk(1, "a", sitting_id="dr_2026-06-22", page=3),
        RagAdapter.make_chunk(2, "b", sitting_id="dr_2026-06-22", page=3),  # duplicate
        RagAdapter.make_chunk(3, "c", sitting_id=None, page=None),          # unusable
    ]
    refs = RagAdapter.citations_from_chunks(chunks)
    assert [(r.sitting_id, r.pages) for r in refs] == [("dr_2026-06-22", (3,))]


def test_unparseable_citations_are_dropped_not_fatal():
    refs = RagAdapter.parse_citations(["dr_2026-06-22, ms. 3", "the blue folder", ""])
    assert [r.sitting_id for r in refs] == ["dr_2026-06-22"]


def test_dotted_field_paths_survive_a_missing_key():
    body = {"choices": [{"message": {"content": "hello"}}]}
    assert dig(body, "choices.0.message.content") == "hello"
    assert dig(body, "choices.9.message.content", "fallback") == "fallback"
    assert dig(body, "usage.prompt_tokens") is None


def test_nvidia_adapter_maps_a_blueprint_response(monkeypatch):
    monkeypatch.setenv("NVIDIA_RAG_BASE_URL", "http://rtx6000.test:8081")
    adapter = NvidiaRagAdapter(collection="parliament-hansard-eval", top_k=2)

    body = {
        "choices": [{"message": {"content": "3,404 trainees."}}],
        "citations": {"results": [
            {"content": "…jumlah 3,404…", "score": 0.81, "document_id": "d1",
             "metadata": {"sitting_id": "dr_2026-6-22", "page_number": 9}},
            {"content": "…", "score": 0.42, "document_id": "d2",
             "metadata": {"sitting_id": "dn_2026-08-04", "page_number": 15}},
        ]},
        "usage": {"prompt_tokens": 900, "completion_tokens": 40},
        "model": "llama-3.3-70b",
    }
    trace = adapter._populate(adapter._trace("How many trainees?", "q1"), body)

    assert trace.generated_answer == "3,404 trainees."
    assert trace.model == "llama-3.3-70b"
    assert trace.total_tokens == 940
    assert [c.sitting_id for c in trace.retrieved_chunks] == ["dr_2026-06-22", "dn_2026-08-04"]
    assert trace.retrieved_chunks[0].page == 9
    assert trace.cited_sources[0].sitting_id == "dr_2026-06-22"


def test_nvidia_adapter_reports_lost_metadata_rather_than_guessing(monkeypatch):
    monkeypatch.setenv("NVIDIA_RAG_BASE_URL", "http://rtx6000.test:8081")
    adapter = NvidiaRagAdapter()
    body = {"choices": [{"message": {"content": "an answer"}}],
            "citations": {"results": [{"content": "…", "metadata": {}}]}}
    trace = adapter._populate(adapter._trace("q?", "q1"), body)

    coverage = adapter.metadata_coverage(trace)
    assert coverage == {"chunks": 1, "with_sitting_id": 0, "with_page": 0, "sittings": []}


def test_nvidia_adapter_needs_a_base_url(monkeypatch):
    monkeypatch.delenv("NVIDIA_RAG_BASE_URL", raising=False)
    with pytest.raises(ValueError, match="base_url"):
        NvidiaRagAdapter()


def test_tanyaparlimen_adapter_describes_itself_without_secrets(monkeypatch):
    monkeypatch.setenv("TANYAPARLIMEN_BASE_URL", "https://api.test")
    monkeypatch.setenv("TANYAPARLIMEN_API_KEY", "super-secret")
    described = json.dumps(TanyaParlimenAdapter(top_k=4).describe())

    assert "super-secret" not in described
    assert "api.test" in described
