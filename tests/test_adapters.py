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
    adapter = NvidiaRagAdapter(collection="parliament_hansard_eval", top_k=2)

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


def test_probe_environment_defaults_to_unknown_not_a_guess(monkeypatch):
    # An unreachable stack must yield "unknown", never a plausible-looking
    # value: a manifest that guesses is worse than one that admits it.
    monkeypatch.setenv("NVIDIA_RAG_BASE_URL", "http://unreachable.invalid:8081")
    env = NvidiaRagAdapter().probe_environment()

    assert env["embedding_model"] == "unknown"
    assert env["retrieval_mode"] == "unknown"
    assert "probe failed" in env["embedding_model_source"]


def test_a_declared_value_is_recorded_as_unverified(monkeypatch):
    monkeypatch.setenv("NVIDIA_RAG_BASE_URL", "http://unreachable.invalid:8081")
    env = NvidiaRagAdapter(retrieval_mode="hybrid").probe_environment()

    assert env["retrieval_mode"] == "hybrid"
    assert env["retrieval_mode_source"] == "declared in config (unverified)"


def test_the_served_config_wins_over_a_declared_model(monkeypatch):
    # /v1/configuration is evidence; a config entry is a claim. A collection's
    # recorded embedding_model is a label written at ingest time and can be
    # stale against what the server actually loaded, so it must never silently
    # become the manifest's answer.
    monkeypatch.setenv("NVIDIA_RAG_BASE_URL", "http://rtx6000.test:8081")
    import rag_eval.adapters.nvidia as nv
    monkeypatch.setattr(nv, "get_json", lambda url, **kw: {
        "models": {"embedding_model": "nvidia/llama-nemotron-embed-1b-v2",
                   "llm_model": "google/gemma-4-31B-it",
                   "reranker_model": "nvidia/llama-nemotron-rerank-1b-v2"},
        "feature_toggles": {"enable_reranker": True, "enable_query_rewriting": False},
        "rag_configuration": {"vdb_top_k": 100, "reranker_top_k": 10},
    })
    env = nv.NvidiaRagAdapter(
        embedding_model="nvidia/llama-3.2-nv-embedqa-1b-v2"
    ).probe_environment()

    assert env["embedding_model"] == "nvidia/llama-nemotron-embed-1b-v2"
    assert env["embedding_model_source"] == "served: /v1/configuration"


def test_probe_captures_the_generator_and_reranker_too(monkeypatch):
    # A different LLM moves every judge score; toggling the reranker reorders
    # retrieval. Neither is the embedder, and both invalidate a comparison.
    monkeypatch.setenv("NVIDIA_RAG_BASE_URL", "http://rtx6000.test:8081")
    import rag_eval.adapters.nvidia as nv
    monkeypatch.setattr(nv, "get_json", lambda url, **kw: {
        "models": {"embedding_model": "e", "llm_model": "google/gemma-4-31B-it",
                   "reranker_model": "r"},
        "feature_toggles": {"enable_reranker": False},
        "rag_configuration": {"vdb_top_k": 100, "reranker_top_k": 10},
    })
    env = nv.NvidiaRagAdapter().probe_environment()

    assert env["llm_model"] == "google/gemma-4-31B-it"
    assert env["reranker_enabled"] is False
    assert env["vdb_top_k"] == 100


def test_retrieval_mode_is_reported_unknown_when_the_api_cannot_say(monkeypatch):
    # The live ingest API exposes no vector-field or index information, so
    # dense-vs-hybrid is not probeable. Saying "unknown" is the honest answer;
    # inferring one would put a guess where the diff expects evidence.
    monkeypatch.setenv("NVIDIA_RAG_BASE_URL", "http://rtx6000.test:8081")
    import rag_eval.adapters.nvidia as nv
    monkeypatch.setattr(nv, "get_json", lambda url, **kw: {"models": {}})
    env = nv.NvidiaRagAdapter().probe_environment()

    assert env["retrieval_mode"] == "unknown"
    assert env["retrieval_mode_source"] == "not exposed by this API"


def test_collection_status_reports_a_missing_collection(monkeypatch):
    # The failure that matters most: evaluating against a collection nobody
    # ever created. That is not a bad score, it is no score.
    monkeypatch.setenv("NVIDIA_RAG_BASE_URL", "http://rtx6000.test:8081")
    monkeypatch.setenv("NVIDIA_INGEST_BASE_URL", "http://rtx6000.test:8082")
    import rag_eval.adapters.nvidia as nv
    monkeypatch.setattr(nv, "get_json", lambda url, **kw: {
        "collections": [{"collection_name": "dr_20260121", "num_entities": 160,
                         "metadata_schema": [{"name": "session_date"}]}]
    })
    status = nv.NvidiaRagAdapter(collection="parliament_hansard_eval").collection_status()

    assert status["checked"] is True
    assert status["exists"] is False
    assert status["total_collections"] == 1


def test_collection_status_reports_a_present_collection(monkeypatch):
    monkeypatch.setenv("NVIDIA_RAG_BASE_URL", "http://rtx6000.test:8081")
    monkeypatch.setenv("NVIDIA_INGEST_BASE_URL", "http://rtx6000.test:8082")
    import rag_eval.adapters.nvidia as nv
    monkeypatch.setattr(nv, "get_json", lambda url, **kw: {
        "collections": [{"collection_name": "parliament_hansard_eval", "num_entities": 4200,
                         "metadata_schema": [{"name": "session_date"}, {"name": "dewan"}]}]
    })
    status = nv.NvidiaRagAdapter(collection="parliament_hansard_eval").collection_status()

    assert status["exists"] is True
    assert status["entities"] == 4200
    assert status["metadata_fields"] == ["session_date", "dewan"]


def test_collection_status_is_skipped_without_an_ingest_url(monkeypatch):
    monkeypatch.setenv("NVIDIA_RAG_BASE_URL", "http://rtx6000.test:8081")
    monkeypatch.delenv("NVIDIA_INGEST_BASE_URL", raising=False)
    status = NvidiaRagAdapter().collection_status()

    assert status["checked"] is False
    assert "NVIDIA_INGEST_BASE_URL" in status["reason"]


def test_sitting_id_is_derived_from_chamber_and_date(monkeypatch):
    # The existing Hansard collections on the stack record dewan + session_date
    # and no sitting_id. Without derivation those chunks match nothing and every
    # retrieval metric reads zero despite the collection holding what it needs.
    monkeypatch.setenv("NVIDIA_RAG_BASE_URL", "http://rtx6000.test:8081")
    adapter = NvidiaRagAdapter()

    assert adapter._sitting_of(
        {"metadata": {"dewan": "dewan rakyat", "session_date": "2026-06-22"}}
    ) == "dr_2026-06-22"
    assert adapter._sitting_of(
        {"metadata": {"dewan": "Dewan Negara", "session_date": "2026-08-04"}}
    ) == "dn_2026-08-04"
    assert adapter._sitting_of(
        {"metadata": {"dewan": "kamar khas dewan rakyat", "session_date": "2026-07-14"}}
    ) == "kkdr_2026-07-14"


def test_an_explicit_sitting_id_wins_over_derivation(monkeypatch):
    monkeypatch.setenv("NVIDIA_RAG_BASE_URL", "http://rtx6000.test:8081")
    chunk = {"metadata": {"sitting_id": "dr_2026-06-22",
                          "dewan": "dewan negara", "session_date": "1999-01-01"}}
    assert NvidiaRagAdapter()._sitting_of(chunk) == "dr_2026-06-22"


def test_derivation_refuses_to_guess(monkeypatch):
    # Half the facts, or an unrecognised chamber, must yield None rather than a
    # plausible-looking sitting id that silently matches the wrong document.
    monkeypatch.setenv("NVIDIA_RAG_BASE_URL", "http://rtx6000.test:8081")
    adapter = NvidiaRagAdapter()

    assert adapter._sitting_of({"metadata": {"session_date": "2026-06-22"}}) is None
    assert adapter._sitting_of({"metadata": {"dewan": "dewan rakyat"}}) is None
    assert adapter._sitting_of(
        {"metadata": {"dewan": "senate", "session_date": "2026-01-01"}}
    ) is None


def test_derived_sittings_flow_through_to_the_trace(monkeypatch):
    monkeypatch.setenv("NVIDIA_RAG_BASE_URL", "http://rtx6000.test:8081")
    adapter = NvidiaRagAdapter()
    body = {
        "choices": [{"message": {"content": "an answer"}}],
        "citations": {"results": [
            {"content": "…", "metadata": {"dewan": "dewan rakyat",
                                          "session_date": "2026-06-22",
                                          "page_number": 15}},
        ]},
    }
    trace = adapter._populate(adapter._trace("q?", "q1"), body)

    assert trace.retrieved_chunks[0].sitting_id == "dr_2026-06-22"
    assert trace.retrieved_chunks[0].page == 15
    assert trace.cited_sources[0].sitting_id == "dr_2026-06-22"


# -- embedding profile ------------------------------------------------------
PROFILES = {"default": "text", "profiles": [
    {"name": "text", "label": "Text (llama-nemotron-embed-1b-v2)", "dimensions": 2048,
     "capabilities": {"multimodal": False, "summary": True, "reranker": True,
                      "formats": ["pdf", "docx"]}},
    {"name": "vl", "label": "VL (llama-nemotron-embed-vl-1b-v2)", "dimensions": 2048,
     "capabilities": {"multimodal": True, "summary": False, "reranker": False,
                      "formats": ["pdf"]}},
]}
CONFIG = {"models": {"embedding_model": "nvidia/llama-nemotron-embed-1b-v2",
                     "llm_model": "google/gemma-4-31B-it"},
          "feature_toggles": {"enable_reranker": True},
          "rag_configuration": {"vdb_top_k": 100, "reranker_top_k": 10}}


def _routed(url, **kw):
    return PROFILES if "embedding-profiles" in url else CONFIG


def test_the_profile_determines_the_embedding_model(monkeypatch):
    # /v1/configuration reports the *text* embedder regardless, so a vl
    # collection would be recorded under the wrong model without this.
    monkeypatch.setenv("NVIDIA_RAG_BASE_URL", "http://rtx6000.test:8081")
    import rag_eval.adapters.nvidia as nv
    monkeypatch.setattr(nv, "get_json", _routed)

    env = nv.NvidiaRagAdapter(embedding_profile="vl").probe_environment()
    assert env["embedding_model"] == "llama-nemotron-embed-vl-1b-v2"
    assert env["embedding_profile"] == "vl"
    assert env["profile_multimodal"] is True


def test_vl_reports_the_reranker_as_effectively_off(monkeypatch):
    # The server has reranking on, but the vl profile cannot use it. Recording
    # "reranker_enabled: true" would be a false provenance record.
    monkeypatch.setenv("NVIDIA_RAG_BASE_URL", "http://rtx6000.test:8081")
    import rag_eval.adapters.nvidia as nv
    monkeypatch.setattr(nv, "get_json", _routed)

    vl = nv.NvidiaRagAdapter(embedding_profile="vl").probe_environment()
    assert vl["reranker_configured"] is True
    assert vl["reranker_enabled"] is False

    text = nv.NvidiaRagAdapter(embedding_profile="text").probe_environment()
    assert text["reranker_enabled"] is True


def test_the_profile_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("NVIDIA_RAG_BASE_URL", "http://rtx6000.test:8081")
    monkeypatch.setenv("NVIDIA_RAG_EMBEDDING_PROFILE", "vl")
    assert NvidiaRagAdapter().embedding_profile == "vl"
    assert NvidiaRagAdapter(embedding_profile="text").embedding_profile == "text"
