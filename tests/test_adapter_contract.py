"""The contract every RAG adapter must satisfy.

Parametrised over the registry, so a newly registered adapter is held to the
same standard automatically rather than by whoever remembers to write tests.

Everything runs against a real loopback HTTP service (``mock_rag_service``)
rather than patched transport, because half the contract is about behaviour
when the network misbehaves -- and that cannot be tested by replacing the call
that would have failed.
"""

from __future__ import annotations

import pytest
from mock_rag_service import DROP, EMPTY, HTTP_ERROR, MALFORMED, OK, SSE, MockRagService

from rag_eval.adapters import available, get_adapter
from rag_eval.types import RagTrace

#: Adapters that speak HTTP, and the env var each reads its base URL from.
HTTP_ADAPTERS = {
    "nvidia": "NVIDIA_RAG_BASE_URL",
    "tanyaparlimen": "TANYAPARLIMEN_BASE_URL",
}
QUESTION = "Who resigned as a Member of the Dewan Rakyat on 18 May 2026?"


@pytest.fixture
def service():
    with MockRagService(OK) as svc:
        yield svc


def build(name: str, service: MockRagService, monkeypatch, **options):
    """Point an adapter at the stub service using only its documented config."""
    monkeypatch.setenv(HTTP_ADAPTERS[name], service.base_url)
    return get_adapter(name, top_k=5, timeout_s=10, **options)


def test_every_registered_adapter_is_covered_here():
    # If someone registers an adapter and does not add it to HTTP_ADAPTERS or
    # the in-process list, this fails rather than silently skipping it.
    assert set(available()) == set(HTTP_ADAPTERS) | {"mock"}


# -- the shape of a successful trace ---------------------------------------
@pytest.mark.parametrize("name", sorted(HTTP_ADAPTERS))
def test_answer_returns_a_populated_trace(name, service, monkeypatch):
    trace = build(name, service, monkeypatch).answer(QUESTION, "tp-0001")

    assert isinstance(trace, RagTrace)
    assert trace.ok, trace.error
    assert trace.question == QUESTION
    assert trace.question_id == "tp-0001"
    assert trace.adapter == name
    assert trace.generated_answer
    assert trace.latency_ms is not None and trace.latency_ms >= 0


@pytest.mark.parametrize("name", sorted(HTTP_ADAPTERS))
def test_chunks_are_ranked_and_normalised(name, service, monkeypatch):
    trace = build(name, service, monkeypatch).answer(QUESTION, "tp-0001")
    chunks = trace.retrieved_chunks

    assert len(chunks) == 2
    assert [c.rank for c in chunks] == [1, 2]
    # Every adapter must resolve the sitting to the canonical form, whatever
    # shape its service reports it in — metrics join on this.
    assert [c.sitting_id for c in chunks] == ["dr_2026-06-22", "dn_2026-08-04"]
    assert [c.page for c in chunks] == [9, 15]
    assert all(c.text for c in chunks)
    assert all(isinstance(c.score, float) for c in chunks)


@pytest.mark.parametrize("name", sorted(HTTP_ADAPTERS))
def test_tokens_and_citations_are_captured(name, service, monkeypatch):
    trace = build(name, service, monkeypatch).answer(QUESTION, "tp-0001")

    assert trace.total_tokens == 940
    assert trace.cited_sources
    assert trace.cited_sources[0].sitting_id == "dr_2026-06-22"


# -- failure must be data, never an exception ------------------------------
@pytest.mark.parametrize("name", sorted(HTTP_ADAPTERS))
@pytest.mark.parametrize("mode", [HTTP_ERROR, MALFORMED, DROP])
def test_a_broken_service_yields_an_error_trace_not_a_crash(name, mode, service, monkeypatch):
    # One bad service must not end a 371-question run.
    adapter = build(name, service, monkeypatch)
    service.set_mode(mode)
    trace = adapter.answer(QUESTION, "tp-0001")

    assert isinstance(trace, RagTrace)
    assert not trace.ok
    assert trace.error and trace.question_id == "tp-0001"


@pytest.mark.parametrize("name", sorted(HTTP_ADAPTERS))
def test_an_empty_result_is_an_error_not_a_silent_success(name, service, monkeypatch):
    # A service that answers nothing is a failure to record, not a valid trace
    # with zero chunks that would quietly score as a retrieval miss.
    adapter = build(name, service, monkeypatch)
    service.set_mode(EMPTY)
    trace = adapter.answer(QUESTION, "tp-0001")

    assert not trace.ok
    assert trace.retrieved_chunks == []


@pytest.mark.parametrize("name", sorted(HTTP_ADAPTERS))
def test_an_unreachable_service_yields_an_error_trace(name, monkeypatch):
    with MockRagService(OK) as svc:
        dead_url = svc.base_url            # captured, then the server stops
    monkeypatch.setenv(HTTP_ADAPTERS[name], dead_url)
    trace = get_adapter(name, timeout_s=3).answer(QUESTION, "tp-0001")

    assert not trace.ok
    assert trace.error


# -- configuration and disclosure ------------------------------------------
@pytest.mark.parametrize("name", sorted(HTTP_ADAPTERS))
def test_configuration_comes_from_options_not_hardcoding(name, service, monkeypatch):
    monkeypatch.delenv(HTTP_ADAPTERS[name], raising=False)
    adapter = get_adapter(name, base_url=service.base_url, top_k=3, timeout_s=10)
    assert adapter.answer(QUESTION, "q1").ok


@pytest.mark.parametrize("name", sorted(HTTP_ADAPTERS))
def test_missing_configuration_fails_loudly_at_construction(name, monkeypatch):
    # Better to refuse to build than to run 371 questions against nothing.
    monkeypatch.delenv(HTTP_ADAPTERS[name], raising=False)
    with pytest.raises(ValueError, match="base_url"):
        get_adapter(name)


@pytest.mark.parametrize("name", sorted(HTTP_ADAPTERS))
def test_describe_never_leaks_a_secret(name, service, monkeypatch):
    import json
    monkeypatch.setenv("NVIDIA_RAG_API_KEY", "super-secret-key")
    monkeypatch.setenv("TANYAPARLIMEN_API_KEY", "super-secret-key")
    described = json.dumps(build(name, service, monkeypatch).describe())

    assert "super-secret-key" not in described
    assert service.base_url in described


@pytest.mark.parametrize("name", sorted(available()))
def test_health_reports_a_verdict_and_a_reason(name, service, monkeypatch):
    adapter = (get_adapter("mock") if name == "mock"
               else build(name, service, monkeypatch))
    healthy, message = adapter.health()

    assert isinstance(healthy, bool)
    assert isinstance(message, str) and message


@pytest.mark.parametrize("name", sorted(available()))
def test_adapters_are_context_managers(name, service, monkeypatch):
    adapter = (get_adapter("mock") if name == "mock"
               else build(name, service, monkeypatch))
    with adapter as entered:
        assert entered is adapter


# -- wire-format independence ----------------------------------------------
def test_the_nvidia_adapter_handles_both_json_and_sse(service, monkeypatch):
    # The same server, the same assertions, two wire formats — a server that
    # ignores stream=false must not change what the framework sees.
    adapter = build("nvidia", service, monkeypatch)
    plain = adapter.answer(QUESTION, "q1")
    service.set_mode(SSE)
    streamed = adapter.answer(QUESTION, "q1")

    assert plain.generated_answer == streamed.generated_answer
    assert [c.sitting_id for c in plain.retrieved_chunks] == \
           [c.sitting_id for c in streamed.retrieved_chunks]
    assert [c.page for c in plain.retrieved_chunks] == \
           [c.page for c in streamed.retrieved_chunks]


# -- the smoke command -----------------------------------------------------
def test_smoke_passes_against_a_working_service(service, monkeypatch, capsys):
    from rag_eval.cli.main import main
    monkeypatch.setenv("NVIDIA_RAG_BASE_URL", service.base_url)
    rc = main(["smoke", "--adapter", "nvidia", "--sample", "2", "--no-verify-checksum"])

    out = capsys.readouterr().out
    assert rc == 0
    assert "PASS" in out
    assert "carry sitting id + page" in out


def test_smoke_fails_when_every_question_fails(service, monkeypatch, capsys):
    # The point of a smoke test: exit non-zero so it can gate a real run.
    from rag_eval.cli.main import main
    monkeypatch.setenv("NVIDIA_RAG_BASE_URL", service.base_url)
    service.set_mode(HTTP_ERROR)
    rc = main(["smoke", "--adapter", "nvidia", "--sample", "2", "--no-verify-checksum"])

    assert rc == 1
    assert "FAIL" in capsys.readouterr().err


def test_smoke_spreads_across_sittings(service, monkeypatch, capsys):
    # Five questions from one sitting could pass while the corpus is broken.
    from rag_eval.cli.main import main
    monkeypatch.setenv("NVIDIA_RAG_BASE_URL", service.base_url)
    main(["smoke", "--adapter", "nvidia", "--sample", "4", "--no-verify-checksum"])

    golds = [ln.split("gold=")[1].split()[0]
             for ln in capsys.readouterr().out.splitlines() if "gold=" in ln]
    assert len(golds) == len(set(golds)) == 4
