"""The ``run`` stage: push every golden question through an adapter.

Traces are appended to disk as they complete, so a 371-question run against a
slow GPU stack can be interrupted and resumed rather than restarted, and a
crash still leaves the work already paid for.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Sequence

from rag_eval.adapters.base import RagAdapter
from rag_eval.store import RunStore
from rag_eval.types import GoldenItem, RagTrace

ProgressFn = Callable[[int, int, RagTrace], None]


def run_dataset(
    adapter: RagAdapter,
    items: Sequence[GoldenItem],
    store: RunStore,
    *,
    concurrency: int = 1,
    on_progress: ProgressFn | None = None,
    resume: bool = False,
) -> list[RagTrace]:
    """Run ``items`` through ``adapter``, appending each trace to ``store``.

    ``concurrency`` above 1 fans out across questions; keep it modest against a
    shared GPU stack, because a queued request inflates the latency numbers the
    ops block reports.
    """
    done: dict[str, RagTrace] = {}
    if resume:
        done = {t.question_id: t for t in store.read_traces()}
        items = [i for i in items if i.id not in done]

    total = len(items)
    results: list[RagTrace] = []

    if concurrency <= 1:
        for index, item in enumerate(items, 1):
            trace = _answer_one(adapter, item)
            store.append_trace(trace)
            results.append(trace)
            if on_progress:
                on_progress(index, total, trace)
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(_answer_one, adapter, item): item for item in items}
            for index, future in enumerate(as_completed(futures), 1):
                trace = future.result()
                store.append_trace(trace)
                results.append(trace)
                if on_progress:
                    on_progress(index, total, trace)

    return list(done.values()) + results


def _answer_one(adapter: RagAdapter, item: GoldenItem) -> RagTrace:
    """Never let one bad question end a run; record the failure as the trace."""
    try:
        trace = adapter.answer(item.question, item.id)
    except Exception as exc:  # noqa: BLE001 - recorded as an error trace
        return RagTrace(
            question_id=item.id,
            question=item.question,
            adapter=adapter.name,
            error=f"{type(exc).__name__}: {exc}",
        )
    trace.question_id = trace.question_id or item.id
    return trace


def ingest_check(
    adapter: RagAdapter,
    items: Sequence[GoldenItem],
    *,
    sample: int = 10,
    with_answer: bool = False,
    on_probe=None,
) -> dict[str, object]:
    """Verify the ingested collection can support the retrieval metrics.

    Design spec §6: page numbers must map reliably to chunk metadata, and if
    they do not, ``page_citation_accuracy`` is not computable at all. This is
    the check to run on 3-5 PDFs before a full ingest, and again before anyone
    reads a scorecard.

    Sampling deliberately spreads across *sittings*, not rows: a hundred
    questions from one well-ingested sitting proves nothing about the other 15.

    Retrieval-only by default. This check asks whether chunks carry usable
    metadata, and never looks at the generated answer -- so paying ~130s per
    probe to make the LLM write one it discards is waste. ``with_answer``
    exercises the full generate path when citation parsing is what is in doubt.
    """
    probe_fn = (
        adapter.answer
        if with_answer or not hasattr(adapter, "retrieve")
        else adapter.retrieve
    )
    by_sitting: dict[str, list[GoldenItem]] = {}
    for item in items:
        if item.reference:
            by_sitting.setdefault(item.reference.sitting_id, []).append(item)

    probes: list[GoldenItem] = []
    round_index = 0
    while len(probes) < sample and by_sitting:
        added = False
        for sitting in sorted(by_sitting):
            bucket = by_sitting[sitting]
            if round_index < len(bucket) and len(probes) < sample:
                probes.append(bucket[round_index])
                added = True
        if not added:
            break
        round_index += 1

    healthy, message = adapter.health()
    # Before asking how good retrieval is, ask whether there is anything to
    # retrieve from. A scorecard against a collection that was never created is
    # not a bad score -- it is no score.
    collection = (
        adapter.collection_status() if hasattr(adapter, "collection_status") else {"checked": False}
    )
    report: dict[str, object] = {
        "adapter": adapter.name,
        "reachable": healthy,
        "health": message,
        "collection_status": collection,
        "golden_sittings": sorted(by_sitting),
        "probes": len(probes),
        "mode": "answer" if probe_fn is adapter.answer else "retrieval-only",
        "probe_results": [],
    }

    chunks_total = with_sitting = with_page = 0
    seen_sittings: set[str] = set()

    for index, item in enumerate(probes, 1):
        try:
            trace = probe_fn(item.question, item.id)
        except Exception as exc:  # noqa: BLE001
            trace = RagTrace(question_id=item.id, question=item.question,
                             adapter=adapter.name, error=f"{type(exc).__name__}: {exc}")
        if on_probe:
            on_probe(index, len(probes), item, trace)
        chunks = trace.retrieved_chunks
        chunks_total += len(chunks)
        with_sitting += sum(1 for c in chunks if c.sitting_id)
        with_page += sum(1 for c in chunks if c.page is not None)
        seen_sittings.update(c.sitting_id for c in chunks if c.sitting_id)
        report["probe_results"].append(  # type: ignore[union-attr]
            {
                "question_id": item.id,
                "gold_sitting": item.reference.sitting_id if item.reference else None,
                "error": trace.error,
                "chunks": len(chunks),
                "with_sitting_id": sum(1 for c in chunks if c.sitting_id),
                "with_page": sum(1 for c in chunks if c.page is not None),
                "retrieved_sittings": sorted({c.sitting_id for c in chunks if c.sitting_id}),
                "hit_gold_sitting": bool(
                    item.reference
                    and any(c.sitting_id == item.reference.sitting_id for c in chunks)
                ),
            }
        )

    missing = sorted(set(by_sitting) - seen_sittings)
    report.update(
        {
            "chunks_retrieved": chunks_total,
            "sitting_id_coverage": round(with_sitting / chunks_total, 4) if chunks_total else None,
            "page_coverage": round(with_page / chunks_total, 4) if chunks_total else None,
            "sittings_seen": sorted(seen_sittings),
            "sittings_not_seen_in_probe": missing,
            "page_citation_accuracy_computable": bool(chunks_total) and with_page > 0,
            "verdict": _ingest_verdict(
                healthy, chunks_total, with_sitting, with_page, collection
            ),
        }
    )
    return report


def _ingest_verdict(
    healthy: bool,
    chunks: int,
    with_sitting: int,
    with_page: int,
    collection: dict[str, object] | None = None,
) -> str:
    if not healthy:
        return "FAIL: RAG service unreachable"
    collection = collection or {}
    if collection.get("checked") and not collection.get("exists"):
        return (
            f"FAIL: collection {collection.get('collection')!r} does not exist on the "
            f"ingest server ({collection.get('total_collections')} collections present) "
            f"— nothing has been ingested to evaluate against"
        )
    if chunks == 0:
        return "FAIL: no chunks retrieved — is the eval collection ingested?"
    if with_sitting == 0:
        return "FAIL: no chunk carries a sitting id — retrieval metrics are not computable"
    if with_page == 0:
        return "FAIL: no chunk carries a page number — page-citation accuracy is not computable"
    if with_sitting < chunks or with_page < chunks:
        return (
            f"WARN: metadata is partial ({with_sitting}/{chunks} sitting ids, "
            f"{with_page}/{chunks} pages) — metrics will understate retrieval quality"
        )
    return "PASS: every retrieved chunk carries sitting id and page"
