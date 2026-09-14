"""``rag-eval`` — the eval harness CLI.

Stages are deliberately separate commands over one run directory:

    rag-eval convert-golden       Excel  -> versioned golden_v1.jsonl (+ manifest)
    rag-eval corpus-check         are the Hansard PDFs the ones we checksummed?
    rag-eval page-offsets         measure printed-ms. vs physical-page offsets
    rag-eval ingest               create the eval collection, then upload the PDFs
    rag-eval ingest-check         is the collection's metadata good enough to score?
    rag-eval run                  golden set -> traces.jsonl
    rag-eval judge                traces    -> judgments.jsonl
    rag-eval score                both      -> scorecard.json
    rag-eval report               scorecard -> scorecard.html (+ regression diff)
    rag-eval eval                 run + judge + score + report in one shot

Splitting them matters in practice: judging is the expensive stage, so a
metric bug should be fixable with ``score`` alone, and a judge-prompt change
should be re-runnable without paying for retrieval again.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from rag_eval import __version__
from rag_eval.adapters import available, get_adapter
from rag_eval.config import Config, load_config, load_dotenv
from rag_eval.ingest import build_plan, execute as execute_ingest, reconcile
from rag_eval.dataset.pageoffsets import measure_corpus, offsets_payload, read_verified
from rag_eval.dataset.convert import convert_workbook
from rag_eval.dataset.corpus import (
    DEFAULT_CORPUS_DIR,
    CorpusError,
    build_manifest,
    coverage,
    manifest_path_for,
    read_corpus_manifest,
    verify_corpus,
    write_manifest,
)
from rag_eval.dataset.loader import (
    DatasetError,
    dataset_checksum,
    load_golden_set,
    read_manifest,
)
from rag_eval.judges.calibration import (
    CalibrationSample,
    agreement_report,
    append_calibration,
    load_calibration,
    samples_from_adjudications,
)
from rag_eval.judges.client import build_client
from rag_eval.judges.panel import JudgePanel, PanelUsage, flagged_rows
from rag_eval.metrics.scorecard import build_scorecard
from rag_eval.reporting import compare, write_report
from rag_eval.runner import ingest_check, run_dataset
from rag_eval.store import RunManifest, RunStore, list_runs, previous_run
from rag_eval.types import GoldenItem, RagTrace

DEFAULT_DATASET = "datasets/golden_v1.jsonl"


# ---------------------------------------------------------------- helpers
def _out(message: str = "") -> None:
    print(message, file=sys.stdout, flush=True)


def _err(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _load_items(args: argparse.Namespace) -> list[GoldenItem]:
    return load_golden_set(
        args.dataset,
        verify_checksum=not args.no_verify_checksum,
        limit=args.limit,
        ids=args.ids.split(",") if getattr(args, "ids", None) else None,
    )


def _adapter_options(config: Config, args: argparse.Namespace) -> dict[str, Any]:
    options = config.adapter_options(args.adapter)
    for pair in getattr(args, "option", None) or []:
        key, _, value = pair.partition("=")
        options[key.strip()] = _coerce(value.strip())
    if getattr(args, "top_k", None):
        options["top_k"] = args.top_k
    return options


def _coerce(value: str) -> Any:
    for caster in (int, float):
        try:
            return caster(value)
        except ValueError:
            pass
    lowered = value.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    return value


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _resolve_store(config: Config, run_id: str | None) -> RunStore:
    if run_id:
        return RunStore.open(config.runs_dir, run_id)
    runs = list_runs(config.runs_dir)
    if not runs:
        raise SystemExit("no runs found; start with 'rag-eval run'")
    return runs[-1]


def _judge_panel(config: Config, offline_fixture: str | None = None) -> JudgePanel:
    """The real gateway panel, or a scripted one for offline smoke runs.

    The fixture is either one rubric object (every judge replies the same, so
    nothing ever splits) or a ``{model_id: rubric}`` map, which is how a split
    and the adjudication flow get exercised without a gateway.
    """
    if not offline_fixture:
        return JudgePanel(build_client(config.judge), config.judge)

    from rag_eval.judges.client import ScriptedChatClient

    raw = Path(offline_fixture).read_text(encoding="utf-8")
    payload = json.loads(raw)
    model_ids = {m.id for m in config.judge.models}
    if isinstance(payload, dict) and set(payload) & model_ids:
        responses = {
            model: json.dumps(reply) for model, reply in payload.items() if model in model_ids
        }
        missing = model_ids - set(responses)
        if missing:
            raise ValueError(
                f"offline judge fixture has no reply for: {', '.join(sorted(missing))}"
            )
        return JudgePanel(ScriptedChatClient(responses), config.judge)
    return JudgePanel(ScriptedChatClient({}, default=raw), config.judge)


# ---------------------------------------------------------------- commands
def cmd_convert_golden(args: argparse.Namespace, config: Config) -> int:
    items, report, manifest = convert_workbook(
        args.xlsx, args.output, sheet=args.sheet, strict_refs=args.strict
    )
    _out(f"Converted {report.converted} of {report.total_rows} row(s) -> {args.output}")
    _out(f"  sha256 {manifest.sha256[:16]}…  ({len(report.sittings)} sittings)")
    if report.skipped_empty:
        _out(f"  skipped {report.skipped_empty} row(s) with no question or answer")
    if report.missing_reference:
        _out(
            f"  {len(report.missing_reference)} row(s) have no reference and are excluded from "
            f"retrieval metrics: {', '.join(report.missing_reference[:10])}"
        )
    if report.unparsed_reference:
        _out(f"  {len(report.unparsed_reference)} reference(s) could not be parsed:")
        for item_id, raw, error in report.unparsed_reference[:10]:
            _out(f"    {item_id}: {raw!r} — {error}")
    if report.duplicate_numbers:
        _out(
            f"  {len(report.duplicate_numbers)} duplicate 'No' value(s) in the workbook, "
            f"disambiguated by Excel row (fix the sheet so ids stay stable):"
        )
        for number, row_no in report.duplicate_numbers:
            _out(f"    No {number} reused at Excel row {row_no} -> id tp-{int(number):04d}-r{row_no}")
    if report.aliased_doc_types:
        _out(
            f"  {len(report.aliased_doc_types)} reference(s) used a known-typo prefix and were "
            f"corrected (see datasets/README.md):"
        )
        for item_id, raw, sitting in report.aliased_doc_types:
            _out(f"    {item_id}: {raw!r} -> {sitting}")
    if report.unknown_doc_types:
        _out(
            f"  unknown document prefix(es): {report.unknown_doc_types} "
            f"— confirm the sitting type before ingest (design spec §6)"
        )
    if report.multi_page_refs:
        _out(f"  {report.multi_page_refs} reference(s) span multiple pages")

    _out("\nSittings referenced by the golden set:")
    for sitting, count in sorted(report.sittings.items()):
        _out(f"  {sitting:<22} {count:>4} question(s)")

    if args.report:
        Path(args.report).write_text(
            json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        _out(f"\nConversion report -> {args.report}")
    return 0


def cmd_dataset_check(args: argparse.Namespace, config: Config) -> int:
    try:
        items = load_golden_set(args.dataset, verify_checksum=not args.no_verify_checksum)
    except DatasetError as exc:
        _err(f"FAIL: {exc}")
        return 1

    manifest = read_manifest(args.dataset)
    sittings: dict[str, int] = {}
    no_ref = 0
    for item in items:
        if item.reference:
            sittings[item.reference.sitting_id] = sittings.get(item.reference.sitting_id, 0) + 1
        else:
            no_ref += 1

    _out(f"PASS: {len(items)} question(s) in {args.dataset}")
    if manifest:
        _out(f"  sha256 {manifest.sha256[:16]}…  converted {manifest.created_at} from {manifest.source}")
    _out(f"  {len(sittings)} sitting(s); {no_ref} question(s) without a reference")
    for sitting, count in sorted(sittings.items()):
        _out(f"    {sitting:<22} {count:>4}")
    return 0


def cmd_corpus_check(args: argparse.Namespace, config: Config) -> int:
    """Verify the Hansard PDFs against their manifest, or write a new manifest.

    The golden set is checksummed; the documents it points at get the same
    treatment. A re-downloaded or truncated Hansard changes what the retriever
    can possibly find, and every score compared across that change is
    meaningless -- so this fails loudly rather than letting it pass quietly.
    """
    items: list[GoldenItem] = []
    golden_sha = ""
    if not args.no_golden_set:
        try:
            items = load_golden_set(args.dataset, verify_checksum=not args.no_verify_checksum)
            golden_sha = dataset_checksum(args.dataset)
        except DatasetError as exc:
            _err(f"error: golden set: {exc}")
            return 1

    if args.write:
        manifest = build_manifest(
            args.corpus_dir, items,
            golden_set=Path(args.dataset).name if items else "",
            golden_set_sha256=golden_sha,
        )
        path = write_manifest(args.corpus_dir, manifest)
        _out(f"Manifest -> {path}  ({len(manifest.documents)} document(s), "
             f"{manifest.total_bytes / 1e6:.1f} MB)")
        _out(f"  {'document':<24}{'pages':>7}{'questions':>11}  sha256")
        for doc in manifest.documents:
            _out(f"  {doc.filename:<24}{_fmt(doc.page_count):>7}"
                 f"{doc.golden_questions:>11}  {doc.sha256[:16]}…")
        _out("\nCommit the manifest, then verify anytime with 'rag-eval corpus-check'.")
        return 0

    try:
        manifest = read_corpus_manifest(args.corpus_dir)
    except CorpusError as exc:
        _err(f"error: {exc}")
        return 1

    report = verify_corpus(args.corpus_dir, manifest, items,
                           golden_set_sha256=golden_sha)

    _out(f"Corpus     {report.corpus_dir}")
    _out(f"Manifest   {manifest_path_for(args.corpus_dir)}  "
         f"(written {manifest.created_at or 'unknown'})")
    _out(f"Verified   {report.verified}/{report.documents} document(s), "
         f"{report.checked_bytes / 1e6:.1f} MB of sha256 checked")

    if items:
        stats = coverage(manifest, items, report.present_sittings)
        _out(f"Coverage   {stats['questions_covered']}/{stats['questions_with_a_reference']} "
             f"referenced question(s) have a PDF ({_pct(stats['coverage'])})")

    for failure in report.failures:
        _out(f"  FAIL  {failure}")
    for warning in report.warnings:
        _out(f"  warn  {warning}")

    _out(f"\n{report.verdict}")
    if report.ok and items:
        _out("Page counts bound the 'ms.' references, but printed page numbers may be "
             "offset from physical PDF pages — 'rag-eval ingest-check' is what confirms "
             "the mapping actually survived ingestion.")

    if args.output:
        Path(args.output).write_text(
            json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        _out(f"Full report -> {args.output}")
    return 0 if report.ok else 1


def cmd_page_offsets(args: argparse.Namespace, config: Config) -> int:
    """Measure each document's printed-vs-physical page offset.

    Content-based: it locates golden answers in the PDF text and compares where
    they land against the ``ms.`` the golden set records. Only measurements that
    clear the confidence bar become offsets; the rest are reported as
    undetermined rather than guessed at.
    """
    manifest = read_corpus_manifest(args.corpus_dir)
    items = load_golden_set(args.dataset, verify_checksum=not args.no_verify_checksum)
    results = measure_corpus(args.corpus_dir, manifest, items)

    # A human who opened the PDF and read "ms. 1 is on page 6" outranks any
    # text-matching heuristic. Existing verified entries are carried forward so
    # re-running the measurement never discards them.
    verified = read_verified(args.output) if args.output else {}
    for pair in args.set or []:
        key, _, value = pair.partition("=")
        try:
            verified[key.strip()] = int(value)
        except ValueError:
            _err(f"error: --set expects <sitting_id>=<integer>, got {pair!r}")
            return 1

    _out(f"{'document':<24}{'offset':>7}{'samples':>9}{'agree':>8}{'unread':>8}  note")
    for r in results:
        offset = str(r.offset) if r.confident else "—"
        agree = f"{r.agreement:.0%}" if r.agreement is not None else "—"
        _out(f"{r.filename:<24}{offset:>7}{r.samples:>9}{agree:>8}"
             f"{r.pages_unreadable:>4}/{r.pages_total:<3} {r.note[:44]}")

    payload = offsets_payload(results, verified)
    _out(f"\n{sum(1 for r in results if r.confident)}/{len(results)} measured confidently"
         + (f", {len(payload['verified'])} human-verified" if payload["verified"] else ""))
    for sitting, value in payload["verified"].items():
        row = next((e for e in payload["evidence"] if e["sitting_id"] == sitting), {})
        agrees = row.get("agrees_with_measurement")
        mark = ("agrees with measurement" if agrees
                else "measurement said %s" % row.get("offset") if agrees is False
                else "no measurement to compare")
        _out(f"  verified {sitting} = {value}  ({mark})")
    if payload["undetermined"]:
        _out(f"undetermined: {', '.join(payload['undetermined'])}")
        _out("  those documents are ingested without ms_offset — their chunks keep the "
             "physical page_number and printed 'ms.' stays unrecoverable for them.")

    if args.output:
        Path(args.output).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        _out(f"\nWrote {args.output}")
        _out(f"Use it with: rag-eval ingest --offsets {args.output}")
    else:
        _out("\nPass --output <file> to write the offsets file.")
    return 0


def cmd_ingest(args: argparse.Namespace, config: Config) -> int:
    """Create the eval collection and upload the Hansard corpus (D1).

    Dry-run by default. This writes to shared infrastructure and a collection's
    metadata schema cannot be changed after the fact, so the plan is printed for
    a human to read before anything is created.
    """
    manifest = read_corpus_manifest(args.corpus_dir)
    items = [] if args.no_golden_set else load_golden_set(
        args.dataset, verify_checksum=not args.no_verify_checksum
    )

    report = verify_corpus(args.corpus_dir, manifest, items)
    if not report.ok:
        _err("error: the corpus does not match its manifest; fix that before ingesting")
        for failure in report.failures:
            _err(f"  FAIL  {failure}")
        return 1

    offsets = json.loads(Path(args.offsets).read_text(encoding="utf-8")) if args.offsets else {}
    adapter = get_adapter("nvidia", **_adapter_options(config, args))
    profile = args.embedding_profile or adapter.embedding_profile
    plan = build_plan(
        adapter.collection, args.corpus_dir, manifest, items,
        offsets=offsets, embedding_profile=profile,
    )

    # One collection holds every Hansard PDF. Create it if missing, reuse it if
    # present -- but only after confirming it can actually serve the eval.
    status = adapter.collection_status()
    existing_docs = adapter.document_states() if status.get("exists") else {}
    plan = reconcile(plan, status, existing_docs, replace=args.replace)

    _out(f"Ingest server   {adapter.ingest_base_url or '(NVIDIA_INGEST_BASE_URL not set)'}")
    _out(f"Collection      {plan.collection}")
    if status.get("checked"):
        _out(f"  action        {plan.action.upper()}"
             + (f" — already exists with {status.get('entities')} entities, "
                f"{len(existing_docs)} document(s)" if status.get("exists") else " — does not exist yet"))
    info = adapter._profile_info(plan.embedding_profile) or {}
    caps = info.get("capabilities") or {}
    _out(f"Embedding profile  {plan.embedding_profile}"
         + (f"  — {info.get('label')}" if info else "  (unknown to the server)"))
    if info:
        _out(f"  multimodal {caps.get('multimodal')}   reranker {caps.get('reranker')}"
             f"   summary {caps.get('summary')}   formats {','.join(caps.get('formats', []))}")
        if caps.get("reranker") is False:
            _out("  ! This profile does NOT support the reranker. The server has reranking "
                 "on (top 100 -> 10), but it will not apply to this collection — retrieval "
                 "returns raw vector hits. Scores are not comparable with a 'text' collection.")
        if caps.get("summary") is False:
            _out("  ! This profile does not support summaries.")
        missing = [f for f in ("pdf",) if f not in (caps.get("formats") or [])]
        if missing:
            _out(f"  ! This profile does not accept {', '.join(missing)} — the corpus is PDFs.")
    if plan.must_create:
        _out(f"\nMetadata schema ({len(plan.metadata_schema)} field(s)) — "
             f"NOT changeable after creation:")
        for f in plan.metadata_schema:
            req = "required" if f.get("required") else "optional"
            _out(f"  {f['name']:<14}{f['type']:<9}{req:<9} {f.get('description','')[:60]}")
    else:
        _out(f"\nExisting metadata schema: "
             f"{', '.join(status.get('metadata_fields') or []) or '(none)'}")

    uploading = plan.to_upload
    _out(f"\nDocuments: {len(plan.documents)} in the corpus, "
         f"{len(uploading)} to upload ({plan.total_bytes / 1e6:.1f} MB)"
         + (f", {len(plan.skipped)} already present" if plan.skipped else "")
         + (f", {len(plan.stale)} registered-but-empty (will re-upload)" if plan.stale else ""))
    stale = set(plan.stale)
    for d in plan.documents:
        ms = d.metadata.get("ms_offset")
        mark = ("EMPT" if d.path.name in stale
                else "skip" if d.path.name in set(plan.skipped) else "  ->")
        _out(f"  {mark} {d.path.name:<24}{d.sitting_id:<18}{d.golden_questions:>4} qs"
             + (f"   ms_offset={ms}" if ms is not None else ""))
    for warning in plan.warnings:
        _out(f"  ~ {warning}")
    for problem in plan.problems:
        _out(f"  ! {problem}")

    if not offsets:
        _out("\n  ! No --offsets supplied: chunks will carry nv-ingest's PHYSICAL page_number "
             "with no way to recover the printed 'ms.' the golden set cites. "
             "See datasets/README.md.")

    if args.reset and status.get("exists") and existing_docs:
        names = sorted(existing_docs)
        if not args.yes:
            _out(f"\n--reset would DELETE {len(names)} document(s) from "
                 f"{plan.collection!r}: {', '.join(names)}")
        else:
            _out(f"\nResetting: deleting {len(names)} document(s) from {plan.collection!r}…")
            for n in names:
                _out(f"  - {n} ({existing_docs[n]} elements)")
            adapter.delete_documents(names)
            remaining = adapter.document_states()
            if remaining:
                _err(f"error: {len(remaining)} document(s) still present after reset: "
                     f"{sorted(remaining)}")
                return 1
            _out("  collection is now empty (the collection itself and its schema are kept)")
            # Rebuild from scratch rather than re-reconciling the old plan:
            # reconcile() appends problems, so the pre-reset warnings (stale
            # documents that no longer exist) would otherwise still block the run.
            plan = reconcile(
                build_plan(adapter.collection, args.corpus_dir, manifest, items,
                           offsets=offsets, embedding_profile=profile),
                {**status, "exists": True, "metadata_fields":
                    status.get("metadata_fields") or [f["name"] for f in plan.metadata_schema]},
                {}, replace=args.replace,
            )
            uploading = plan.to_upload
            _out(f"  {len(uploading)} document(s) to upload")

    if not uploading and not plan.problems:
        _out(f"\nNothing to do — all {len(plan.documents)} document(s) are already in "
             f"{plan.collection!r}. Pass --replace to re-upload them.")
        return 0
    if plan.problems and not args.force:
        _err("\nerror: refusing to ingest with unresolved problems (pass --force to override)")
        return 1

    if not args.yes:
        _out("\nDRY RUN — nothing was created or uploaded. Re-run with --yes to execute.")
        return 0

    _out("\nExecuting…")
    task_ids: list[str] = []
    unhealthy: list[dict[str, Any]] = []
    for event in execute_ingest(adapter, plan, batch_size=args.batch_size,
                                blocking=args.blocking):
        stage = event["stage"]
        if stage == "created":
            _out(f"  collection created: {str(event['response'])[:120]}")
        elif stage == "upload-error":
            _out(f"    !! TRANSPORT FAILED {', '.join(event['files'])}: {event['error']}")
            unhealthy.append({"document": ", ".join(event["files"]), "elements": 0,
                              "page_count": None, "verdict": "upload-error"})
        elif stage == "removing-stale":
            _out(f"  removing {len(event['documents'])} empty shell(s) so they can be "
                 f"re-uploaded: {', '.join(event['documents'])}")
        elif stage == "reused":
            _out(f"  reusing existing collection {event['collection']!r}")
        elif stage == "nothing-to-upload":
            _out("  nothing to upload")
        elif stage == "upload":
            _out(f"  batch {event['batch']}/{event['of']}: {', '.join(event['files'])}")
        elif stage == "uploaded":
            for h in event.get("health") or []:
                mark = {"ok": "  ok ", "thin": " THIN", "empty": "EMPTY",
                        "unknown": "  ? "}.get(h["verdict"], "  ? ")
                _out(f"    [{mark}] {h['document']:<24}{h['elements']:>6} elements"
                     f"  / {h['page_count']} pages"
                     + (f"  = {h['elements_per_page']}/pg" if h["elements_per_page"] is not None else ""))
                if h["verdict"] in ("empty", "thin"):
                    unhealthy.append(h)
            r = event["response"] or {}
            if event.get("task_id"):
                task_ids.append(event["task_id"])
            _out(f"    -> {r.get('message', 'ok')} "
                 f"({r.get('documents_completed', '?')}/{r.get('total_documents', '?')})"
                 + (f"  task {event['task_id']}" if event.get("task_id") else "  (no task id)"))
            for failed in (r.get("failed_documents") or []):
                name = failed.get("document_name", failed) if isinstance(failed, dict) else failed
                why = failed.get("error_message", "") if isinstance(failed, dict) else ""
                _out(f"    !! FAILED {name}: {why[:160]}")
    if unhealthy:
        _out(f"\n!! {len(unhealthy)} document(s) reported success but hold little or no "
             f"content — nv-ingest's parse shim can drop a page silently:")
        for h in unhealthy:
            _out(f"     {h['document']:<24}{h['elements']} elements / {h['page_count']} pages")
        _out("   Re-upload them (rag-eval ingest --replace) or raise the parse shim's "
             "PARSE_MAX_NEW_TOKENS server-side; do NOT score against this collection yet.")
    if task_ids:
        _out(f"\nTask id(s): {', '.join(task_ids)}")
        _out(f"  follow with: rag-eval ingest-status --task {task_ids[0]}")
    if args.output:
        Path(args.output).write_text(
            json.dumps({**plan.to_dict(), "executed": True, "task_ids": task_ids,
                        "unhealthy": unhealthy,
                        "collection_status_before": status}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        _out(f"Ingest report -> {args.output}")
    _out("\nNext: rag-eval ingest-check --adapter nvidia")
    return 0


def cmd_ingest_status(args: argparse.Namespace, config: Config) -> int:
    """Per-document ingestion progress for a task id."""
    adapter = get_adapter("nvidia", **_adapter_options(config, args))
    for task in args.task:
        body = adapter.ingestion_status(task)
        state = body.get("state", "?")
        nv = body.get("nv_ingest_status") or {}
        _out(f"task {task}  state={state}  extracted={nv.get('extraction_completed', '?')}")
        for name, st in (nv.get("document_wise_status") or {}).items():
            _out(f"    {name:<28}{st}")
        result = body.get("result") or {}
        for failed in (result.get("failed_documents") or []):
            name = failed.get("document_name", failed) if isinstance(failed, dict) else failed
            why = failed.get("error_message", "") if isinstance(failed, dict) else ""
            _out(f"    !! FAILED {name}: {why[:160]}")
    return 0


def cmd_ingest_check(args: argparse.Namespace, config: Config) -> int:
    items = _load_items(args)
    adapter = get_adapter(args.adapter, **_adapter_options(config, args))

    def progress(index, total, item, trace):
        mark = "!" if not trace.ok else " "
        meta = sum(1 for c in trace.retrieved_chunks if c.sitting_id and c.page is not None)
        _out(f"  [{index:>2}/{total}] {mark} {item.id:<12}"
             f"{len(trace.retrieved_chunks):>3} chunks, {meta} with metadata"
             f"  {(trace.latency_ms or 0) / 1000:.1f}s"
             + (f"  {trace.error[:60]}" if trace.error else ""))

    with adapter:
        report = ingest_check(adapter, items, sample=args.sample,
                              with_answer=args.with_answer,
                              on_probe=None if args.quiet else progress)

    _out(f"\nAdapter    {report['adapter']}  ({report['mode']})")
    _out(f"Health     {report['health']}")
    _out(f"Probes     {report['probes']} question(s) across {len(report['golden_sittings'])} sitting(s)")
    _out(f"Chunks     {report['chunks_retrieved']} retrieved")
    _out(f"  sitting id coverage  {_pct(report['sitting_id_coverage'])}")
    _out(f"  page coverage        {_pct(report['page_coverage'])}")
    if report["sittings_not_seen_in_probe"]:
        _out(
            f"  sittings never retrieved in this probe: "
            f"{', '.join(report['sittings_not_seen_in_probe'])}"
        )
    _out(f"\n{report['verdict']}")

    if args.output:
        Path(args.output).write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        _out(f"Full report -> {args.output}")
    return 0 if str(report["verdict"]).startswith(("PASS", "WARN")) else 1


def _probe_environment(adapter: Any) -> dict[str, Any]:
    """Record what the stack actually is; never let a probe abort a run."""
    try:
        env = adapter.probe_environment()
    except Exception as exc:  # noqa: BLE001
        return {"probe_error": f"{type(exc).__name__}: {exc}"[:200]}
    if env:
        _out(
            f"  stack: retrieval={env.get('retrieval_mode', '?')}  "
            f"embedding={env.get('embedding_model', '?')}"
        )
    return env


def cmd_run(args: argparse.Namespace, config: Config) -> int:
    items = _load_items(args)
    options = _adapter_options(config, args)
    adapter = get_adapter(args.adapter, **options)

    if args.resume:
        store = RunStore.open(config.runs_dir, args.resume)
        manifest = store.read_manifest()
    else:
        store = RunStore.create(config.runs_dir, args.adapter)
        dataset_manifest = read_manifest(args.dataset)
        manifest = RunManifest(
            run_id=store.run_id,
            adapter=args.adapter,
            created_at=_now(),
            dataset=str(args.dataset),
            dataset_sha256=dataset_manifest.sha256 if dataset_manifest else "",
            dataset_count=dataset_manifest.count if dataset_manifest else len(items),
            questions_run=len(items),
            adapter_config=adapter.describe(),
            rag_environment=_probe_environment(adapter),
            judge_models=[m.id for m in config.judge.models],
            config=config.to_dict(),
            notes=args.note,
        )
        store.write_manifest(manifest)

    _out(f"Run {store.run_id} — {len(items)} question(s) through '{args.adapter}'")

    def progress(index: int, total: int, trace: RagTrace) -> None:
        mark = "!" if trace.error else "."
        end = "\n" if index == total or index % 50 == 0 else ""
        print(mark, end=end, file=sys.stdout, flush=True)

    with adapter:
        traces = run_dataset(
            adapter,
            items,
            store,
            concurrency=args.concurrency,
            on_progress=None if args.quiet else progress,
            resume=bool(args.resume),
        )
    _out("")

    errors = sum(1 for t in traces if not t.ok)
    store.stamp_stage("run")
    _out(f"Traces -> {store.traces_path}  ({len(traces)} trace(s), {errors} error(s))")
    if errors:
        _out("  failed questions are kept as error traces and excluded from judging")
    _out(f"\nNext: rag-eval judge --run {store.run_id}")
    return 0


def cmd_judge(args: argparse.Namespace, config: Config) -> int:
    store = _resolve_store(config, args.run)
    traces = store.read_traces()
    if not traces:
        _err(f"run {store.run_id} has no traces; run 'rag-eval run' first")
        return 1

    items = {i.id: i for i in _load_items(args)}
    pairs = [(items[t.question_id], t) for t in traces if t.question_id in items]
    if args.limit:
        pairs = pairs[: args.limit]

    panel = _judge_panel(config, args.offline_judge)
    _out(
        f"Judging {len(pairs)} question(s) with "
        f"{', '.join(m.label for m in config.judge.models)}"
    )

    verdicts = []
    store.judgments_path.unlink(missing_ok=True)
    for index, (item, trace) in enumerate(pairs, 1):
        verdict = panel.judge_one(item, trace)
        verdicts.append(verdict)
        store.append_judgment(verdict)
        if not args.quiet:
            mark = "?" if verdict.needs_human_review else "."
            end = "\n" if index == len(pairs) or index % 50 == 0 else ""
            print(mark, end=end, file=sys.stdout, flush=True)
    _out("")

    flagged = flagged_rows(verdicts)
    manifest = store.read_manifest()
    manifest.judge_models = [m.id for m in config.judge.models]
    manifest.judge_usage = panel.usage.to_dict()
    store.write_manifest(manifest)
    store.stamp_stage("judge")

    _out(f"Judgments -> {store.judgments_path}")
    _out(
        f"  {panel.usage.calls} judge call(s), {panel.usage.failed_calls} failed, "
        f"est. ${panel.usage.cost_usd:.4f}"
    )
    _out(f"  {len(flagged)} row(s) split the panel and need human adjudication")
    if flagged:
        _out(f"    e.g. {', '.join(v.question_id for v in flagged[:8])}")
        _out(f"    export them with: rag-eval adjudicate --run {store.run_id} --export flagged.json")
    _out(f"\nNext: rag-eval score --run {store.run_id}")
    return 0


def cmd_score(args: argparse.Namespace, config: Config) -> int:
    store = _resolve_store(config, args.run)
    manifest = store.read_manifest()
    traces = store.read_traces()
    if not traces:
        _err(f"run {store.run_id} has no traces")
        return 1
    verdicts = store.read_judgments()

    items = _load_items(argparse.Namespace(
        dataset=args.dataset or manifest.dataset or DEFAULT_DATASET,
        no_verify_checksum=args.no_verify_checksum,
        limit=None,
        ids=None,
    ))

    usage = PanelUsage(**{
        k: v for k, v in (manifest.judge_usage or {}).items()
        if k in PanelUsage.__dataclass_fields__
    }) if manifest.judge_usage else None

    card = build_scorecard(
        store.run_id,
        manifest.adapter,
        items,
        traces,
        verdicts,
        config=config,
        dataset=manifest.dataset,
        created_at=manifest.created_at,
        judge_usage=usage,
        rag_environment=manifest.rag_environment,
    )
    store.write_scorecard(card.to_dict())
    store.stamp_stage("score")

    _out(f"Scorecard -> {store.scorecard_path}\n")
    for row in card.headline():
        value = row["value"]
        shown = "—" if value is None else f"{value:.4f}" if isinstance(value, float) else value
        _out(f"  {row['block']:<11} {row['metric']:<26} {shown}")
    if card.warnings:
        _out("\nCaveats:")
        for warning in card.warnings:
            _out(f"  ! {warning}")
    _out(f"\nNext: rag-eval report --run {store.run_id}")
    return 0


def cmd_report(args: argparse.Namespace, config: Config) -> int:
    store = _resolve_store(config, args.run)
    manifest = store.read_manifest()
    card = store.read_scorecard()

    baseline_card = None
    if args.baseline:
        baseline_card = RunStore.open(config.runs_dir, args.baseline).read_scorecard()
    elif not args.no_baseline:
        baseline = previous_run(config.runs_dir, manifest.adapter, before=store.run_id)
        if baseline:
            baseline_card = baseline.read_scorecard()

    diff = compare(card, baseline_card)
    (store.dir / "diff.json").write_text(
        json.dumps(diff, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    path = write_report(args.output or store.report_path, card, diff, manifest.to_dict())
    store.stamp_stage("report")

    _out(f"Scorecard  -> {path}")
    _out(f"Diff       -> {store.dir / 'diff.json'}")
    if not diff.get("comparable", True):
        _out("\n  !! NOT COMPARABLE — the RAG stack changed between these runs:")
        for d in diff["environment_drift"]["differences"]:
            _out(f"       {d['field']}: {d['baseline']} -> {d['current']}")
        _out("     The deltas below measure a different system, not a change to the same one.")
    if diff["has_baseline"]:
        _out(f"\nvs {diff['baseline_run_id']}: " + " · ".join(
            f"{v} {k}" for k, v in sorted(diff["summary"].items())
        ))
        for row in diff["regressed"]:
            _out(f"  REGRESSED {row['block']}.{row['metric']}: "
                 f"{row['baseline']} -> {row['current']} ({row['delta']:+.4f})")
    else:
        _out("\nNo previous run for this adapter — this run becomes the baseline.")
    return 0


def cmd_eval(args: argparse.Namespace, config: Config) -> int:
    """run + judge + score + report, for the common before/after-a-change case."""
    rc = cmd_run(args, config)
    if rc:
        return rc
    args.run = list_runs(config.runs_dir, args.adapter)[-1].run_id
    if not args.skip_judge:
        rc = cmd_judge(args, config)
        if rc:
            return rc
    rc = cmd_score(args, config)
    if rc:
        return rc
    return cmd_report(args, config)


def cmd_adjudicate(args: argparse.Namespace, config: Config) -> int:
    """Export panel splits for a human, or import the human's decisions back."""
    store = _resolve_store(config, args.run)
    verdicts = store.read_judgments()

    if args.export:
        flagged = flagged_rows(verdicts)
        traces = {t.question_id: t for t in store.read_traces()}
        items = {i.id: i for i in _load_items(args)}
        payload = [
            {
                "question_id": v.question_id,
                "question": items[v.question_id].question if v.question_id in items else "",
                "expected_answer": (
                    items[v.question_id].expected_answer if v.question_id in items else ""
                ),
                "generated_answer": (
                    traces[v.question_id].generated_answer if v.question_id in traces else ""
                ),
                "flagged_dimensions": v.flagged_dimensions,
                "panel_scores": v.scores,
                "judge_scores": {j.model: j.scores for j in v.verdicts if j.ok},
                "rationales": {j.model: j.rationale for j in v.verdicts if j.rationale},
                "human_scores": {d: None for d in v.flagged_dimensions},
            }
            for v in flagged
        ]
        Path(args.export).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        _out(f"Exported {len(payload)} flagged row(s) -> {args.export}")
        _out("Fill in 'human_scores', then: rag-eval adjudicate --run "
             f"{store.run_id} --import {args.export}")
        return 0

    if args.import_path:
        decisions = json.loads(Path(args.import_path).read_text(encoding="utf-8"))
        by_id = {v.question_id: v for v in verdicts}
        applied = 0
        for row in decisions:
            verdict = by_id.get(str(row.get("question_id")))
            if not verdict:
                continue
            scores = {
                d: int(s) for d, s in (row.get("human_scores") or {}).items() if s is not None
            }
            if scores:
                verdict.human_scores.update(scores)
                applied += 1
        store.write_judgments(verdicts)

        added = append_calibration(
            samples_from_adjudications(
                verdicts, run_id=store.run_id, labelled_by=args.labelled_by
            ),
            args.calibration,
        )
        _out(f"Applied {applied} adjudication(s) to {store.judgments_path}")
        _out(f"Appended {added} new sample(s) to {args.calibration}")
        _out(f"\nRe-score to fold them in: rag-eval score --run {store.run_id}")
        return 0

    _err("pass --export <file> or --import <file>")
    return 2


def cmd_calibrate(args: argparse.Namespace, config: Config) -> int:
    """Report how well the panel agrees with the human-labelled samples."""
    store = _resolve_store(config, args.run)
    verdicts = store.read_judgments()
    samples = load_calibration(args.calibration)
    if not samples:
        _err(f"no calibration samples in {args.calibration}")
        return 1

    report = agreement_report(verdicts, samples)
    _out(f"Calibration set: {report['calibration_set_size']} sample(s), "
         f"{report['labelled_samples']} overlapping run {store.run_id}\n")
    _out(f"  {'dimension':<20}{'labelled':>9}{'exact':>9}{'within 1':>10}{'bias':>8}")
    for dimension, stats in report["per_dimension"].items():
        _out(
            f"  {dimension:<20}{stats['labelled']:>9}"
            f"{_pct(stats['exact_agreement']):>9}{_pct(stats['within_1']):>10}"
            f"{_fmt(stats['mean_bias']):>8}"
        )
    if report["per_judge"]:
        _out(f"\n  {'judge':<20}{'compared':>9}{'exact':>9}{'within 1':>10}{'bias':>8}")
        for model, stats in report["per_judge"].items():
            _out(
                f"  {model:<20}{stats['compared']:>9}"
                f"{_pct(stats['exact_agreement']):>9}{_pct(stats['within_1']):>10}"
                f"{_fmt(stats['mean_bias']):>8}"
            )
    if args.output:
        Path(args.output).write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        _out(f"\nFull report -> {args.output}")
    return 0


def cmd_label(args: argparse.Namespace, config: Config) -> int:
    """Import a human-labelled calibration file (the ~50-sample seed set)."""
    rows = json.loads(Path(args.file).read_text(encoding="utf-8"))
    samples = [
        CalibrationSample(
            question_id=str(row["question_id"]),
            human_scores={k: int(v) for k, v in (row.get("human_scores") or {}).items()},
            labelled_by=row.get("labelled_by", args.labelled_by),
            notes=row.get("notes", ""),
        )
        for row in rows
        if row.get("human_scores")
    ]
    added = append_calibration(samples, args.calibration)
    _out(f"Added {added} new sample(s) to {args.calibration} "
         f"({len(samples) - added} already labelled)")
    return 0


def cmd_list_runs(args: argparse.Namespace, config: Config) -> int:
    runs = list_runs(config.runs_dir, args.adapter)
    if not runs:
        _out("no runs yet")
        return 0
    _out(f"{'run id':<32}{'stages':<28}{'questions':>10}")
    for store in runs:
        try:
            manifest = store.read_manifest()
        except FileNotFoundError:
            _out(f"{store.run_id:<32}{'(no manifest)':<28}")
            continue
        _out(
            f"{store.run_id:<32}{','.join(manifest.stages) or '-':<28}"
            f"{manifest.questions_run:>10}"
        )
    return 0


def cmd_diff(args: argparse.Namespace, config: Config) -> int:
    current = RunStore.open(config.runs_dir, args.run).read_scorecard()
    baseline = RunStore.open(config.runs_dir, args.baseline).read_scorecard()
    diff = compare(current, baseline)
    _out(f"{args.baseline} -> {args.run}\n")
    _out(f"  {'metric':<44}{'baseline':>11}{'current':>11}{'delta':>11}  verdict")
    for row in diff["metrics"]:
        _out(
            f"  {row['block'] + '.' + row['metric']:<44}"
            f"{_fmt(row['baseline']):>11}{_fmt(row['current']):>11}"
            f"{_fmt(row['delta']):>11}  {row['verdict']}"
        )
    if args.output:
        Path(args.output).write_text(
            json.dumps(diff, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return 1 if args.fail_on_regression and diff["regressed"] else 0


def cmd_adapters(args: argparse.Namespace, config: Config) -> int:
    for name in available():
        _out(name)
    return 0


def _pct(value: Any) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    return f"{value:.4f}" if isinstance(value, float) else str(value)


# ---------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-eval",
        description="Evaluation harness for RAG services: retrieval, generation and ops.",
    )
    parser.add_argument("--version", action="version", version=f"rag-eval {__version__}")
    parser.add_argument("--config", help="path to rag-eval.config.json")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_dataset_args(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--dataset", default=DEFAULT_DATASET, help="golden set JSONL")
        sub.add_argument("--no-verify-checksum", action="store_true",
                         help="load the golden set even if it no longer matches its manifest")
        sub.add_argument("--limit", type=int, help="only the first N questions")
        sub.add_argument("--ids", help="comma-separated question ids")

    def add_adapter_args(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--adapter", required=True, choices=available())
        sub.add_argument("--option", action="append", metavar="KEY=VALUE",
                         help="adapter option override (repeatable)")
        sub.add_argument("--top-k", type=int, help="chunks to retrieve per question")

    p = subparsers.add_parser("convert-golden", help="convert the Q&A workbook into golden_v1.jsonl")
    p.add_argument("--xlsx", default="datasets/TanyaParlimen_QnA.xlsx")
    p.add_argument("--output", default=DEFAULT_DATASET)
    p.add_argument("--sheet", type=int, default=1)
    p.add_argument("--strict", action="store_true",
                   help="fail on any unparseable or unknown reference")
    p.add_argument("--report", help="write the conversion report as JSON")
    p.set_defaults(func=cmd_convert_golden)

    p = subparsers.add_parser("dataset-check", help="validate a golden set's schema and checksum")
    add_dataset_args(p)
    p.set_defaults(func=cmd_dataset_check)

    p = subparsers.add_parser(
        "corpus-check", help="verify the Hansard PDFs against their sha256 manifest"
    )
    p.add_argument("--corpus-dir", default=str(DEFAULT_CORPUS_DIR))
    p.add_argument("--dataset", default=DEFAULT_DATASET,
                   help="golden set to cross-check the corpus against")
    p.add_argument("--no-verify-checksum", action="store_true")
    p.add_argument("--no-golden-set", action="store_true",
                   help="check the PDFs alone, without cross-checking the golden set")
    p.add_argument("--write", action="store_true",
                   help="(re)generate the manifest from what is on disk")
    p.add_argument("--output", help="write the full report as JSON")
    p.set_defaults(func=cmd_corpus_check)

    p = subparsers.add_parser(
        "page-offsets", help="measure printed-ms. vs physical-page offset per document"
    )
    add_dataset_args(p)
    p.add_argument("--corpus-dir", default=str(DEFAULT_CORPUS_DIR))
    p.add_argument("--output", help="write the offsets JSON (for 'ingest --offsets')")
    p.add_argument("--set", action="append", metavar="SITTING=OFFSET",
                   help="record a human-verified offset; survives re-measurement (repeatable)")
    p.set_defaults(func=cmd_page_offsets)

    p = subparsers.add_parser(
        "ingest", help="create the eval collection and upload the Hansard PDFs"
    )
    add_dataset_args(p)
    p.add_argument("--corpus-dir", default=str(DEFAULT_CORPUS_DIR))
    p.add_argument("--no-golden-set", action="store_true")
    p.add_argument("--embedding-profile", default="",
                   help="'vl' (multimodal) or 'text'; defaults to the adapter's setting")
    p.add_argument("--offsets", metavar="FILE",
                   help='JSON {"<sitting_id>": <ms_offset>} stamped onto each document')
    # Serialized by default: this stack drops results when many documents are
    # submitted at once (14 concurrent -> 12 lost with 404s fetching results),
    # and a non-blocking upload reports success for work that later fails.
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--no-blocking", dest="blocking", action="store_false",
                   help="fire-and-forget instead of waiting for each batch "
                        "(faster, but this stack loses documents under load)")
    p.set_defaults(blocking=True)
    p.add_argument("--option", action="append", metavar="KEY=VALUE")
    p.add_argument("--top-k", type=int, help=argparse.SUPPRESS)
    p.add_argument("--output", help="write the ingest report as JSON")
    p.add_argument("--reset", action="store_true",
                   help="delete every document in the collection first (keeps the collection "
                        "and its schema); use to re-ingest from clean")
    p.add_argument("--replace", action="store_true",
                   help="re-upload documents that are already in the collection")
    p.add_argument("--force", action="store_true", help="ingest despite plan problems")
    p.add_argument("--yes", action="store_true",
                   help="actually create and upload (default is a dry run)")
    p.set_defaults(func=cmd_ingest, adapter="nvidia")

    p = subparsers.add_parser("ingest-status", help="per-document progress for an ingest task")
    p.add_argument("--task", action="append", required=True, metavar="TASK_ID")
    p.add_argument("--option", action="append", metavar="KEY=VALUE")
    p.add_argument("--top-k", type=int, help=argparse.SUPPRESS)
    p.set_defaults(func=cmd_ingest_status, adapter="nvidia")

    p = subparsers.add_parser(
        "ingest-check", help="verify the collection carries sitting id + page metadata"
    )
    add_dataset_args(p)
    add_adapter_args(p)
    p.add_argument("--sample", type=int, default=10, help="probe questions, spread across sittings")
    p.add_argument("--with-answer", action="store_true",
                   help="probe via the full generate path (slow: the LLM writes an answer "
                        "this check discards) instead of retrieval only")
    p.add_argument("--quiet", action="store_true", help="suppress per-probe progress")
    p.add_argument("--output", help="write the full report as JSON")
    p.set_defaults(func=cmd_ingest_check)

    p = subparsers.add_parser("run", help="run the golden set through an adapter")
    add_dataset_args(p)
    add_adapter_args(p)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--resume", metavar="RUN_ID", help="continue an interrupted run")
    p.add_argument("--note", default="", help="free-text note stored in the manifest")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_run)

    p = subparsers.add_parser("judge", help="score a run's traces with the judge panel")
    add_dataset_args(p)
    p.add_argument("--run", help="run id (default: the most recent run)")
    p.add_argument("--offline-judge", metavar="FILE",
                   help="reply with canned JSON instead of calling the gateway (smoke tests): "
                        "one rubric object, or a {model_id: rubric} map to script a panel split")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_judge)

    p = subparsers.add_parser("score", help="compute metrics into scorecard.json")
    p.add_argument("--run", help="run id (default: the most recent run)")
    p.add_argument("--dataset", help="override the dataset recorded in the manifest")
    p.add_argument("--no-verify-checksum", action="store_true")
    p.set_defaults(func=cmd_score)

    p = subparsers.add_parser("report", help="render scorecard.html with the regression diff")
    p.add_argument("--run", help="run id (default: the most recent run)")
    p.add_argument("--baseline", help="run id to diff against (default: previous run, same adapter)")
    p.add_argument("--no-baseline", action="store_true", help="render without a diff")
    p.add_argument("--output", help="write the HTML somewhere other than the run directory")
    p.set_defaults(func=cmd_report)

    p = subparsers.add_parser("eval", help="run + judge + score + report in one command")
    add_dataset_args(p)
    add_adapter_args(p)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--note", default="")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--skip-judge", action="store_true",
                   help="retrieval and ops metrics only (no judge cost)")
    p.add_argument("--offline-judge", metavar="FILE")
    p.add_argument("--baseline")
    p.add_argument("--no-baseline", action="store_true")
    p.add_argument("--output")
    p.set_defaults(func=cmd_eval, resume=None, run=None)

    p = subparsers.add_parser("adjudicate", help="export panel splits / import human decisions")
    add_dataset_args(p)
    p.add_argument("--run", help="run id (default: the most recent run)")
    p.add_argument("--export", metavar="FILE")
    p.add_argument("--import", dest="import_path", metavar="FILE")
    p.add_argument("--calibration", default="datasets/judge_calibration.jsonl")
    p.add_argument("--labelled-by", default="")
    p.set_defaults(func=cmd_adjudicate)

    p = subparsers.add_parser("calibrate", help="report panel agreement with human labels")
    p.add_argument("--run", help="run id (default: the most recent run)")
    p.add_argument("--calibration", default="datasets/judge_calibration.jsonl")
    p.add_argument("--output")
    p.set_defaults(func=cmd_calibrate)

    p = subparsers.add_parser("label", help="import human-labelled calibration samples")
    p.add_argument("file")
    p.add_argument("--calibration", default="datasets/judge_calibration.jsonl")
    p.add_argument("--labelled-by", default="")
    p.set_defaults(func=cmd_label)

    p = subparsers.add_parser("diff", help="compare two scored runs")
    p.add_argument("--run", required=True)
    p.add_argument("--baseline", required=True)
    p.add_argument("--output")
    p.add_argument("--fail-on-regression", action="store_true",
                   help="exit non-zero if any metric regressed (opt-in; v1 keeps a human in the loop)")
    p.set_defaults(func=cmd_diff)

    p = subparsers.add_parser("list-runs", help="list run directories")
    p.add_argument("--adapter")
    p.set_defaults(func=cmd_list_runs)

    p = subparsers.add_parser("adapters", help="list registered RAG adapters")
    p.set_defaults(func=cmd_adapters)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    try:
        return args.func(args, config)
    except (DatasetError, CorpusError, FileNotFoundError, ValueError) as exc:
        _err(f"error: {exc}")
        return 1
    except KeyboardInterrupt:  # pragma: no cover - interactive
        _err("\ninterrupted; partial artifacts are kept in the run directory")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
