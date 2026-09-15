"""Build the Hansard eval collection (D1): create, then upload.

The vector DB will not accept documents for a collection that does not exist,
so the order is fixed: ``POST /collection`` declaring the metadata schema, then
``POST /documents`` with the PDFs and their per-document metadata.

The metadata schema is the half that decides what this framework can measure.

A collection returns exactly the fields it declared at creation -- verified on
this stack: ``dr_transcripts`` declares six fields and returns those six;
``__eval_fiqa`` declares only ``filename`` and returns only ``filename``. And
the schema is fixed once set: ``PATCH /collections/{name}/metadata`` updates
catalog fields (description, tags, owner, status) and cannot add a schema field.

So the retrieval metrics need the sitting to be recoverable from chunk metadata,
either directly from ``sitting_id`` or by deriving it from ``dewan`` +
``session_date`` (which is what the adapter's ``dewan_fields`` /
``session_date_fields`` are for). Declare none of the three and every
sitting-based metric -- hit_rate, recall, MRR, page-citation accuracy -- reads
zero no matter how well retrieval actually works, and the only fix is to drop
the collection and re-ingest.

The one field to *not* declare is a page number: nv-ingest auto-extracts
``page_number`` into ``custom_metadata`` on its own, as the physical PDF page.
``ms_offset`` is what makes the printed Hansard page recoverable from it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from rag_eval.dataset.corpus import CorpusManifest
from rag_eval.types import DOC_TYPES, GoldenItem

#: Chamber name as the existing collections record it, keyed by golden prefix.
DEWAN_NAMES = {
    "dr": "dewan rakyat",
    "dn": "dewan negara",
    "kkdr": "kamar khas dewan rakyat",
}

#: The collection's metadata schema.
#:
#: ``page_number`` is deliberately absent: nv-ingest extracts it automatically
#: as the *physical* PDF page. ``ms_offset`` is what lets a consumer recover the
#: printed Hansard page the golden set actually cites, via
#: ``ms = page_number - ms_offset``.
METADATA_SCHEMA: list[dict[str, Any]] = [
    {"name": "sitting_id", "type": "string", "required": True, "user_defined": True,
     "max_length": 32,
     "description": "Hansard sitting key, '<doc_type>_<ISO date>' e.g. dr_2026-06-22"},
    {"name": "doc_type", "type": "string", "required": True, "user_defined": True,
     "max_length": 8, "description": "dr | dn | kkdr"},
    {"name": "dewan", "type": "string", "required": False, "user_defined": True,
     "max_length": 64, "description": "Chamber name, e.g. 'dewan rakyat'"},
    {"name": "session_date", "type": "string", "required": True, "user_defined": True,
     "max_length": 10, "description": "Sitting date, ISO yyyy-mm-dd"},
    {"name": "ms_offset", "type": "integer", "required": False, "user_defined": True,
     "support_dynamic_filtering": False,
     "description": "printed ms. = page_number - ms_offset (front matter offset)"},
]


#: Elements per page below which a document is treated as content-lost.
#:
#: The parse shim can silently drop content: a page whose generation is
#: length-truncated, or whose output fails the grammar, yields fewer elements or
#: none while nv-ingest still reports the job FINISHED with no failed documents.
#: A server-side "success" is therefore not evidence that a document was
#: ingested -- the element count relative to the page count is.
#:
#: Calibrated against Hansard already on this stack rather than a guess:
#: dr_transcripts holds 41,852 elements across 330 sitting PDFs (~127 each,
#: roughly 1 per page for a full sitting), and our one healthy short sitting
#: (kkdr_2026-07-16, 23 pages) yielded 77 (3.3 per page). So normal density here
#: is ~1-3 per page. The floor sits well below that: it is meant to catch
#: catastrophic loss, not to police density, because a false "thin" would send
#: someone chasing a healthy document.
MIN_ELEMENTS_PER_PAGE = 0.5


@dataclass
class DocumentPlan:
    path: Path
    sitting_id: str
    metadata: dict[str, Any]
    golden_questions: int = 0
    page_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "file": self.path.name,
            "sitting_id": self.sitting_id,
            "metadata": self.metadata,
            "golden_questions": self.golden_questions,
            "page_count": self.page_count,
        }


@dataclass
class IngestPlan:
    collection: str
    documents: list[DocumentPlan] = field(default_factory=list)
    metadata_schema: list[dict[str, Any]] = field(default_factory=lambda: list(METADATA_SCHEMA))
    embedding_profile: str = "text"
    #: Blocking: the ingest must not proceed (bad schema, missing document,
    #: unverifiable collection). Anything the plan already handles belongs in
    #: ``warnings`` -- refusing to run over a condition we have just resolved
    #: only teaches people to pass --force, which disables the real checks too.
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    #: False when the collection already exists and is being reused.
    must_create: bool = True
    #: Documents already present *and* holding content, skipped rather than
    #: uploaded twice.
    skipped: list[str] = field(default_factory=list)
    #: Registered but empty -- a failed ingest left the name behind with zero
    #: elements. These are re-uploaded, never skipped.
    stale: list[str] = field(default_factory=list)

    @property
    def to_upload(self) -> list[DocumentPlan]:
        skip = set(self.skipped)
        return [d for d in self.documents if d.path.name not in skip]

    @property
    def action(self) -> str:
        return "create" if self.must_create else "reuse"

    @property
    def total_bytes(self) -> int:
        return sum(d.path.stat().st_size for d in self.to_upload if d.path.exists())

    def to_dict(self) -> dict[str, Any]:
        return {
            "collection": self.collection,
            "action": self.action,
            "embedding_profile": self.embedding_profile,
            "metadata_schema": self.metadata_schema,
            "documents": [d.to_dict() for d in self.documents],
            "document_count": len(self.documents),
            "uploading": len(self.to_upload),
            "skipped": list(self.skipped),
            "stale": list(self.stale),
            "total_bytes": self.total_bytes,
            "problems": self.problems,
            "warnings": self.warnings,
        }


def required_fields() -> set[str]:
    return {f["name"] for f in METADATA_SCHEMA if f.get("required")}


def reconcile(
    plan: IngestPlan,
    status: dict[str, Any],
    existing_documents: Sequence[str] | dict[str, int] = (),
    *,
    replace: bool = False,
) -> IngestPlan:
    """Adapt the plan to what is already on the server.

    Missing collection -> create it. Existing collection -> reuse it, but only
    after checking it can actually serve the eval: a collection created without
    the required metadata fields cannot be repaired (the schema is fixed at
    creation), so ingesting into it would produce a permanently unscoreable
    corpus. That is a refusal, not a warning.
    """
    if not status.get("checked"):
        plan.problems.append(
            f"could not confirm whether {plan.collection!r} exists "
            f"({status.get('reason', 'unknown')})"
        )
        return plan

    if not status.get("exists"):
        plan.must_create = True
        return plan

    plan.must_create = False
    present = {f for f in (status.get("metadata_fields") or []) if f}
    missing_required = sorted(required_fields() - present)
    if missing_required:
        plan.problems.append(
            f"{plan.collection!r} already exists but its metadata schema is missing "
            f"{', '.join(missing_required)}. The schema cannot be changed after creation, "
            f"so chunks from it can never be scored — drop the collection and re-create it."
        )
    missing_optional = sorted(
        {f["name"] for f in METADATA_SCHEMA} - required_fields() - present
    )
    if missing_optional:
        plan.warnings.append(
            f"{plan.collection!r} is missing optional field(s) {', '.join(missing_optional)}; "
            f"anything depending on them (e.g. ms_offset for printed page numbers) will not work."
        )

    # A mapping carries element counts; a bare sequence predates that and is
    # taken at face value.
    if isinstance(existing_documents, dict):
        counts = existing_documents
    else:
        counts = {name: -1 for name in existing_documents}

    names = {d.path.name for d in plan.documents}
    plan.stale = sorted(n for n, elements in counts.items() if n in names and elements == 0)
    if plan.stale:
        plan.warnings.append(
            f"{len(plan.stale)} document(s) are registered but hold zero elements "
            f"({', '.join(plan.stale)}) — a failed ingest left the name behind. They will be "
            f"re-uploaded; skipping them would leave a silent hole in the corpus."
        )

    if not replace:
        usable = {n for n, elements in counts.items() if elements != 0}
        plan.skipped = sorted(d.path.name for d in plan.documents if d.path.name in usable)
    return plan


def build_plan(
    collection: str,
    corpus_dir: str | Path,
    manifest: CorpusManifest,
    items: Sequence[GoldenItem] = (),
    *,
    offsets: dict[str, int] | None = None,
    embedding_profile: str = "text",
) -> IngestPlan:
    """Plan the ingest from the checksummed corpus, not from a directory listing.

    Going through the manifest means only documents whose bytes were verified
    get uploaded, and the collection is reproducible: the same manifest yields
    the same ingest.
    """
    corpus_dir = Path(corpus_dir)
    plan = IngestPlan(collection=collection, embedding_profile=embedding_profile)
    # Accept either a bare {sitting_id: offset} map or the richer file written
    # by 'rag-eval page-offsets', which nests them under "offsets".
    offsets = (offsets or {}).get("offsets", offsets) if isinstance(offsets, dict) else {}
    offsets = offsets or {}

    counts: dict[str, int] = {}
    for item in items:
        if item.reference:
            counts[item.reference.sitting_id] = counts.get(item.reference.sitting_id, 0) + 1

    for doc in sorted(manifest.documents, key=lambda d: d.sitting_id):
        path = corpus_dir / doc.filename
        if not path.exists():
            plan.problems.append(f"{doc.filename}: listed in the manifest but not on disk")
            continue
        if doc.doc_type not in DOC_TYPES:
            plan.problems.append(
                f"{doc.filename}: unrecognised document prefix {doc.doc_type!r}"
            )

        metadata: dict[str, Any] = {
            "sitting_id": doc.sitting_id,
            "doc_type": doc.doc_type,
            "dewan": DEWAN_NAMES.get(doc.doc_type, ""),
            "session_date": doc.sitting_date,
        }
        if doc.sitting_id in offsets:
            metadata["ms_offset"] = int(offsets[doc.sitting_id])

        plan.documents.append(
            DocumentPlan(
                path=path,
                sitting_id=doc.sitting_id,
                metadata=metadata,
                golden_questions=counts.get(doc.sitting_id, 0),
                page_count=doc.page_count,
            )
        )

    cited = set(counts)
    planned = {d.sitting_id for d in plan.documents}
    for sitting in sorted(cited - planned):
        plan.problems.append(
            f"{sitting}: {counts[sitting]} golden question(s) cite this sitting but no "
            f"document is planned for it"
        )
    return plan


@dataclass
class DocumentHealth:
    """Whether a document actually landed, judged on content rather than status."""

    name: str
    elements: int
    page_count: int | None
    verdict: str            # ok | thin | empty | unknown

    @property
    def per_page(self) -> float | None:
        if not self.page_count:
            return None
        return round(self.elements / self.page_count, 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "document": self.name, "elements": self.elements,
            "page_count": self.page_count, "elements_per_page": self.per_page,
            "verdict": self.verdict,
        }


def check_document(name: str, elements: int, page_count: int | None) -> DocumentHealth:
    """Judge one ingested document by content density, not by reported status."""
    if elements <= 0:
        return DocumentHealth(name, elements, page_count, "empty")
    if page_count:
        ratio = elements / page_count
        if ratio < MIN_ELEMENTS_PER_PAGE:
            return DocumentHealth(name, elements, page_count, "thin")
        return DocumentHealth(name, elements, page_count, "ok")
    return DocumentHealth(name, elements, page_count, "unknown")


def audit(adapter: Any, plan: IngestPlan) -> list[DocumentHealth]:
    """Content health of every planned document currently in the collection."""
    states = adapter.document_states()
    return [
        check_document(d.path.name, states.get(d.path.name, 0), d.page_count)
        for d in plan.documents
    ]


def execute(adapter: Any, plan: IngestPlan, *, batch_size: int = 4, blocking: bool = False):
    """Create the collection if needed, then upload in batches.

    One collection holds every Hansard PDF: ``create_collection`` is called at
    most once, and never when the collection already exists.
    """
    if plan.must_create:
        yield {"stage": "create", "collection": plan.collection}
        created = adapter.create_collection(
            plan.metadata_schema,
            embedding_profile=plan.embedding_profile,
            description="Hansard sittings referenced by the TanyaParlimen golden set",
            tags=["rag-eval", "hansard", "parliament"],
        )
        yield {"stage": "created", "response": created}
    else:
        yield {"stage": "reused", "collection": plan.collection}

    documents = plan.to_upload
    if not documents:
        yield {"stage": "nothing-to-upload", "skipped": list(plan.skipped)}
        return

    # POST /documents refuses a name that is already registered ("Document X
    # already exists. Use update document API instead."), so a registered-but-
    # empty document cannot simply be re-uploaded over -- it has to go first.
    # Without this, stale shells are detected and then never actually repaired.
    stale_here = [d.path.name for d in documents if d.path.name in set(plan.stale)]
    if stale_here:
        yield {"stage": "removing-stale", "documents": stale_here}
        adapter.delete_documents(stale_here)

    batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]
    for index, batch in enumerate(batches, 1):
        yield {"stage": "upload", "batch": index, "of": len(batches),
               "files": [d.path.name for d in batch]}
        # A failed document is recorded and the run continues. Losing an hour
        # of completed uploads because document 2 of 13 hit a dropped
        # connection is the worst possible trade.
        try:
            response = adapter.upload_documents(
                [(d.path, d.metadata) for d in batch], blocking=blocking
            )
        except Exception as exc:  # noqa: BLE001
            yield {
                "stage": "upload-error",
                "batch": index,
                "files": [d.path.name for d in batch],
                "error": f"{type(exc).__name__}: {exc}"[:300],
            }
            continue
        # The server returns a task_id for background ingestion; without it the
        # only way to follow progress is polling collection counts, which cannot
        # tell "still working" from "died quietly".
        event = {
            "stage": "uploaded",
            "batch": index,
            "task_id": (response or {}).get("task_id", ""),
            "files": [d.path.name for d in batch],
            "response": response,
        }
        # Only a blocking upload has finished by the time it returns, so only
        # then is the element count meaningful.
        if blocking:
            states = adapter.document_states()
            event["health"] = [
                check_document(d.path.name, states.get(d.path.name, 0), d.page_count).to_dict()
                for d in batch
            ]
        yield event
