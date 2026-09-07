"""Hansard PDF corpus: manifest generation and integrity verification (D1).

The golden set is versioned and checksummed; the PDFs it points at need the
same treatment, for the same reason. A silently re-downloaded, truncated or
re-paginated Hansard changes what the retriever can possibly find, and every
score compared across that boundary is meaningless. ``corpus-check`` makes that
a loud failure instead of a slow mystery.

The manifest also records the link between each PDF and the golden set — how
many questions cite the sitting, and which pages — so the two artifacts can be
checked against each other rather than drifting apart independently.

Stdlib only: page counting reads the PDF's own page tree rather than taking a
dependency for a number we only need at manifest time.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from rag_eval import __version__
from rag_eval.dataset.loader import sha256_file
from rag_eval.dataset.refs import normalise_sitting_id
from rag_eval.types import DOC_TYPES, GoldenItem

DEFAULT_CORPUS_DIR = Path("datasets/hansard_pdfs")
MANIFEST_NAME = "manifest.json"

_TYPE_PAGE_RE = re.compile(rb"/Type\s*/Page[^s]")
_COUNT_RE = re.compile(rb"/Count\s+(\d+)")
_HEADER_RE = re.compile(rb"^%PDF-(\d+\.\d+)")


class CorpusError(ValueError):
    """The PDF corpus does not match its manifest, or cannot be read."""


# ---------------------------------------------------------------- pdf probing
def pdf_page_count(path: str | Path) -> int | None:
    """Page count read from the PDF's page tree, or ``None`` if uncertain.

    Two independent signals are compared -- the number of ``/Type /Page``
    objects and the page tree's ``/Count`` -- and a count is only returned when
    they agree. Some producers compress the page tree into object streams,
    where neither signal is visible; guessing there would put a wrong number in
    a manifest that later checks claim to have verified, so we return ``None``
    and say the count is unknown.
    """
    data = Path(path).read_bytes()
    if not data.startswith(b"%PDF-"):
        return None

    type_pages = len(_TYPE_PAGE_RE.findall(data))
    counts = [int(m) for m in _COUNT_RE.findall(data)]
    tree_count = max(counts) if counts else None

    if type_pages and tree_count is not None:
        return type_pages if type_pages == tree_count else None
    return type_pages or tree_count or None


def pdf_version(path: str | Path) -> str:
    with open(path, "rb") as fh:
        m = _HEADER_RE.match(fh.read(16))
    return m.group(1).decode() if m else ""


def is_pdf(path: Path) -> bool:
    """Suffix *and* magic bytes -- a renamed HTML error page is not a Hansard."""
    if path.suffix.lower() != ".pdf" or not path.is_file():
        return False
    with open(path, "rb") as fh:
        return fh.read(5) == b"%PDF-"


# ---------------------------------------------------------------- records
@dataclass
class CorpusDocument:
    """One Hansard PDF. Stored as ``<sitting_id>.pdf``; keyed by ``sitting_id``."""

    filename: str
    sitting_id: str
    doc_type: str = ""
    chamber: str = ""
    sitting_date: str = ""
    sha256: str = ""
    bytes: int = 0
    page_count: int | None = None
    pdf_version: str = ""
    golden_questions: int = 0
    referenced_pages: list[int] = field(default_factory=list)

    @property
    def max_referenced_page(self) -> int | None:
        return max(self.referenced_pages) if self.referenced_pages else None

    def to_dict(self) -> dict[str, Any]:
        # sitting_id is the manifest key, so it is not repeated in the value.
        # The first three fields are the chain the corpus contract names:
        # sitting_id -> filename -> sha256 -> page_count.
        return {
            "filename": self.filename,
            "sha256": self.sha256,
            "page_count": self.page_count,
            "doc_type": self.doc_type,
            "chamber": self.chamber,
            "sitting_date": self.sitting_date,
            "bytes": self.bytes,
            "pdf_version": self.pdf_version,
            "golden_questions": self.golden_questions,
            "referenced_pages": {
                "count": len(self.referenced_pages),
                "min": min(self.referenced_pages) if self.referenced_pages else None,
                "max": self.max_referenced_page,
            },
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any], sitting_id: str = "") -> "CorpusDocument":
        pages = d.get("referenced_pages") or {}
        lo, hi = pages.get("min"), pages.get("max")
        return cls(
            filename=d["filename"],
            sitting_id=sitting_id or d.get("sitting_id", ""),
            doc_type=d.get("doc_type", ""),
            chamber=d.get("chamber", ""),
            sitting_date=d.get("sitting_date", ""),
            sha256=d.get("sha256", ""),
            bytes=int(d.get("bytes", 0)),
            page_count=d.get("page_count"),
            pdf_version=d.get("pdf_version", ""),
            golden_questions=int(d.get("golden_questions", 0)),
            # Only the bounds are persisted; they are all the checks need.
            referenced_pages=[lo, hi] if lo is not None and hi is not None else [],
        )


@dataclass
class CorpusManifest:
    directory: str
    documents: list[CorpusDocument] = field(default_factory=list)
    created_at: str = ""
    golden_set: str = ""
    golden_set_sha256: str = ""
    framework_version: str = __version__
    notes: str = ""

    @property
    def total_bytes(self) -> int:
        return sum(d.bytes for d in self.documents)

    def by_filename(self) -> dict[str, CorpusDocument]:
        return {d.filename: d for d in self.documents}

    def to_dict(self) -> dict[str, Any]:
        return {
            "directory": self.directory,
            "created_at": self.created_at,
            "framework_version": self.framework_version,
            "golden_set": self.golden_set,
            "golden_set_sha256": self.golden_set_sha256,
            "document_count": len(self.documents),
            "total_bytes": self.total_bytes,
            "notes": self.notes,
            "documents": {
                d.sitting_id: d.to_dict()
                for d in sorted(self.documents, key=lambda d: d.sitting_id)
            },
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CorpusManifest":
        raw = d.get("documents", {})
        # Keyed by sitting_id. A list is also accepted so a manifest written by
        # an earlier build still loads instead of crashing.
        documents = (
            [CorpusDocument.from_dict(v, k) for k, v in raw.items()]
            if isinstance(raw, dict)
            else [CorpusDocument.from_dict(x) for x in raw]
        )
        return cls(
            directory=d.get("directory", str(DEFAULT_CORPUS_DIR)),
            documents=documents,
            created_at=d.get("created_at", ""),
            golden_set=d.get("golden_set", ""),
            golden_set_sha256=d.get("golden_set_sha256", ""),
            framework_version=d.get("framework_version", ""),
            notes=d.get("notes", ""),
        )


def manifest_path_for(corpus_dir: str | Path) -> Path:
    return Path(corpus_dir) / MANIFEST_NAME


def read_corpus_manifest(corpus_dir: str | Path) -> CorpusManifest:
    path = manifest_path_for(corpus_dir)
    if not path.exists():
        raise CorpusError(
            f"no {MANIFEST_NAME} in {corpus_dir}; create one with "
            f"'rag-eval corpus-check --write'"
        )
    return CorpusManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))


# ---------------------------------------------------------------- build
def scan_pdfs(corpus_dir: str | Path) -> list[Path]:
    """Every readable PDF in the corpus directory, sorted.

    Non-PDF files are ignored rather than reported as corpus members: Windows
    downloads leave ``:Zone.Identifier`` streams beside each file, and those are
    not a corpus problem.
    """
    root = Path(corpus_dir)
    if not root.is_dir():
        raise CorpusError(f"corpus directory not found: {root}")
    return sorted(p for p in root.iterdir() if is_pdf(p))


def build_manifest(
    corpus_dir: str | Path,
    items: Sequence[GoldenItem] = (),
    *,
    golden_set: str = "",
    golden_set_sha256: str = "",
) -> CorpusManifest:
    """Checksum every PDF and link it to the golden set."""
    corpus_dir = Path(corpus_dir)
    pages_by_sitting: dict[str, list[int]] = {}
    questions_by_sitting: dict[str, int] = {}
    for item in items:
        if not item.reference:
            continue
        sitting = item.reference.sitting_id
        questions_by_sitting[sitting] = questions_by_sitting.get(sitting, 0) + 1
        pages_by_sitting.setdefault(sitting, []).extend(item.reference.pages)

    documents = []
    seen: dict[str, str] = {}
    for path in scan_pdfs(corpus_dir):
        sitting_id = normalise_sitting_id(path.stem) or path.stem
        # The manifest is keyed by sitting_id, so two files claiming the same
        # sitting would silently overwrite each other -- and one of the two
        # Hansards would then never be checksummed at all.
        if sitting_id in seen:
            raise CorpusError(
                f"{path.name} and {seen[sitting_id]} both resolve to sitting "
                f"{sitting_id!r}; keep one PDF per sitting, named {sitting_id}.pdf"
            )
        seen[sitting_id] = path.name
        doc_type, _, sitting_date = sitting_id.partition("_")
        documents.append(
            CorpusDocument(
                filename=path.name,
                sitting_id=sitting_id,
                doc_type=doc_type,
                chamber=DOC_TYPES.get(doc_type, "Unknown"),
                sitting_date=sitting_date,
                sha256=sha256_file(path),
                bytes=path.stat().st_size,
                page_count=pdf_page_count(path),
                pdf_version=pdf_version(path),
                golden_questions=questions_by_sitting.get(sitting_id, 0),
                referenced_pages=sorted(set(pages_by_sitting.get(sitting_id, []))),
            )
        )

    return CorpusManifest(
        directory=str(corpus_dir),
        documents=documents,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        golden_set=golden_set,
        golden_set_sha256=golden_set_sha256,
        notes=(
            "Hansard sittings referenced by the golden set. Page counts are read "
            "from each PDF's page tree; 'ms.' references are printed page numbers "
            "and may be offset from physical PDF pages."
        ),
    )


def write_manifest(corpus_dir: str | Path, manifest: CorpusManifest) -> Path:
    path = manifest_path_for(corpus_dir)
    path.write_text(
        json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return path


# ---------------------------------------------------------------- verify
@dataclass
class CorpusReport:
    corpus_dir: str
    documents: int = 0
    verified: int = 0
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checked_bytes: int = 0
    present_sittings: list[str] = field(default_factory=list)
    corpus_empty: bool = False

    @property
    def ok(self) -> bool:
        return not self.failures

    @property
    def verdict(self) -> str:
        if self.corpus_empty:
            return f"FAIL: no Hansard PDFs in {self.corpus_dir} — corpus not fetched"
        if self.failures:
            return f"FAIL: {len(self.failures)} problem(s) in {self.corpus_dir}"
        if self.warnings:
            return f"WARN: {self.verified} document(s) verified, {len(self.warnings)} warning(s)"
        return f"PASS: {self.verified} document(s) verified against the manifest"

    def to_dict(self) -> dict[str, Any]:
        return {
            "corpus_dir": self.corpus_dir,
            "documents": self.documents,
            "verified": self.verified,
            "checked_bytes": self.checked_bytes,
            "present_sittings": self.present_sittings,
            "corpus_empty": self.corpus_empty,
            "failures": self.failures,
            "warnings": self.warnings,
            "verdict": self.verdict,
        }


def verify_corpus(
    corpus_dir: str | Path,
    manifest: CorpusManifest,
    items: Sequence[GoldenItem] = (),
    *,
    golden_set_sha256: str = "",
) -> CorpusReport:
    """Check the PDFs on disk against the manifest, and against the golden set.

    Failures (exit non-zero) are things that make a score wrong or impossible:
    a changed or missing PDF, an unmanifested one, a golden sitting with no
    document, or a citation pointing past the end of its PDF. Warnings are
    things worth a human's attention that do not invalidate a run.
    """
    corpus_dir = Path(corpus_dir)
    report = CorpusReport(corpus_dir=str(corpus_dir), documents=len(manifest.documents))

    on_disk = {p.name: p for p in scan_pdfs(corpus_dir)}
    recorded = manifest.by_filename()

    # The PDFs are not in version control, so a fresh clone has the manifest and
    # none of the documents. That is one actionable fact -- "fetch the corpus" --
    # not fourteen missing-file failures plus fourteen orphaned-sitting ones.
    if recorded and not on_disk:
        report.corpus_empty = True
        report.failures.append(
            f"no PDFs in {corpus_dir} — the corpus is not in version control. "
            f"Fetch the {len(recorded)} Hansard document(s) listed in "
            f"{MANIFEST_NAME} (see datasets/README.md), then re-run."
        )
        return report

    for name in sorted(set(recorded) - set(on_disk)):
        report.failures.append(f"{name}: in the manifest but missing from {corpus_dir}")
    for name in sorted(set(on_disk) - set(recorded)):
        report.failures.append(
            f"{name}: present on disk but not in the manifest "
            f"(re-run with --write after reviewing it)"
        )

    for name in sorted(set(recorded) & set(on_disk)):
        expected, path = recorded[name], on_disk[name]
        size = path.stat().st_size
        actual = sha256_file(path)
        report.checked_bytes += size

        if actual != expected.sha256:
            report.failures.append(
                f"{name}: sha256 {actual[:12]}… does not match the manifest's "
                f"{expected.sha256[:12]}… — the PDF changed since it was recorded"
            )
            continue
        if size != expected.bytes:
            report.failures.append(
                f"{name}: {size} bytes on disk, manifest records {expected.bytes}"
            )
            continue

        pages = pdf_page_count(path)
        if expected.page_count is not None and pages is not None and pages != expected.page_count:
            report.failures.append(
                f"{name}: {pages} pages, manifest records {expected.page_count}"
            )
            continue
        if pages is None:
            report.warnings.append(
                f"{name}: page count could not be read (compressed page tree); "
                f"page-bounds checks are skipped for this document"
            )
        report.verified += 1

    if (
        golden_set_sha256
        and manifest.golden_set_sha256
        and golden_set_sha256 != manifest.golden_set_sha256
    ):
        # Not a failure: the PDFs can be perfectly intact while the golden set
        # moves underneath them. But the question counts and page bounds in this
        # manifest describe a different golden set, so they are stale.
        report.warnings.append(
            f"the manifest was built against golden set "
            f"{manifest.golden_set_sha256[:12]}… but {manifest.golden_set or 'the current one'} "
            f"is now {golden_set_sha256[:12]}… — re-run with --write to refresh the linkage"
        )

    # Cross-checks run against what is actually on disk, not merely what the
    # manifest claims: a manifested-but-missing PDF must read as "these
    # questions are unanswerable", not as full coverage.
    present = {
        doc.sitting_id for name, doc in recorded.items() if name in on_disk
    }
    report.present_sittings = sorted(present)
    report.failures.extend(_golden_set_problems(manifest, items, report, present))
    return report


def _golden_set_problems(
    manifest: CorpusManifest,
    items: Sequence[GoldenItem],
    report: CorpusReport,
    present: set[str],
) -> list[str]:
    """Cross-check the corpus against what the golden set actually cites."""
    if not items:
        return []

    problems: list[str] = []
    by_sitting = manifest_by_sitting(manifest)

    cited_pages: dict[str, set[int]] = {}
    for item in items:
        if item.reference:
            cited_pages.setdefault(item.reference.sitting_id, set()).update(item.reference.pages)

    for sitting in sorted(set(cited_pages) - present):
        n = sum(1 for i in items if i.reference and i.reference.sitting_id == sitting)
        problems.append(
            f"{sitting}: {n} golden question(s) cite this sitting but no PDF for it is present "
            f"— those questions can never be answered from the corpus"
        )

    for sitting, pages in sorted(cited_pages.items()):
        doc = by_sitting.get(sitting)
        if not doc or doc.page_count is None or not pages:
            continue
        beyond = sorted(p for p in pages if p > doc.page_count)
        if beyond:
            problems.append(
                f"{doc.filename}: golden set cites ms. {', '.join(map(str, beyond[:5]))} "
                f"but the PDF has {doc.page_count} pages"
            )

    for sitting in sorted(present - set(cited_pages)):
        report.warnings.append(
            f"{by_sitting[sitting].filename}: no golden question cites this sitting "
            f"(ingesting it is harmless, but it is not being evaluated)"
        )

    for doc in manifest.documents:
        if doc.filename != f"{doc.sitting_id}.pdf":
            report.warnings.append(
                f"{doc.filename}: corpus convention is <sitting_id>.pdf — "
                f"rename it to {doc.sitting_id}.pdf"
            )
        if doc.doc_type not in DOC_TYPES:
            report.warnings.append(
                f"{doc.filename}: unrecognised document prefix {doc.doc_type!r} "
                f"— confirm the sitting type before ingest"
            )
    return problems


def manifest_by_sitting(manifest: CorpusManifest) -> dict[str, CorpusDocument]:
    return {d.sitting_id: d for d in manifest.documents}


def coverage(
    manifest: CorpusManifest,
    items: Iterable[GoldenItem],
    present: Sequence[str] | None = None,
) -> dict[str, Any]:
    """How much of the golden set the corpus can actually serve.

    ``present`` restricts the count to sittings whose PDF is on disk; without it
    the manifest is taken at its word.
    """
    by_sitting = set(present) if present is not None else set(manifest_by_sitting(manifest))
    total = covered = 0
    missing: dict[str, int] = {}
    for item in items:
        if not item.reference:
            continue
        total += 1
        if item.reference.sitting_id in by_sitting:
            covered += 1
        else:
            missing[item.reference.sitting_id] = missing.get(item.reference.sitting_id, 0) + 1
    return {
        "documents": len(by_sitting),
        "questions_with_a_reference": total,
        "questions_covered": covered,
        "coverage": round(covered / total, 4) if total else None,
        "missing_sittings": dict(sorted(missing.items())),
    }
