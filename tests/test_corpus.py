"""Hansard PDF corpus: manifest generation and integrity verification."""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import item, ref

from rag_eval.dataset.corpus import (
    CorpusError,
    build_manifest,
    coverage,
    is_pdf,
    manifest_path_for,
    pdf_page_count,
    read_corpus_manifest,
    verify_corpus,
    write_manifest,
)

REAL_CORPUS = Path(__file__).resolve().parent.parent / "datasets" / "hansard_pdfs"


def make_pdf(path: Path, pages: int = 3, body: bytes = b"") -> Path:
    """A minimal but structurally real PDF: a page tree the counter can read."""
    objects = b"".join(
        (b"%d 0 obj\n<< /Type /Page /Parent 2 0 R >>\nendobj\n" % (i + 3))
        for i in range(pages)
    )
    header = b"%PDF-1.7\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    tree = b"2 0 obj\n<< /Type /Pages /Count %d >>\nendobj\n" % pages
    path.write_bytes(header + tree + objects + body + b"trailer\n<< /Root 1 0 R >>\n%%EOF\n")
    return path


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "hansard_pdfs"
    root.mkdir()
    make_pdf(root / "dr_2026-06-22.pdf", pages=155)
    make_pdf(root / "dn_2026-08-04.pdf", pages=135)
    return root


@pytest.fixture
def items():
    return [
        item("q1", ref("dr", "2026-06-22", 3)),
        item("q2", ref("dr", "2026-06-22", 70)),
        item("q3", ref("dn", "2026-08-04", 46)),
        item("q4", None),  # no reference -> irrelevant to the corpus
    ]


# -- page counting ---------------------------------------------------------
def test_page_count_reads_the_page_tree(tmp_path):
    assert pdf_page_count(make_pdf(tmp_path / "a.pdf", pages=22)) == 22


def test_page_count_is_none_when_the_two_signals_disagree(tmp_path):
    # /Count says 9, but only 3 page objects exist. Guessing here would put a
    # wrong number in a manifest that later checks claim to have verified.
    p = tmp_path / "bad.pdf"
    p.write_bytes(
        b"%PDF-1.7\n2 0 obj\n<< /Type /Pages /Count 9 >>\nendobj\n"
        + b"".join((b"%d 0 obj\n<< /Type /Page >>\nendobj\n" % i) for i in range(3))
        + b"%%EOF\n"
    )
    assert pdf_page_count(p) is None


def test_non_pdf_is_rejected_by_magic_bytes_not_just_suffix(tmp_path):
    # A download that failed and saved an HTML error page as .pdf.
    fake = tmp_path / "dr_2026-06-22.pdf"
    fake.write_bytes(b"<!doctype html><title>404</title>")
    assert not is_pdf(fake)
    assert pdf_page_count(fake) is None


# -- manifest building -----------------------------------------------------
def test_manifest_records_checksums_and_golden_linkage(corpus, items):
    manifest = build_manifest(corpus, items, golden_set="golden_v1.jsonl")

    assert len(manifest.documents) == 2
    doc = manifest.by_filename()["dr_2026-06-22.pdf"]
    assert doc.sitting_id == "dr_2026-06-22"
    assert doc.chamber == "Dewan Rakyat"
    assert doc.page_count == 155
    assert len(doc.sha256) == 64
    assert doc.golden_questions == 2
    assert doc.referenced_pages == [3, 70]


def test_manifest_ignores_non_pdf_files(corpus, items):
    # Windows downloads leave :Zone.Identifier streams beside each file.
    (corpus / "dr_2026-06-22.pdf:Zone.Identifier").write_text("[ZoneTransfer]")
    (corpus / "notes.txt").write_text("scratch")
    assert len(build_manifest(corpus, items).documents) == 2


def test_manifest_round_trips_through_json(corpus, items):
    written = write_manifest(corpus, build_manifest(corpus, items))
    assert written == manifest_path_for(corpus)

    reloaded = read_corpus_manifest(corpus)
    assert len(reloaded.documents) == 2
    assert reloaded.by_filename()["dn_2026-08-04.pdf"].page_count == 135


def test_reading_a_missing_manifest_says_how_to_make_one(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(CorpusError, match="--write"):
        read_corpus_manifest(tmp_path / "empty")


# -- verification ----------------------------------------------------------
def verified(corpus, items):
    return verify_corpus(corpus, build_manifest(corpus, items), items)


def test_an_intact_corpus_passes(corpus, items):
    report = verified(corpus, items)
    assert report.ok
    assert report.verified == 2
    assert report.verdict.startswith("PASS")


def test_a_modified_pdf_fails_the_checksum(corpus, items):
    manifest = build_manifest(corpus, items)
    target = corpus / "dr_2026-06-22.pdf"
    target.write_bytes(target.read_bytes() + b"% tampered\n")

    report = verify_corpus(corpus, manifest, items)
    assert not report.ok
    assert any("sha256" in f and "dr_2026-06-22.pdf" in f for f in report.failures)


def test_a_missing_pdf_fails_and_names_the_orphaned_questions(corpus, items):
    manifest = build_manifest(corpus, items)
    (corpus / "dr_2026-06-22.pdf").unlink()

    report = verify_corpus(corpus, manifest, items)
    assert not report.ok
    assert any("missing from" in f for f in report.failures)
    # The point is not that a file vanished; it is that 2 questions are now
    # unanswerable from the corpus.
    assert any("2 golden question(s)" in f for f in report.failures)
    assert coverage(manifest, items, report.present_sittings)["questions_covered"] == 1


def test_an_unmanifested_pdf_fails(corpus, items):
    manifest = build_manifest(corpus, items)
    make_pdf(corpus / "dn_2099-01-01.pdf", pages=4)

    report = verify_corpus(corpus, manifest, items)
    assert not report.ok
    assert any("not in the manifest" in f for f in report.failures)


def test_a_citation_past_the_end_of_the_pdf_fails(corpus):
    # A truncated or wrong-sitting download: page 900 of a 155-page Hansard.
    bad = [item("q1", ref("dr", "2026-06-22", 900))]
    manifest = build_manifest(corpus, bad)

    report = verify_corpus(corpus, manifest, bad)
    assert not report.ok
    assert any("ms. 900" in f and "155 pages" in f for f in report.failures)


def test_a_pdf_nothing_cites_is_a_warning_not_a_failure(corpus, items):
    make_pdf(corpus / "dn_2019-05-06.pdf", pages=117)
    manifest = build_manifest(corpus, items)

    report = verify_corpus(corpus, manifest, items)
    assert report.ok
    assert any("no golden question cites" in w for w in report.warnings)


def test_a_golden_set_that_moved_underneath_the_manifest_warns(corpus, items):
    manifest = build_manifest(corpus, items, golden_set_sha256="a" * 64)
    report = verify_corpus(corpus, manifest, items, golden_set_sha256="b" * 64)

    # The PDFs are fine; it is the recorded linkage that is stale.
    assert report.ok
    assert any("re-run with --write" in w for w in report.warnings)


def test_verification_without_a_golden_set_checks_only_the_files(corpus):
    manifest = build_manifest(corpus)
    assert verify_corpus(corpus, manifest).ok


# -- the real corpus in this repo -----------------------------------------
@pytest.mark.skipif(not REAL_CORPUS.is_dir(), reason="Hansard PDFs not present")
def test_the_committed_manifest_matches_the_real_pdfs():
    from rag_eval.dataset.loader import load_golden_set

    manifest = read_corpus_manifest(REAL_CORPUS)
    golden = REAL_CORPUS.parent / "golden_v1.jsonl"
    items = load_golden_set(golden) if golden.exists() else []

    report = verify_corpus(REAL_CORPUS, manifest, items)
    assert report.ok, report.failures
    # One PDF per sitting the golden set cites, and every cited page in bounds.
    assert report.verified == len(manifest.documents) == 14
    if items:
        assert coverage(manifest, items, report.present_sittings)["coverage"] == 1.0


# -- the manifest layout contract ------------------------------------------
def test_manifest_is_keyed_by_sitting_id(corpus, items):
    # The corpus contract is sitting_id -> filename -> sha256 -> page count,
    # so sitting_id is the key rather than a repeated field in the value.
    payload = build_manifest(corpus, items).to_dict()

    assert set(payload["documents"]) == {"dr_2026-06-22", "dn_2026-08-04"}
    entry = payload["documents"]["dr_2026-06-22"]
    assert list(entry)[:3] == ["filename", "sha256", "page_count"]
    assert entry["filename"] == "dr_2026-06-22.pdf"
    assert entry["page_count"] == 155
    assert "sitting_id" not in entry


def test_pdfs_are_stored_as_sitting_id_dot_pdf(corpus, items):
    manifest = build_manifest(corpus, items)
    for doc in manifest.documents:
        assert doc.filename == f"{doc.sitting_id}.pdf"


def test_a_non_canonical_filename_warns_with_the_name_to_use(tmp_path, items):
    root = tmp_path / "hansard_pdfs"
    root.mkdir()
    make_pdf(root / "KKDR_2026-7-14.pdf", pages=22)  # unnormalised export name
    manifest = build_manifest(root, items)

    assert manifest.documents[0].sitting_id == "kkdr_2026-07-14"
    report = verify_corpus(root, manifest, items)
    assert any("rename it to kkdr_2026-07-14.pdf" in w for w in report.warnings)


def test_two_files_claiming_one_sitting_is_a_hard_error(tmp_path):
    # Keyed by sitting_id, these would overwrite each other and one Hansard
    # would never be checksummed at all.
    root = tmp_path / "hansard_pdfs"
    root.mkdir()
    make_pdf(root / "dr_2026-06-22.pdf", pages=155)
    make_pdf(root / "dr_2026-6-22.pdf", pages=99)

    with pytest.raises(CorpusError, match="both resolve to sitting"):
        build_manifest(root)


def test_a_manifest_written_as_a_list_still_loads(corpus, items):
    # Tolerated so an earlier manifest does not crash the checker.
    from rag_eval.dataset.corpus import CorpusManifest

    payload = build_manifest(corpus, items).to_dict()
    payload["documents"] = [
        {**v, "sitting_id": k} for k, v in payload["documents"].items()
    ]
    reloaded = CorpusManifest.from_dict(payload)

    assert {d.sitting_id for d in reloaded.documents} == {"dr_2026-06-22", "dn_2026-08-04"}
    assert reloaded.by_filename()["dr_2026-06-22.pdf"].page_count == 155


def test_an_empty_corpus_reports_one_actionable_line(corpus, items):
    # The PDFs are gitignored, so this is what a fresh clone hits. Fourteen
    # missing-file failures plus fourteen orphaned-sitting ones would bury the
    # single fact that matters: the corpus has not been fetched.
    manifest = build_manifest(corpus, items)
    for pdf in corpus.glob("*.pdf"):
        pdf.unlink()

    report = verify_corpus(corpus, manifest, items)
    assert not report.ok
    assert report.corpus_empty
    assert len(report.failures) == 1
    assert "not in version control" in report.failures[0]
    assert report.verdict.endswith("corpus not fetched")


def test_a_partially_fetched_corpus_still_names_each_problem(corpus, items):
    # Not empty -- so this is corruption or an interrupted fetch, and the
    # per-document detail is exactly what is wanted.
    manifest = build_manifest(corpus, items)
    (corpus / "dr_2026-06-22.pdf").unlink()

    report = verify_corpus(corpus, manifest, items)
    assert not report.corpus_empty
    assert len(report.failures) == 2  # missing file + the questions it orphans
