"""Golden-set conversion, schema validation and checksum enforcement."""

import json
from pathlib import Path

import pytest

from rag_eval.dataset.convert import convert_workbook
from rag_eval.dataset.loader import (
    DatasetError,
    dataset_checksum,
    load_golden_set,
    read_manifest,
    write_golden_set,
)

WORKBOOK = Path(__file__).resolve().parent.parent / "datasets" / "TanyaParlimen_QnA.xlsx"


def test_round_trips_through_jsonl(tmp_path, golden_items):
    path = tmp_path / "golden_v1.jsonl"
    write_golden_set(path, golden_items, source="synthetic")

    loaded = load_golden_set(path)
    assert [i.id for i in loaded] == [i.id for i in golden_items]
    assert loaded[1].reference.pages == (15, 16)
    assert loaded[3].reference is None


def test_manifest_records_provenance(tmp_path, golden_items):
    path = tmp_path / "golden_v1.jsonl"
    manifest = write_golden_set(path, golden_items, source="book.xlsx", source_sha256="abc")

    on_disk = read_manifest(path)
    assert on_disk.sha256 == manifest.sha256 == dataset_checksum(path)
    assert on_disk.count == len(golden_items)
    assert on_disk.source == "book.xlsx"


def test_an_edited_golden_set_is_refused(golden_file):
    # The whole point of the checksum: a quiet edit must not silently move scores.
    with open(golden_file, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"id": "sneaky", "question": "q", "expected_answer": "a"}) + "\n")

    with pytest.raises(DatasetError, match="does not match its manifest"):
        load_golden_set(golden_file)

    assert len(load_golden_set(golden_file, verify_checksum=False)) == 5


def test_missing_manifest_is_refused_unless_opted_out(tmp_path, golden_items):
    path = tmp_path / "bare.jsonl"
    write_golden_set(path, golden_items)
    path.with_suffix("").with_suffix(".manifest.json").unlink()

    with pytest.raises(DatasetError, match="no manifest"):
        load_golden_set(path)
    assert load_golden_set(path, verify_checksum=False)


def test_schema_violations_name_the_line(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(
        json.dumps({"id": "a", "question": "q", "expected_answer": "a"}) + "\n"
        + json.dumps({"id": "b", "question": "q"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(DatasetError, match="line 2: missing required field"):
        load_golden_set(path, verify_checksum=False)


def test_duplicate_ids_are_refused(tmp_path):
    path = tmp_path / "dupe.jsonl"
    row = json.dumps({"id": "a", "question": "q", "expected_answer": "a"})
    path.write_text(row + "\n" + row + "\n", encoding="utf-8")

    with pytest.raises(DatasetError, match="duplicate question id"):
        load_golden_set(path, verify_checksum=False)


def test_slicing_by_limit_and_ids(golden_file):
    assert len(load_golden_set(golden_file, limit=2)) == 2
    assert [i.id for i in load_golden_set(golden_file, ids=["q3", "q1"])] == ["q1", "q3"]
    with pytest.raises(DatasetError, match="unknown question id"):
        load_golden_set(golden_file, ids=["nope"])


@pytest.mark.skipif(not WORKBOOK.exists(), reason="source workbook not in the repo")
def test_converts_the_real_workbook(tmp_path):
    out = tmp_path / "golden_v1.jsonl"
    items, report, manifest = convert_workbook(WORKBOOK, out)

    # The design spec commits to 371 Q&A pairs; if that moves, the golden set
    # changed and every historical score is suspect.
    assert report.converted == 371
    assert report.unparsed_reference == []
    assert manifest.count == 371

    # 14, not the spec's 16: the two `kr_` references were a typo of `kkdr_` on
    # dates that already had a kkdr_ sitting, so they were never separate
    # sittings and never two more Hansard PDFs to collect. See datasets/README.md.
    assert len(report.sittings) == 14

    # The workbook has since been corrected at source, so nothing needs aliasing
    # any more. DOC_TYPE_ALIASES stays as a guard against the typo reappearing —
    # this asserts the source is clean, not that the guard was removed.
    assert report.aliased_doc_types == []

    loaded = load_golden_set(out)
    assert len(loaded) == 371
    assert all(i.question and i.expected_answer for i in loaded)
    assert {i.reference.doc_type for i in loaded if i.reference} == {"dr", "dn", "kkdr"}


@pytest.mark.skipif(not WORKBOOK.exists(), reason="source workbook not in the repo")
def test_conversion_is_deterministic(tmp_path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    _, _, first = convert_workbook(WORKBOOK, a)
    _, _, second = convert_workbook(WORKBOOK, b)
    assert first.sha256 == second.sha256
