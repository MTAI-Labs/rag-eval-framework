"""Building the eval collection: plan, metadata schema, and the upload encoding."""

from __future__ import annotations

import json

import pytest
from conftest import item, ref
from test_corpus import make_pdf

from rag_eval.adapters.multipart import encode
from rag_eval.dataset.corpus import build_manifest
from rag_eval.ingest import METADATA_SCHEMA, build_plan, execute, reconcile


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "hansard_pdfs"
    root.mkdir()
    make_pdf(root / "dr_2026-06-22.pdf", pages=155)
    make_pdf(root / "kkdr_2026-07-14.pdf", pages=22)
    return root


@pytest.fixture
def items():
    return [
        item("q1", ref("dr", "2026-06-22", 3)),
        item("q2", ref("dr", "2026-06-22", 9)),
        item("q3", ref("kkdr", "2026-07-14", 1)),
    ]


def plan_for(corpus, items, **kw):
    return build_plan("parliament_hansard_eval", corpus,
                      build_manifest(corpus, items), items, **kw)


# -- the schema is the part that decides whether metrics are computable ----
def test_schema_declares_the_fields_every_metric_depends_on():
    names = {f["name"] for f in METADATA_SCHEMA}
    # Without sitting_id no retrieval metric can be computed at all, and the
    # field cannot be added after the collection is created.
    assert "sitting_id" in names
    assert {"doc_type", "session_date"} <= names

    sitting = next(f for f in METADATA_SCHEMA if f["name"] == "sitting_id")
    assert sitting["required"] is True
    assert sitting["type"] == "string"


def test_schema_carries_ms_offset_rather_than_a_page_field():
    # nv-ingest auto-extracts page_number as the PHYSICAL page; declaring our own
    # would collide. ms_offset is what lets a consumer recover the printed page.
    names = {f["name"] for f in METADATA_SCHEMA}
    assert "page_number" not in names
    assert "ms_offset" in names


# -- planning --------------------------------------------------------------
def test_plan_derives_metadata_from_the_sitting_id(corpus, items):
    plan = plan_for(corpus, items)
    doc = next(d for d in plan.documents if d.sitting_id == "dr_2026-06-22")

    assert doc.metadata == {
        "sitting_id": "dr_2026-06-22",
        "doc_type": "dr",
        "dewan": "dewan rakyat",
        "session_date": "2026-06-22",
    }
    assert doc.golden_questions == 2
    assert plan.problems == []


def test_kkdr_maps_to_its_chamber_name(corpus, items):
    plan = plan_for(corpus, items)
    doc = next(d for d in plan.documents if d.sitting_id.startswith("kkdr"))
    assert doc.metadata["dewan"] == "kamar khas dewan rakyat"


def test_offsets_are_stamped_onto_the_documents(corpus, items):
    plan = plan_for(corpus, items, offsets={"dr_2026-06-22": 6, "kkdr_2026-07-14": 2})
    by = {d.sitting_id: d for d in plan.documents}

    assert by["dr_2026-06-22"].metadata["ms_offset"] == 6
    assert by["kkdr_2026-07-14"].metadata["ms_offset"] == 2


def test_a_sitting_with_no_pdf_is_a_planning_problem(corpus, items):
    orphan = items + [item("q9", ref("dn", "2026-08-04", 1))]
    plan = build_plan("c", corpus, build_manifest(corpus, orphan), orphan)

    assert any("dn_2026-08-04" in p and "no document is planned" in p for p in plan.problems)


def test_the_plan_follows_the_manifest_not_the_directory(corpus, items):
    # Only bytes that were checksummed get uploaded, so the same manifest always
    # produces the same collection.
    manifest = build_manifest(corpus, items)
    make_pdf(corpus / "dn_2099-01-01.pdf", pages=3)   # dropped in after checksumming

    plan = build_plan("c", corpus, manifest, items)
    assert {d.path.name for d in plan.documents} == {
        "dr_2026-06-22.pdf", "kkdr_2026-07-14.pdf"
    }


# -- execution order -------------------------------------------------------
class FakeAdapter:
    """Models the server's ordering rule: documents need a collection to exist.

    ``exists`` is whether the collection is already there before this run, so
    the reuse path can upload without creating while the create path still
    cannot upload first.
    """

    collection = "parliament_hansard_eval"

    def __init__(self, exists: bool = False):
        self.exists = exists
        self.calls: list[str] = []
        self.uploaded: list[tuple[str, dict]] = []

    def create_collection(self, schema, **kw):
        if self.exists:
            raise AssertionError("created a collection that already existed")
        self.calls.append("create")
        self.exists = True
        self.schema, self.kw = schema, kw
        return {"message": "created"}

    def upload_documents(self, files, **kw):
        if not self.exists:
            raise AssertionError("uploaded before the collection existed")
        self.calls.append("upload")
        self.uploaded += [(p.name, m) for p, m in files]
        return {"message": "ok", "total_documents": len(files),
                "documents_completed": len(files)}


def test_the_collection_is_created_before_any_upload(corpus, items):
    adapter = FakeAdapter()
    events = list(execute(adapter, plan_for(corpus, items), batch_size=1))

    assert adapter.calls[0] == "create"
    assert adapter.calls.count("upload") == 2
    assert [e["stage"] for e in events[:2]] == ["create", "created"]
    assert len(adapter.uploaded) == 2


def test_uploads_are_batched(corpus, items):
    adapter = FakeAdapter()
    list(execute(adapter, plan_for(corpus, items), batch_size=5))
    assert adapter.calls == ["create", "upload"]  # both documents in one batch


def test_each_uploaded_document_carries_its_metadata(corpus, items):
    adapter = FakeAdapter()
    list(execute(adapter, plan_for(corpus, items, offsets={"dr_2026-06-22": 6})))

    meta = dict(adapter.uploaded)["dr_2026-06-22.pdf"]
    assert meta["sitting_id"] == "dr_2026-06-22"
    assert meta["ms_offset"] == 6


# -- multipart encoding ----------------------------------------------------
def test_multipart_carries_both_the_json_and_the_file(tmp_path):
    pdf = make_pdf(tmp_path / "dr_2026-06-22.pdf", pages=2)
    body, ctype = encode([("data", json.dumps({"collection_name": "c"}))],
                         [("documents", pdf)])

    assert ctype.startswith("multipart/form-data; boundary=")
    boundary = ctype.split("boundary=")[1]
    assert body.count(boundary.encode()) == 3          # two parts + terminator
    assert b'name="data"' in body and b'name="documents"' in body
    assert b'filename="dr_2026-06-22.pdf"' in body
    assert b"application/pdf" in body
    assert pdf.read_bytes() in body                     # file sent byte-exact
    assert body.endswith(f"--{boundary}--\r\n".encode())


# -- one collection, created once or reused -------------------------------
def status(exists: bool, fields=None, entities=0) -> dict:
    return {"checked": True, "exists": exists, "collection": "parliament_hansard_eval",
            "entities": entities,
            "metadata_fields": fields if fields is not None
            else [f["name"] for f in METADATA_SCHEMA]}


def test_one_create_call_for_the_whole_corpus(corpus, items):
    # All Hansard PDFs live in a single collection; create_collection must be
    # called exactly once no matter how many documents or batches there are.
    adapter = FakeAdapter()
    list(execute(adapter, plan_for(corpus, items), batch_size=1))
    assert adapter.calls.count("create") == 1


def test_a_missing_collection_is_created(corpus, items):
    plan = reconcile(plan_for(corpus, items), status(exists=False))
    assert plan.must_create is True
    assert plan.action == "create"
    assert plan.problems == []


def test_an_existing_collection_is_reused_not_recreated(corpus, items):
    plan = reconcile(plan_for(corpus, items), status(exists=True, entities=4200))
    adapter = FakeAdapter(exists=True)
    events = [e["stage"] for e in execute(adapter, plan)]

    assert plan.action == "reuse"
    assert "create" not in adapter.calls
    assert "reused" in events
    assert adapter.calls == ["upload"]


def test_documents_already_present_are_skipped(corpus, items):
    plan = reconcile(plan_for(corpus, items), status(exists=True),
                     ["dr_2026-06-22.pdf"])
    assert plan.skipped == ["dr_2026-06-22.pdf"]
    assert [d.path.name for d in plan.to_upload] == ["kkdr_2026-07-14.pdf"]


def test_replace_re_uploads_everything(corpus, items):
    plan = reconcile(plan_for(corpus, items), status(exists=True),
                     ["dr_2026-06-22.pdf"], replace=True)
    assert plan.skipped == []
    assert len(plan.to_upload) == 2


def test_a_fully_ingested_collection_uploads_nothing(corpus, items):
    plan = reconcile(plan_for(corpus, items), status(exists=True),
                     ["dr_2026-06-22.pdf", "kkdr_2026-07-14.pdf"])
    adapter = FakeAdapter(exists=True)
    events = [e["stage"] for e in execute(adapter, plan)]

    assert plan.to_upload == []
    assert adapter.calls == []          # neither created nor uploaded
    assert events == ["reused", "nothing-to-upload"]


def test_reuse_refuses_a_collection_missing_required_schema_fields(corpus, items):
    # The schema cannot be patched after creation, so ingesting into a
    # collection without sitting_id would produce a permanently unscoreable
    # corpus. That has to stop the ingest, not warn.
    plan = reconcile(plan_for(corpus, items),
                     status(exists=True, fields=["filename", "session_date"]))

    assert any("sitting_id" in p and "cannot be changed" in p for p in plan.problems)


def test_reuse_warns_about_missing_optional_fields(corpus, items):
    plan = reconcile(plan_for(corpus, items),
                     status(exists=True,
                            fields=["sitting_id", "doc_type", "session_date"]))
    assert any("ms_offset" in p for p in plan.problems)


def test_an_unverifiable_collection_is_a_problem_not_an_assumption(corpus, items):
    # No ingest URL configured: we do not know whether it exists, so we must not
    # guess "create" and blow up mid-run, nor guess "reuse" and upload nowhere.
    plan = reconcile(plan_for(corpus, items),
                     {"checked": False, "reason": "NVIDIA_INGEST_BASE_URL is not set"})
    assert any("could not confirm" in p for p in plan.problems)


# -- offsets file handling -------------------------------------------------
def test_offsets_accept_both_the_bare_map_and_the_generated_file(corpus, items):
    bare = plan_for(corpus, items, offsets={"dr_2026-06-22": 6})
    rich = plan_for(corpus, items, offsets={
        "_comment": "…", "offsets": {"dr_2026-06-22": 6}, "undetermined": [], "evidence": [],
    })
    for plan in (bare, rich):
        doc = next(d for d in plan.documents if d.sitting_id == "dr_2026-06-22")
        assert doc.metadata["ms_offset"] == 6


def test_a_document_without_a_measured_offset_carries_none(corpus, items):
    # Better an absent ms_offset than a guessed one: a wrong offset silently
    # maps every citation to the wrong page.
    plan = plan_for(corpus, items, offsets={"dr_2026-06-22": 6})
    doc = next(d for d in plan.documents if d.sitting_id == "kkdr_2026-07-14")
    assert "ms_offset" not in doc.metadata
