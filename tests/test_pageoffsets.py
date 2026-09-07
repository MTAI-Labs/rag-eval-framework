"""Measuring the printed-ms. vs physical-page offset."""

from __future__ import annotations

import zlib

from conftest import item, ref

from rag_eval.dataset.pageoffsets import (
    MIN_SAMPLES,
    OffsetResult,
    distinctive_fragments,
    measure,
    offsets_payload,
    page_texts,
)


def pdf_with_pages(path, texts):
    """A PDF whose page N contains texts[N-1], with real Flate content streams."""
    objs, body = [], b""
    header = b"%PDF-1.7\n"
    num = 3
    for text in texts:
        content = f"BT /F1 12 Tf 1 0 0 1 72 700 Tm ({text}) Tj ET".encode("latin-1")
        stream = zlib.compress(content)
        page = (b"%d 0 obj\n<< /Type /Page /Parent 2 0 R /Contents %d 0 R >>\nendobj\n"
                % (num, num + 1))
        cobj = (b"%d 0 obj\n<< /Length %d /Filter /FlateDecode >>\nstream\n"
                % (num + 1, len(stream))) + stream + b"\nendstream\nendobj\n"
        body += page + cobj
        objs.append(num)
        num += 2
    tree = b"2 0 obj\n<< /Type /Pages /Count %d >>\nendobj\n" % len(texts)
    path.write_bytes(header + b"1 0 obj\n<< /Type /Catalog >>\nendobj\n" + tree + body
                     + b"trailer\n<< /Root 1 0 R >>\n%%EOF\n")
    return path


def test_page_texts_reads_each_page_in_order(tmp_path):
    pdf = pdf_with_pages(tmp_path / "a.pdf", ["cover", "contents", "first real page"])
    texts = page_texts(pdf)
    assert len(texts) == 3
    assert "first real page" in texts[2]


def test_measures_a_known_offset(tmp_path):
    # Two cover pages, so printed page 1 sits on physical page 3: offset 2.
    answers = [f"Jumlah keseluruhan adalah {i}00 orang" for i in range(1, 9)]
    pdf = pdf_with_pages(tmp_path / "kkdr_2026-07-14.pdf", ["cover", "kandungan"] + answers)
    items = [
        item(f"q{i}", ref("kkdr", "2026-07-14", i + 1))
        for i in range(len(answers))
    ]
    for i, a in enumerate(answers):
        items[i].expected_answer = a

    result = measure(pdf, "kkdr_2026-07-14", items)
    assert result.offset == 2
    assert result.confident
    assert result.agreement == 1.0
    assert result.samples >= MIN_SAMPLES


def test_too_few_samples_is_not_confident(tmp_path):
    pdf = pdf_with_pages(tmp_path / "dr_2004-06-14.pdf", ["cover", "Peruntukan khas RM50 juta"])
    it = item("q1", ref("dr", "2004-06-14", 1))
    it.expected_answer = "Peruntukan khas RM50 juta"

    result = measure(pdf, "dr_2004-06-14", [it])
    assert result.offset == 1          # a single vote exists…
    assert not result.confident        # …but one sample is not evidence
    assert "below the confidence bar" in result.note


def test_a_sitting_nothing_cites_is_reported_not_guessed(tmp_path):
    pdf = pdf_with_pages(tmp_path / "dn_2019-05-06.pdf", ["cover", "text"])
    result = measure(pdf, "dn_2019-05-06", [item("q1", ref("dr", "2026-06-22", 3))])

    assert result.offset is None
    assert result.note == "no golden question cites this sitting"


def test_payload_separates_confident_from_undetermined(tmp_path):
    good = pdf_with_pages(tmp_path / "kkdr_2026-07-14.pdf",
                          ["cover", "x"] + [f"Bilangan {i}00 kes" for i in range(1, 9)])
    items = []
    for i in range(8):
        it = item(f"q{i}", ref("kkdr", "2026-07-14", i + 1))
        it.expected_answer = f"Bilangan {i + 1}00 kes"
        items.append(it)
    confident = measure(good, "kkdr_2026-07-14", items)
    unknown = measure(pdf_with_pages(tmp_path / "dn_2019-05-06.pdf", ["a"]),
                      "dn_2019-05-06", items)

    payload = offsets_payload([confident, unknown])
    assert payload["offsets"] == {"kkdr_2026-07-14": confident.offset}
    assert payload["undetermined"] == ["dn_2019-05-06"]
    # Every document appears in the evidence, measured or not.
    assert len(payload["evidence"]) == 2


def test_distinctive_fragments_prefer_phrases_then_numbers_and_names():
    frags = distinctive_fragments(
        "Jumlah keseluruhan setakat ini adalah 3,404 orang di Pandan."
    )
    assert any("Jumlah keseluruhan setakat ini adalah" in f for f in frags)
    assert "3,404" in frags
    assert "Pandan" in frags


# -- human-verified offsets ------------------------------------------------
def measured(sitting: str, offset, samples: int, agreement: float) -> OffsetResult:
    return OffsetResult(sitting_id=sitting, filename=f"{sitting}.pdf", offset=offset,
                        samples=samples, agreement=agreement)


def test_a_verified_offset_rescues_an_undetermined_document():
    # A human reading "ms. 1 is on PDF page 13" is better evidence than any
    # amount of text matching, and must not be discarded for missing the bar.
    low = measured("dr_2004-06-14", 12, samples=2, agreement=1.0)
    assert not low.confident

    payload = offsets_payload([low], {"dr_2004-06-14": 12})
    assert payload["offsets"]["dr_2004-06-14"] == 12
    assert payload["undetermined"] == []
    assert payload["verified"] == {"dr_2004-06-14": 12}


def test_a_verified_offset_overrides_a_confident_measurement():
    wrong = measured("dn_2026-08-03", 4, samples=20, agreement=0.9)
    assert wrong.confident

    payload = offsets_payload([wrong], {"dn_2026-08-03": 5})
    assert payload["offsets"]["dn_2026-08-03"] == 5
    row = payload["evidence"][0]
    assert row["source"] == "human-verified"
    assert row["agrees_with_measurement"] is False
    assert row["offset"] == 4          # the measurement is kept, not erased


def test_agreement_between_human_and_measurement_is_recorded():
    payload = offsets_payload([measured("kkdr_2026-07-14", 2, 16, 0.75)],
                              {"kkdr_2026-07-14": 2})
    assert payload["evidence"][0]["agrees_with_measurement"] is True


def test_evidence_labels_where_each_offset_came_from():
    payload = offsets_payload(
        [measured("a", 5, 20, 0.9), measured("b", 3, 1, 1.0), measured("c", 7, 30, 0.8)],
        {"c": 7},
    )
    assert {e["sitting_id"]: e["source"] for e in payload["evidence"]} == {
        "a": "measured", "b": "undetermined", "c": "human-verified",
    }


def test_verified_block_round_trips_through_a_file(tmp_path):
    import json

    from rag_eval.dataset.pageoffsets import read_verified

    path = tmp_path / "offsets.json"
    path.write_text(json.dumps(offsets_payload(
        [measured("dr_2004-06-14", 12, 2, 1.0)], {"dr_2004-06-14": 12}
    )))
    # Re-measuring must carry the human's work forward, not silently drop it.
    assert read_verified(path) == {"dr_2004-06-14": 12}
    assert read_verified(tmp_path / "absent.json") == {}


def test_a_corrupt_offsets_file_yields_no_verified_entries(tmp_path):
    path = tmp_path / "offsets.json"
    path.write_text("{not json")
    from rag_eval.dataset.pageoffsets import read_verified
    assert read_verified(path) == {}
