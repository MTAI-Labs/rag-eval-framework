"""The Excel reference field is the source of truth for two metrics; parse it exactly."""

import pytest

from rag_eval.dataset.refs import (
    ReferenceParseError,
    normalise_sitting_id,
    parse_pages,
    parse_reference,
)


@pytest.mark.parametrize(
    "raw,sitting,pages",
    [
        ("dr_2026-06-22, ms. 3", "dr_2026-06-22", (3,)),
        ("dn_2026-08-04, ms. 15-16", "dn_2026-08-04", (15, 16)),
        ("kkdr_2026-7-14, ms 1-2, 5", "kkdr_2026-07-14", (1, 2, 5)),
        ("kkdr_2026-7-15, ms 19", "kkdr_2026-07-15", (19,)),
        ("kr_2026-7-14, ms 3", "kkdr_2026-07-14", (3,)),  # known typo, see below
        ("  dr_2004-06-14 , ms. 7 ", "dr_2004-06-14", (7,)),
        ("dn_2019-05-06", "dn_2019-05-06", ()),
    ],
)
def test_parses_every_format_in_the_workbook(raw, sitting, pages):
    parsed = parse_reference(raw)
    assert parsed is not None
    assert parsed.sitting_id == sitting
    assert parsed.pages == pages
    assert parsed.raw == raw.strip()


def test_single_digit_months_normalise_to_iso():
    # kkdr_2026-7-14 and kkdr_2026-07-14 must be the same sitting, or the golden
    # set silently splits one sitting into two retrieval targets.
    assert parse_reference("kkdr_2026-7-14, ms 1").sitting_id == "kkdr_2026-07-14"


def test_empty_reference_is_none_not_an_error():
    assert parse_reference("") is None
    assert parse_reference("   ") is None


def test_unparseable_reference_raises():
    with pytest.raises(ReferenceParseError):
        parse_reference("see the blue folder")


def test_strict_mode_rejects_unknown_prefix():
    assert parse_reference("zz_2026-01-01, ms. 1").doc_type == "zz"
    with pytest.raises(ReferenceParseError):
        parse_reference("zz_2026-01-01, ms. 1", strict=True)


def test_reversed_and_absurd_ranges():
    assert parse_pages("16-15") == (15, 16)
    # A 300-page "range" is a typo; keep the endpoints rather than 300 gold pages.
    assert parse_pages("3-300") == (3, 300)


def test_kr_is_corrected_to_kkdr_not_treated_as_a_sitting_type():
    # `kr_` is a dropped-keystroke `kkdr_`: both occurrences sit inside contiguous
    # kkdr_ blocks with the same owner, date and page. Left alone it invents a
    # sitting no chunk can match, so those questions score a permanent zero.
    parsed = parse_reference("kr_2026-7-14, ms 9")
    assert parsed.sitting_id == "kkdr_2026-07-14"
    assert parsed.doc_type == "kkdr"
    assert parsed.chamber == "Kamar Khas Dewan Rakyat"
    # Provenance survives the correction.
    assert parsed.raw == "kr_2026-7-14, ms 9"


def test_aliases_apply_to_chunk_metadata_too():
    # Otherwise a corrected golden reference could never match an uncorrected chunk.
    assert normalise_sitting_id("kr_2026-7-15.pdf") == "kkdr_2026-07-15"


def test_chamber_names_are_resolved():
    assert parse_reference("dn_2026-08-04, ms. 1").chamber == "Dewan Negara"
    assert parse_reference("kkdr_2026-07-14, ms. 1").chamber == "Kamar Khas Dewan Rakyat"
    assert parse_reference("zz_2026-01-01, ms. 1").chamber == "Unknown"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("KKDR_2026-7-14.pdf", "kkdr_2026-07-14"),
        ("dr_2026-06-22", "dr_2026-06-22"),
        ("dr_2026-6-2_chunk_17", "dr_2026-06-02"),
        (None, None),
        ("", None),
    ],
)
def test_sitting_ids_from_chunk_metadata_are_normalised(raw, expected):
    assert normalise_sitting_id(raw) == expected
