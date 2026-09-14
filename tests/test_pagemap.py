"""Verifying that a chunk's page number means the golden set's ms. value."""

from __future__ import annotations

from conftest import item, ref

from rag_eval import pagemap
from rag_eval.types import RagTrace, RetrievedChunk


def chunk(rank, sitting, page, text):
    return RetrievedChunk(rank=rank, text=text, chunk_id=f"c{rank}",
                          score=1.0 / rank, sitting_id=sitting, page=page)


class FakeAdapter:
    """Returns canned chunks; `retrieve` is preferred over `answer` by the check."""
    name = "fake"

    def __init__(self, chunks, error=None):
        self.chunks, self.error, self.calls = chunks, error, []

    def retrieve(self, question, question_id=""):
        self.calls.append(question_id)
        t = RagTrace(question_id=question_id, question=question, adapter=self.name)
        t.error = self.error
        t.retrieved_chunks = list(self.chunks)
        return t


def gold(qid="tp-0003", sitting=("dn", "2026-08-04"), ms=129, answer="Jumlah keseluruhan 3,404 orang"):
    g = item(qid, ref(sitting[0], sitting[1], ms))
    g.expected_answer = answer
    return g


# -- header parsing --------------------------------------------------------
def test_header_gives_the_printed_page():
    assert pagemap.header_page("DN 4.8.2026 129\n\nPengurusan Dewan") == 129
    assert pagemap.header_page("DR.14.6.2004 3 Kawasan tanggungjawab") == 3
    assert pagemap.header_page("KKDR.14.7.2026 3 Antara kesalahan") == 3


def test_roman_front_matter_is_not_a_page():
    # "DN 26.2.2026 iv" is front matter, not an ms. page. Reading it as one
    # would silently corrupt the comparison.
    assert pagemap.header_page("DN 26.2.2026 iv\n\nDEWAN NEGARA") is None
    assert pagemap.header_page("no running header at all") is None


# -- the three-way comparison ---------------------------------------------
def test_a_correct_mapping_matches():
    # page_number 133 (0-based) + 1 - offset 5 = 129, and the header says 129.
    a = FakeAdapter([chunk(1, "dn_2026-08-04", 133,
                           "DN 4.8.2026 129 Jumlah keseluruhan 3,404 orang")])
    r = pagemap.probe_one(a, gold(), {"dn_2026-08-04": 5})

    assert r.verdict == "match"
    assert (r.header, r.computed, r.excel_ms) == (129, 129, 129)
    assert r.matched_on == "answer"
    assert r.observed_base == 1


def test_a_wrong_offset_is_a_formula_mismatch():
    # Offset 4 instead of 5 -> computed 130 while the page itself says 129.
    a = FakeAdapter([chunk(1, "dn_2026-08-04", 133,
                           "DN 4.8.2026 129 Jumlah keseluruhan 3,404 orang")])
    r = pagemap.probe_one(a, gold(), {"dn_2026-08-04": 4})

    assert r.verdict == "formula-mismatch"
    assert r.observed_base == 0        # the data says the assumed base is wrong


def test_a_wrong_golden_reference_is_reported_separately():
    # Formula is right; it is the Excel ms. that disagrees. Different fault,
    # different fix — so it must not be collapsed into "mismatch".
    a = FakeAdapter([chunk(1, "dn_2026-08-04", 133,
                           "DN 4.8.2026 129 Jumlah keseluruhan 3,404 orang")])
    r = pagemap.probe_one(a, gold(ms=42), {"dn_2026-08-04": 5})

    assert r.verdict == "golden-mismatch"
    assert (r.header, r.computed, r.excel_ms) == (129, 129, 42)


def test_the_chunk_containing_the_answer_is_preferred():
    # Two chunks from the right sitting; only one holds the golden answer.
    a = FakeAdapter([
        chunk(1, "dn_2026-08-04", 10, "DN 4.8.2026 6 unrelated debate"),
        chunk(2, "dn_2026-08-04", 133, "DN 4.8.2026 129 Jumlah keseluruhan 3,404 orang"),
    ])
    r = pagemap.probe_one(a, gold(), {"dn_2026-08-04": 5})

    assert r.matched_on == "answer"
    assert r.page_number == 133


def test_a_chunk_without_a_header_cannot_be_verified():
    a = FakeAdapter([chunk(1, "dn_2026-08-04", 133, "Jumlah keseluruhan 3,404 orang")])
    r = pagemap.probe_one(a, gold(), {"dn_2026-08-04": 5})
    assert r.verdict == "no-header"


def test_a_miss_on_the_gold_sitting_is_not_a_mapping_failure():
    # Retrieval failing to surface the right sitting is a retrieval result, not
    # evidence about page mapping — it must not be counted as a mismatch.
    a = FakeAdapter([chunk(1, "dr_2026-06-22", 9, "DR.22.6.2026 3 something else")])
    r = pagemap.probe_one(a, gold(), {"dn_2026-08-04": 5})

    assert r.verdict == "no-hit"
    assert "no chunk from dn_2026-08-04" in r.note


def test_an_adapter_error_is_recorded_not_raised():
    a = FakeAdapter([], error="HTTP 503")
    r = pagemap.probe_one(a, gold(), {})
    assert r.verdict == "no-hit"
    assert "503" in r.note


# -- the summary -----------------------------------------------------------
def test_probes_spread_one_per_sitting():
    items = [gold(f"q{i}", ("dn", "2026-08-04"), 1) for i in range(3)]
    items += [gold(f"r{i}", ("dr", "2026-06-22"), 1) for i in range(3)]
    chosen = pagemap.select_probes(items, sample=5)
    assert {i.reference.sitting_id for i in chosen} == {"dn_2026-08-04", "dr_2026-06-22"}


def test_summary_reports_the_observed_base_not_the_assumed_one():
    a = FakeAdapter([chunk(1, "dn_2026-08-04", 133,
                           "DN 4.8.2026 129 Jumlah keseluruhan 3,404 orang")])
    report = pagemap.check(a, [gold()], {"dn_2026-08-04": 4}, sample=1)

    assert report["observed_bases"] == {"0": 1}
    assert report["base_consistent"] is False
    assert report["verdict"].startswith("FAIL")


def test_inconclusive_when_nothing_is_comparable():
    a = FakeAdapter([chunk(1, "dr_2026-06-22", 9, "wrong sitting")])
    report = pagemap.check(a, [gold()], {"dn_2026-08-04": 5}, sample=1)
    assert report["verdict"].startswith("INCONCLUSIVE")
