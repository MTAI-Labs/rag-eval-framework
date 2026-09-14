"""Verify that a retrieved chunk's page maps back to the golden set's ``ms.``.

``ingest-check`` proves chunks *carry* a page number. This proves that number
*means* what the golden set says — which is what ``page_citation_accuracy``
actually depends on, and a different claim entirely.

Three independent values are compared per probe:

``header``    the printed page read out of the chunk's own running header
              ("DN 4.8.2026 129"), which survives ingestion and is ground truth
``computed``  ``page_number + PAGE_BASE - ms_offset``, i.e. what our arithmetic
              predicts
``excel``     the ``ms.`` the golden set records

``header`` vs ``computed`` tests the formula, including whether nv-ingest's
``page_number`` is 0-based. ``header`` vs ``excel`` tests whether the golden
set's own page references are right. Both must hold before a page metric means
anything, and they fail in different ways — so they are reported separately
rather than collapsed into one pass/fail.

The offset relationship is *derived from the evidence*, not assumed: if the
probes consistently show a different base, that is reported rather than
silently corrected.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

from rag_eval.dataset.pageoffsets import distinctive_fragments
from rag_eval.types import GoldenItem, RagTrace

#: Hansard running header: "DN 4.8.2026 129", "DR.14.6.2004 3", "KKDR.14.7.2026 3".
#: Only Arabic numerals are captured — roman front matter ("DN 26.2.2026 iv") is
#: deliberately not an ``ms.`` page and must not be read as one.
HEADER_RE = re.compile(
    r"\b(?:DR|DN|KKDR|KR)[\s.]*\d{1,2}\.\d{1,2}\.\d{4}\s+(\d{1,4})\b",
    re.IGNORECASE,
)

#: Added to ``page_number`` before subtracting the offset. 1 encodes the
#: expectation that nv-ingest numbers pages from zero; the check reports the
#: base it actually observes, so a wrong value here surfaces rather than hides.
PAGE_BASE = 1


def header_page(text: str) -> int | None:
    """The printed page number from a chunk's running header, if present."""
    m = HEADER_RE.search(text or "")
    return int(m.group(1)) if m else None


@dataclass
class PageMapProbe:
    question_id: str
    sitting_id: str
    excel_ms: int
    ms_offset: int | None = None
    page_number: int | None = None
    header: int | None = None
    computed: int | None = None
    matched_on: str = ""          # "answer" | "sitting" | ""
    verdict: str = "no-hit"       # match | formula-mismatch | golden-mismatch | no-header | no-hit
    note: str = ""

    @property
    def observed_base(self) -> int | None:
        """``header - (page_number - ms_offset)`` — the base the data implies."""
        if self.header is None or self.page_number is None or self.ms_offset is None:
            return None
        return self.header - (self.page_number - self.ms_offset)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id, "sitting_id": self.sitting_id,
            "excel_ms": self.excel_ms, "ms_offset": self.ms_offset,
            "page_number": self.page_number, "header": self.header,
            "computed": self.computed, "observed_base": self.observed_base,
            "matched_on": self.matched_on, "verdict": self.verdict, "note": self.note,
        }


def _probe_fn(adapter: Any):
    """Retrieval-only where available: this check never reads a generated answer."""
    return adapter.retrieve if hasattr(adapter, "retrieve") else adapter.answer


def probe_one(adapter: Any, item: GoldenItem, offsets: dict[str, int]) -> PageMapProbe:
    """Retrieve for one golden question and compare the three page values."""
    gold = item.reference
    sitting = gold.sitting_id
    result = PageMapProbe(
        question_id=item.id, sitting_id=sitting,
        excel_ms=gold.pages[0], ms_offset=offsets.get(sitting),
    )

    try:
        trace: RagTrace = _probe_fn(adapter)(item.question, item.id)
    except Exception as exc:  # noqa: BLE001
        result.note = f"{type(exc).__name__}: {exc}"[:150]
        return result
    if not trace.ok:
        result.note = (trace.error or "")[:150]
        return result

    from_sitting = [c for c in trace.retrieved_chunks if c.sitting_id == sitting]
    if not from_sitting:
        result.note = (
            f"no chunk from {sitting} in top {len(trace.retrieved_chunks)} "
            f"(got {sorted({c.sitting_id for c in trace.retrieved_chunks if c.sitting_id})})"
        )
        return result

    # Prefer the chunk that actually contains the golden answer; that pins the
    # comparison to the right page rather than merely the right sitting.
    chosen, matched_on = from_sitting[0], "sitting"
    for fragment in distinctive_fragments(item.expected_answer):
        needle = re.sub(r"\s+", " ", fragment).casefold()
        hit = next((c for c in from_sitting
                    if needle in re.sub(r"\s+", " ", c.text or "").casefold()), None)
        if hit is not None:
            chosen, matched_on = hit, "answer"
            break

    result.matched_on = matched_on
    result.page_number = chosen.page
    result.header = header_page(chosen.text)
    if result.page_number is not None and result.ms_offset is not None:
        result.computed = result.page_number + PAGE_BASE - result.ms_offset

    if result.header is None:
        result.verdict = "no-header"
        result.note = "chunk carries no running header; cannot read the printed page"
    elif result.computed is not None and result.header != result.computed:
        result.verdict = "formula-mismatch"
        result.note = f"header {result.header} vs computed {result.computed}"
    elif matched_on == "answer" and result.header != result.excel_ms:
        result.verdict = "golden-mismatch"
        result.note = f"header {result.header} vs golden ms. {result.excel_ms}"
    else:
        result.verdict = "match"
    return result


def select_probes(items: Sequence[GoldenItem], sample: int) -> list[GoldenItem]:
    """One question per sitting, favouring answers with something quotable in them."""
    by_sitting: dict[str, list[GoldenItem]] = {}
    for item in items:
        if item.reference and item.reference.pages:
            by_sitting.setdefault(item.reference.sitting_id, []).append(item)
    chosen: list[GoldenItem] = []
    for sitting in sorted(by_sitting):
        ranked = sorted(
            by_sitting[sitting],
            key=lambda i: -len(distinctive_fragments(i.expected_answer)),
        )
        if ranked:
            chosen.append(ranked[0])
        if len(chosen) >= sample:
            break
    return chosen


def check(
    adapter: Any, items: Sequence[GoldenItem], offsets: dict[str, int], *, sample: int = 5,
    on_probe=None,
) -> dict[str, Any]:
    """Run the probes and summarise. Never raises on a single bad probe."""
    probes = select_probes(items, sample)
    results: list[PageMapProbe] = []
    for index, item in enumerate(probes, 1):
        r = probe_one(adapter, item, offsets)
        results.append(r)
        if on_probe:
            on_probe(index, len(probes), r)

    bases = [r.observed_base for r in results if r.observed_base is not None]
    counts: dict[str, int] = {}
    for r in results:
        counts[r.verdict] = counts.get(r.verdict, 0) + 1

    comparable = [r for r in results if r.verdict in ("match", "formula-mismatch", "golden-mismatch")]
    ok = counts.get("match", 0)
    verdict = (
        f"PASS: {ok}/{len(comparable)} comparable probe(s) map correctly"
        if comparable and ok == len(comparable)
        else f"FAIL: {len(comparable) - ok} of {len(comparable)} probe(s) did not map"
        if comparable
        else "INCONCLUSIVE: no probe produced a comparable page"
    )
    return {
        "probes": len(results),
        "summary": counts,
        "observed_bases": {str(b): bases.count(b) for b in sorted(set(bases))},
        "assumed_base": PAGE_BASE,
        "base_consistent": bool(bases) and len(set(bases)) == 1 and bases[0] == PAGE_BASE,
        "verdict": verdict,
        "results": [r.to_dict() for r in results],
    }
