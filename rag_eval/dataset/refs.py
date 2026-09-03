"""Parser for the Excel ``Reference`` column.

That field is the source of truth for both retrieval targets and citation
validation (design spec §5.5), and it is hand-typed, so it varies:

    dr_2026-06-22, ms. 3        single page, canonical form
    dn_2026-08-04, ms. 15-16    page range
    kkdr_2026-7-14, ms 1-2, 5   single-digit month, no dot, range + extra page
    kkdr_2026-7-15, ms 19       no dot after ms

Everything is normalised to ``SourceRef(doc_type, ISO date, pages tuple)``.
"""

from __future__ import annotations

import re

from rag_eval.types import DOC_TYPES, SourceRef

#: Known typos in the hand-typed prefix, mapped to the sitting they mean.
#:
#: ``kr_`` is a dropped-keystroke ``kkdr_``, not a sitting type. It occurs twice
#: in the workbook (Excel rows 233 and 249) and both times it sits *inside* a
#: contiguous block of ``kkdr_`` rows with the same owner, the same date and the
#: same page: row 233 between five ``kkdr_2026-7-14, ms 9`` rows, row 249 between
#: four ``kkdr_2026-7-15, ms 5`` rows. Left unaliased it invents a 17th sitting
#: that no chunk can ever match, so those two questions score a permanent zero on
#: hit-rate. Aliasing is applied at parse time and reported by ``convert-golden``;
#: the original string is preserved in ``SourceRef.raw``.
DOC_TYPE_ALIASES: dict[str, str] = {"kr": "kkdr"}

# <prefix>_<date>, <page spec>   -- the page spec is optional so that a
# reference naming only a sitting still yields a usable retrieval target.
_REF_RE = re.compile(
    r"""^\s*
        (?P<doc>[A-Za-z]+)_
        (?P<year>\d{4})-(?P<month>\d{1,2})-(?P<day>\d{1,2})
        (?:\s*,\s*(?:ms\.?|m/s|page|pg\.?)?\s*(?P<pages>[\d\s,–—-]*))?
        \s*$""",
    re.VERBOSE | re.IGNORECASE,
)

# "3", "15-16", "1-2, 5" (en/em dashes included -- Excel autocorrect produces them)
_PAGE_PART_RE = re.compile(r"^(\d+)\s*(?:[-–—]\s*(\d+))?$")

# A range wider than this is almost certainly a typo (e.g. "ms. 3-300"); we keep
# the endpoints rather than silently expanding to hundreds of gold pages.
MAX_RANGE_SPAN = 20


class ReferenceParseError(ValueError):
    """Raised when a reference string cannot be understood at all."""


def parse_pages(spec: str) -> tuple[int, ...]:
    """Expand a page spec into a sorted, de-duplicated tuple of page numbers."""
    pages: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        m = _PAGE_PART_RE.match(part)
        if not m:
            raise ReferenceParseError(f"unparseable page spec: {part!r}")
        start = int(m.group(1))
        end = int(m.group(2)) if m.group(2) else start
        if end < start:
            start, end = end, start
        if end - start > MAX_RANGE_SPAN:
            pages.extend((start, end))
        else:
            pages.extend(range(start, end + 1))
    return tuple(sorted(set(pages)))


def parse_reference(raw: str, *, strict: bool = False) -> SourceRef | None:
    """Parse one Excel reference cell.

    Prefixes in :data:`DOC_TYPE_ALIASES` are corrected to the sitting they mean;
    ``SourceRef.raw`` keeps the string exactly as it was typed.

    Returns ``None`` for an empty cell (a handful of golden rows have no
    reference; those rows are excluded from retrieval metrics rather than
    scored as misses). Raises :class:`ReferenceParseError` for a non-empty cell
    that cannot be parsed.

    ``strict`` additionally rejects document prefixes outside the known set.
    """
    text = (raw or "").strip()
    if not text:
        return None

    m = _REF_RE.match(text)
    if not m:
        raise ReferenceParseError(f"unrecognised reference format: {text!r}")

    doc = DOC_TYPE_ALIASES.get(m.group("doc").lower(), m.group("doc").lower())
    if strict and doc not in DOC_TYPES:
        raise ReferenceParseError(f"unknown document prefix {doc!r} in {text!r}")

    date = f"{int(m.group('year')):04d}-{int(m.group('month')):02d}-{int(m.group('day')):02d}"
    pages = parse_pages(m.group("pages") or "")
    return SourceRef(doc_type=doc, sitting_date=date, pages=pages, raw=text)


def format_sitting_id(doc_type: str, sitting_date: str) -> str:
    """``('dr', '2026-06-22') -> 'dr_2026-06-22'`` with the date normalised."""
    y, mth, d = (int(x) for x in sitting_date.split("-"))
    return f"{doc_type.lower()}_{y:04d}-{mth:02d}-{d:02d}"


def normalise_sitting_id(value: str | None) -> str | None:
    """Normalise a sitting id coming back from a RAG service's chunk metadata.

    Tolerates ``kkdr_2026-7-14``, ``KKDR_2026-07-14`` and a bare filename such
    as ``dr_2026-06-22.pdf`` so that adapters do not each reinvent this.
    """
    if not value:
        return None
    text = str(value).strip()
    for suffix in (".pdf", ".txt", ".json"):
        if text.lower().endswith(suffix):
            text = text[: -len(suffix)]
    m = re.match(r"^([A-Za-z]+)_(\d{4})-(\d{1,2})-(\d{1,2})", text)
    if not m:
        return text.lower() or None
    doc = DOC_TYPE_ALIASES.get(m.group(1).lower(), m.group(1))
    return format_sitting_id(doc, f"{m.group(2)}-{m.group(3)}-{m.group(4)}")
