"""Measure the offset between printed Hansard ``ms.`` pages and physical PDF pages.

A Hansard ``ms.`` number is the page number *printed on the page*; the physical
index in the PDF differs by however much front matter that sitting has. The
offset is per-document (measured: 2 to 12), so a single global constant is wrong
for nearly every file.

The measurement is content-based rather than header-based: for each golden
answer, find the physical page whose text contains a distinctive fragment of it,
and the difference from the recorded ``ms.`` is one vote for that document's
offset. Locating the answer where the golden set says it is tests exactly the
mapping the eval depends on, rather than trusting a page header to be parseable.

Text extraction here is a stdlib PDF reader, not a full library: it inflates
content streams and pulls text-showing operators. Pages it cannot read simply do
not vote, which is why every result carries its sample size and agreement.
"""

from __future__ import annotations

import re
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from rag_eval.types import GoldenItem

_STR = re.compile(rb"\((?:\\.|[^\\()])*\)", re.S)
_SHOW = re.compile(rb"\[(?:\\.|[^\\\]])*\]\s*TJ|\((?:\\.|[^\\()])*\)\s*Tj", re.S)
_OBJ = re.compile(rb"(\d+)\s+0\s+obj")
_PAGE = re.compile(rb"\d+\s+0\s+obj\s*<<[^>]*?/Type\s*/Page[^s]", re.S)
_STREAM = re.compile(rb"stream\r?\n")

#: A document needs this many unambiguous matches, and this share of them
#: agreeing, before its offset is trustworthy enough to stamp onto an ingest.
MIN_SAMPLES = 5
MIN_AGREEMENT = 0.6


def _decode(token: bytes) -> str:
    s = b"".join(x[1:-1] for x in _STR.findall(token))
    s = re.sub(rb"\\([()\\])", rb"\1", s)
    s = re.sub(rb"\\(\d{1,3})", lambda m: bytes([int(m.group(1), 8) & 0xFF]), s)
    return s.decode("latin-1")


def page_texts(path: str | Path) -> list[str]:
    """Text of each physical page, in document order. Unreadable pages are ``""``."""
    data = Path(path).read_bytes()
    objects = {int(m.group(1)): m.start() for m in _OBJ.finditer(data)}
    pages = sorted(
        (off for num, off in objects.items() if _PAGE.match(data[off:off + 600])),
    )

    def stream(num: int) -> bytes:
        off = objects.get(num)
        if off is None:
            return b""
        m = _STREAM.search(data, off)
        if not m:
            return b""
        raw = data[m.end():data.find(b"endstream", m.end())]
        try:
            return zlib.decompress(raw)
        except zlib.error:
            return b""

    out: list[str] = []
    for off in pages:
        # Bound the dictionary to this object. A fixed-size window spills into
        # the next objects on compactly written PDFs and collects their
        # /Contents refs too, concatenating unrelated pages into one.
        end = data.find(b"endobj", off)
        header = data[off:end if end != -1 else off + 1200]
        refs = [int(x) for x in re.findall(rb"/Contents\s+(\d+)\s+0\s+R", header)]
        if not refs:
            arr = re.search(rb"/Contents\s*\[(.*?)\]", header, re.S)
            refs = [int(x) for x in re.findall(rb"(\d+)\s+0\s+R", arr.group(1))] if arr else []
        content = b"".join(stream(r) for r in refs)
        out.append(re.sub(r"\s+", " ", "".join(_decode(m.group(0)) for m in _SHOW.finditer(content))))
    return out


def distinctive_fragments(answer: str) -> list[str]:
    """Bits of a golden answer likely to appear verbatim in the Hansard."""
    phrases = [p.strip() for p in re.split(r"[,.;:()\[\]]", answer) if len(p.strip().split()) >= 5]
    tokens, seen = [], set()
    for tok in re.findall(r"\d[\d,\.]*\d|\b[A-Z][a-z]{4,}\b", answer):
        if tok not in seen:
            seen.add(tok)
            tokens.append(tok)
    return phrases[:3] + tokens[:6]


@dataclass
class OffsetResult:
    sitting_id: str
    filename: str
    offset: int | None = None
    samples: int = 0
    agreement: float | None = None
    votes: dict[int, int] = field(default_factory=dict)
    pages_unreadable: int = 0
    pages_total: int = 0
    note: str = ""

    @property
    def confident(self) -> bool:
        return (
            self.offset is not None
            and self.samples >= MIN_SAMPLES
            and (self.agreement or 0) >= MIN_AGREEMENT
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sitting_id": self.sitting_id,
            "filename": self.filename,
            "offset": self.offset,
            "confident": self.confident,
            "samples": self.samples,
            "agreement": self.agreement,
            "votes": {str(k): v for k, v in sorted(self.votes.items(), key=lambda kv: -kv[1])},
            "pages_total": self.pages_total,
            "pages_unreadable": self.pages_unreadable,
            "note": self.note,
        }


def measure(path: str | Path, sitting_id: str, items: Sequence[GoldenItem]) -> OffsetResult:
    """Measure one document's offset by locating its golden answers."""
    path = Path(path)
    pages = page_texts(path)
    result = OffsetResult(
        sitting_id=sitting_id,
        filename=path.name,
        pages_total=len(pages),
        pages_unreadable=sum(1 for t in pages if len(t.strip()) < 40),
    )
    folded = [t.casefold() for t in pages]

    relevant = [
        i for i in items
        if i.reference and i.reference.sitting_id == sitting_id and i.reference.pages
    ]
    if not relevant:
        result.note = "no golden question cites this sitting"
        return result

    for item in relevant:
        ms = item.reference.pages[0]
        for fragment in distinctive_fragments(item.expected_answer):
            needle = re.sub(r"\s+", " ", fragment).casefold()
            hits = [n for n, text in enumerate(folded, 1) if needle in text]
            if len(hits) == 1:                      # unambiguous location only
                result.votes[hits[0] - ms] = result.votes.get(hits[0] - ms, 0) + 1
                result.samples += 1
                break

    if result.votes:
        offset, count = max(result.votes.items(), key=lambda kv: (kv[1], -abs(kv[0])))
        result.offset = offset
        result.agreement = round(count / result.samples, 4)
    if not result.confident:
        result.note = result.note or (
            f"below the confidence bar ({result.samples} sample(s), "
            f"{result.agreement if result.agreement is not None else 0:.0%} agreement); "
            f"{result.pages_unreadable}/{result.pages_total} pages could not be read"
        )
    return result


def measure_corpus(
    corpus_dir: str | Path, manifest: Any, items: Sequence[GoldenItem]
) -> list[OffsetResult]:
    corpus_dir = Path(corpus_dir)
    out = []
    for doc in sorted(manifest.documents, key=lambda d: d.sitting_id):
        path = corpus_dir / doc.filename
        if path.exists():
            out.append(measure(path, doc.sitting_id, items))
    return out


def offsets_payload(
    results: Sequence[OffsetResult], verified: dict[str, int] | None = None
) -> dict[str, Any]:
    """The file ``rag-eval ingest --offsets`` reads.

    Two sources feed ``offsets``. Measurements that clear the confidence bar,
    and ``verified`` -- offsets a human read off the PDF directly. A human
    reading "ms. 1 is on PDF page 6" is better evidence than any amount of
    text-matching, so verified values win, and they are kept in their own block
    so re-running the measurement cannot silently discard them.

    Anything with neither is listed under ``undetermined`` rather than omitted,
    so a missing document stays visible. ``rag-eval ingest`` leaves ``ms_offset``
    off those documents.
    """
    verified = {k: int(v) for k, v in (verified or {}).items()}
    offsets = {r.sitting_id: r.offset for r in results if r.confident}
    offsets.update(verified)

    evidence = []
    for r in results:
        row = r.to_dict()
        if r.sitting_id in verified:
            row["source"] = "human-verified"
            row["verified_offset"] = verified[r.sitting_id]
            row["agrees_with_measurement"] = (
                r.offset == verified[r.sitting_id] if r.offset is not None else None
            )
        else:
            row["source"] = "measured" if r.confident else "undetermined"
        evidence.append(row)

    return {
        "_comment": (
            "printed ms. = page_number - offset, per document. Generated by "
            "'rag-eval page-offsets'; re-run it if the corpus changes. Entries in "
            "'verified' were confirmed by a human against the PDF and always win."
        ),
        "offsets": dict(sorted(offsets.items())),
        "verified": dict(sorted(verified.items())),
        "undetermined": [
            r.sitting_id for r in results
            if not r.confident and r.sitting_id not in verified
        ],
        "evidence": evidence,
    }


def read_verified(path: str | Path) -> dict[str, int]:
    """Carry forward the human-verified block of an existing offsets file."""
    p = Path(path)
    if not p.exists():
        return {}
    try:
        import json
        return {k: int(v) for k, v in (json.loads(p.read_text()).get("verified") or {}).items()}
    except (ValueError, TypeError):
        return {}
