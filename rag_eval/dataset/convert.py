"""Convert ``TanyaParlimen QnA.xlsx`` into the versioned golden-set JSONL (D2).

The workbook's columns are ``No | Question | Answer | Owner | Reference``.
Conversion is strict about what matters and loud about what does not parse:
every rejected or partially-parsed row appears in the report, because the
reference field drives both retrieval targets and citation validation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rag_eval.dataset.excel import read_table
from rag_eval.dataset.loader import sha256_file, write_golden_set
from rag_eval.dataset.refs import DOC_TYPE_ALIASES, ReferenceParseError, parse_reference
from rag_eval.types import DOC_TYPES, GoldenItem

COLUMN_ALIASES = {
    "no": "no",
    "question": "question",
    "soalan": "question",
    "answer": "answer",
    "jawapan": "answer",
    "owner": "owner",
    "reference": "reference",
    "rujukan": "reference",
}


@dataclass
class ConversionReport:
    total_rows: int = 0
    converted: int = 0
    skipped_empty: int = 0
    missing_reference: list[str] = field(default_factory=list)
    unparsed_reference: list[tuple[str, str, str]] = field(default_factory=list)
    unknown_doc_types: dict[str, int] = field(default_factory=dict)
    duplicate_numbers: list[tuple[str, int]] = field(default_factory=list)
    aliased_doc_types: list[tuple[str, str, str]] = field(default_factory=list)
    sittings: dict[str, int] = field(default_factory=dict)
    owners: dict[str, int] = field(default_factory=dict)
    multi_page_refs: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_rows": self.total_rows,
            "converted": self.converted,
            "skipped_empty": self.skipped_empty,
            "missing_reference": self.missing_reference,
            "unparsed_reference": [
                {"id": i, "raw": r, "error": e} for i, r, e in self.unparsed_reference
            ],
            "unknown_doc_types": self.unknown_doc_types,
            "duplicate_numbers": [
                {"no": n, "excel_row": r} for n, r in self.duplicate_numbers
            ],
            "aliased_doc_types": [
                {"id": i, "raw": raw, "corrected_to": sitting}
                for i, raw, sitting in self.aliased_doc_types
            ],
            "sittings": dict(sorted(self.sittings.items())),
            "sitting_count": len(self.sittings),
            "owners": dict(sorted(self.owners.items())),
            "multi_page_refs": self.multi_page_refs,
        }


def _normalise_columns(record: dict[str, str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in record.items():
        out[COLUMN_ALIASES.get(key.strip().lower(), key.strip().lower())] = value
    return out


def _clean_text(value: str) -> str:
    """Collapse the literal ``\\n`` sequences and stray whitespace in the sheet."""
    text = (value or "").replace("\\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return "\n".join(line.strip() for line in text.splitlines()).strip()


def convert_workbook(
    xlsx_path: str | Path,
    output_path: str | Path,
    *,
    sheet: int = 1,
    strict_refs: bool = False,
) -> tuple[list[GoldenItem], ConversionReport, Any]:
    """Convert the workbook and write ``output_path`` plus its manifest.

    ``strict_refs`` turns an unparseable or unknown-prefix reference into a
    hard failure instead of a reported warning.
    """
    xlsx_path, output_path = Path(xlsx_path), Path(output_path)
    _, rows = read_table(xlsx_path, sheet=sheet)

    items: list[GoldenItem] = []
    report = ConversionReport(total_rows=len(rows))
    used_ids: set[str] = set()

    for row_no, raw_record in rows:
        record = _normalise_columns(raw_record)
        question = _clean_text(record.get("question", ""))
        answer = _clean_text(record.get("answer", ""))
        if not question or not answer:
            report.skipped_empty += 1
            continue

        # The "No" column is what a reviewer quotes, so it drives the id -- but it
        # is hand-maintained and does contain collisions (rows 201 and 202 of the
        # current workbook are both numbered 200). A collision is disambiguated by
        # Excel row and reported, never silently merged: two questions sharing an id
        # would overwrite each other's trace and quietly shrink the eval.
        number = (record.get("no") or "").strip()
        item_id = f"tp-{int(number):04d}" if number.isdigit() else f"tp-row{row_no}"
        if item_id in used_ids:
            report.duplicate_numbers.append((number or str(row_no), row_no))
            item_id = f"{item_id}-r{row_no}"
        used_ids.add(item_id)

        raw_ref = (record.get("reference") or "").strip()
        reference = None
        try:
            reference = parse_reference(raw_ref, strict=strict_refs)
        except ReferenceParseError as exc:
            if strict_refs:
                raise
            report.unparsed_reference.append((item_id, raw_ref, str(exc)))

        if reference is None:
            if raw_ref == "":
                report.missing_reference.append(item_id)
        else:
            report.sittings[reference.sitting_id] = report.sittings.get(reference.sitting_id, 0) + 1
            if len(reference.pages) > 1:
                report.multi_page_refs += 1
            typed_prefix = raw_ref.split("_", 1)[0].lower()
            if typed_prefix in DOC_TYPE_ALIASES:
                report.aliased_doc_types.append((item_id, raw_ref, reference.sitting_id))
            if reference.doc_type not in DOC_TYPES:
                report.unknown_doc_types[reference.doc_type] = (
                    report.unknown_doc_types.get(reference.doc_type, 0) + 1
                )

        owner = (record.get("owner") or "").strip()
        report.owners[owner or "(unassigned)"] = report.owners.get(owner or "(unassigned)", 0) + 1

        tags = []
        if reference is not None:
            tags.append(f"chamber:{reference.doc_type}")
            tags.append(f"sitting:{reference.sitting_id}")
        if not reference or not reference.pages:
            tags.append("no-page-reference")

        items.append(
            GoldenItem(
                id=item_id,
                question=question,
                expected_answer=answer,
                reference=reference,
                owner=owner,
                source_row=row_no,
                tags=tags,
            )
        )

    report.converted = len(items)
    manifest = write_golden_set(
        output_path,
        items,
        source=xlsx_path.name,
        source_sha256=sha256_file(xlsx_path),
        notes=(
            f"Converted from {xlsx_path.name} sheet {sheet}; "
            f"{len(report.sittings)} sittings, "
            f"{len(report.missing_reference)} row(s) without a reference."
        ),
    )
    return items, report, manifest
