"""Minimal .xlsx reader (stdlib only).

An eval framework whose golden set can only be rebuilt when pandas/openpyxl
happen to be installed is a framework that stops being rebuildable. An .xlsx is
a zip of XML; reading the one sheet we need is ~60 lines, so we do that instead
of taking two heavyweight dependencies for a one-shot conversion.
"""

from __future__ import annotations

import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterator

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_COL_RE = re.compile(r"([A-Z]+)(\d+)")


def _shared_strings(zf: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    return ["".join(t.text or "" for t in si.iter(NS + "t")) for si in root.findall(NS + "si")]


def _cell_value(cell: ET.Element, shared: list[str]) -> str:
    kind = cell.get("t")
    if kind == "inlineStr":
        return "".join(t.text or "" for t in cell.iter(NS + "t"))
    v = cell.find(NS + "v")
    if v is None or v.text is None:
        return ""
    if kind == "s":
        idx = int(v.text)
        return shared[idx] if 0 <= idx < len(shared) else ""
    return v.text


def iter_rows(path: str | Path, sheet: int = 1) -> Iterator[tuple[int, dict[str, str]]]:
    """Yield ``(excel_row_number, {column_letter: value})`` for one sheet."""
    with zipfile.ZipFile(path) as zf:
        shared = _shared_strings(zf)
        name = f"xl/worksheets/sheet{sheet}.xml"
        if name not in zf.namelist():
            raise ValueError(f"{Path(path).name} has no sheet {sheet}")
        root = ET.fromstring(zf.read(name))
        data = root.find(NS + "sheetData")
        if data is None:
            return
        for row in data.findall(NS + "row"):
            values: dict[str, str] = {}
            for cell in row.findall(NS + "c"):
                m = _COL_RE.match(cell.get("r") or "")
                if m:
                    values[m.group(1)] = _cell_value(cell, shared)
            yield int(row.get("r") or 0), values


def read_table(path: str | Path, sheet: int = 1) -> tuple[dict[str, str], list[tuple[int, dict[str, str]]]]:
    """Read a header row plus data rows.

    Returns ``(header_by_column_letter, [(row_number, {header: value})])``.
    Rows where every cell is blank are dropped -- the source workbook has
    trailing rows that hold only formatting.
    """
    rows = list(iter_rows(path, sheet))
    if not rows:
        return {}, []
    _, header_cells = rows[0]
    header = {col: (val or "").strip() for col, val in header_cells.items() if (val or "").strip()}

    out: list[tuple[int, dict[str, str]]] = []
    for row_no, cells in rows[1:]:
        record = {name: (cells.get(col) or "").strip() for col, name in header.items()}
        if any(record.values()):
            out.append((row_no, record))
    return header, out
