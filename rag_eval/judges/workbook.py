"""Write the calibration sheet as an .xlsx a person can actually label in.

A CSV is fine for a machine and unpleasant for a human: Hansard answers run to
paragraphs, and unwrapped text in a 8-character column is unreadable. This
lays the same rows out to be read and scored — wrapped text, a frozen header,
1-5 dropdowns on the score columns, and the panel's own scores hidden.

Hiding the panel is the point rather than tidiness: a labeller who sees the
judges' answers before forming their own is anchored by them, and an anchored
label cannot measure the panel it is supposed to check.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from rag_eval.judges.calibration import CSV_COLUMNS, sheet_to_rows
from rag_eval.judges.rubric import DIMENSION_PROMPTS, DIMENSIONS, SCALE_MAX, SCALE_MIN

#: Column widths, in characters. Anything unlisted gets a readable default.
_WIDTHS = {
    "question_id": 13, "owner": 10, "sitting_id": 18, "golden_ms": 10,
    "question": 52, "expected_answer": 60, "generated_answer": 60,
    "retrieved_context": 80, "judge_rationales": 70, "adapter_error": 30,
    "labelled_by": 14, "notes": 34, "flagged_dimensions": 26, "judges_responded": 9,
    "assignee": 12,
}
_TALL = 110  # row height for rows carrying paragraphs


def _assign(rows: Sequence[dict[str, str]], labellers: Sequence[str]) -> None:
    """Round-robin rows across labellers.

    Interleaved rather than split into blocks: the sample is ordered by owner,
    so contiguous blocks would give one labeller mostly one reviewer's
    questions and confound "which labeller" with "whose questions".
    """
    if not labellers:
        return
    for index, row in enumerate(rows):
        row["assignee"] = labellers[index % len(labellers)]


def write_workbook(
    sheet: dict[str, Any],
    path: str | Path,
    *,
    labellers: Sequence[str] = (),
) -> tuple[Path, int]:
    """Write the labelling workbook. Returns ``(path, row count)``."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.datavalidation import DataValidation

    rows = sheet_to_rows(sheet)
    _assign(rows, labellers)
    columns = (["assignee"] if labellers else []) + list(CSV_COLUMNS)

    wb = Workbook()
    ws = wb.active
    ws.title = "labelling"

    header_fill = PatternFill("solid", fgColor="1F3864")
    score_fill = PatternFill("solid", fgColor="FFF2CC")
    header_font = Font(color="FFFFFF", bold=True)
    wrap = Alignment(vertical="top", wrap_text=True)

    ws.append([c.replace("_", " ") for c in columns])
    for cell in ws[1]:
        cell.fill, cell.font = header_fill, header_font
        cell.alignment = Alignment(vertical="center", wrap_text=True)
    ws.freeze_panes = "A2"
    ws.row_dimensions[1].height = 30

    for row in rows:
        ws.append([row.get(c, "") for c in columns])

    human_cols = {f"human_{d}" for d in DIMENSIONS}
    for index, name in enumerate(columns, start=1):
        letter = get_column_letter(index)
        ws.column_dimensions[letter].width = _WIDTHS.get(name, 12)
        # The panel's scores travel with the file but stay out of sight.
        if name.startswith("panel_") or name in ("judge_rationales", "adapter_error"):
            ws.column_dimensions[letter].hidden = True
        if name in human_cols:
            for cell in ws[letter][1:]:
                cell.fill = score_fill

    for r in range(2, len(rows) + 2):
        ws.row_dimensions[r].height = _TALL
        for cell in ws[r]:
            cell.alignment = wrap

    rule = DataValidation(
        type="whole", operator="between", formula1=SCALE_MIN, formula2=SCALE_MAX,
        allow_blank=True, showErrorMessage=True,
        errorTitle="Score out of range",
        error=f"Scores are {SCALE_MIN}-{SCALE_MAX}. Leave blank if you cannot judge it.",
    )
    ws.add_data_validation(rule)
    for name in human_cols:
        letter = get_column_letter(columns.index(name) + 1)
        rule.add(f"{letter}2:{letter}{len(rows) + 1}")

    guide = wb.create_sheet("how to label")
    guide.column_dimensions["A"].width = 22
    guide.column_dimensions["B"].width = 108
    guide["A1"], guide["B1"] = "dimension", "score 1-5"
    for cell in (guide["A1"], guide["B1"]):
        cell.fill, cell.font = header_fill, header_font
    for dim in DIMENSIONS:
        guide.append([dim, DIMENSION_PROMPTS[dim]])
    guide.append([])
    for line in (
        ["how", "Read question, expected answer, generated answer and retrieved context. "
                "Score each dimension 1-5 in the shaded columns."],
        ["order", "Score before looking at what the judges said. Their scores are in hidden "
                  "columns; unhide only after you have formed your own view."],
        ["blanks", "Leave a cell blank if you genuinely cannot judge it. A blank is skipped; "
                   "a guess becomes ground truth."],
        ["priority", "Rows with 'flagged dimensions' filled in are where the three judges "
                     "disagreed. Label those first if short of time."],
        ["save as", "Save/export as CSV, then: rag-eval label <file>.csv --labelled-by <name>"],
    ):
        guide.append(line)
    for row in guide.iter_rows():
        for cell in row:
            cell.alignment = wrap
    guide.freeze_panes = "A2"

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)
    return path, len(rows)
