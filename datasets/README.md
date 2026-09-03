# Golden set

`golden_v1.jsonl` — 371 Q&A pairs converted from `TanyaParlimen_QnA.xlsx`, with
`golden_v1.manifest.json` recording the sha256, row count and provenance of the
conversion.

Rebuild it (deterministic — the same workbook always yields the same sha256):

```bash
rag-eval convert-golden
rag-eval dataset-check
```

Never hand-edit `golden_v1.jsonl`. The loader verifies it against the manifest
and refuses a file that no longer matches, because a quiet edit to the golden
set silently moves every score that was ever compared against it. Change the
workbook, re-run `convert-golden`, and review the resulting sha256.

## Schema

One JSON object per line:

| Field | Notes |
|---|---|
| `id` | `tp-<No>` from the workbook's `No` column, zero-padded |
| `question` | Malay or English |
| `expected_answer` | The human-written golden answer |
| `reference` | `{doc_type, sitting_date, pages, raw}` — or `null` |
| `owner` | Reviewer who wrote the pair |
| `source_row` | Excel row, for tracing back to the workbook |
| `tags` | `chamber:…`, `sitting:…`, `no-page-reference` |

`reference.raw` always preserves the string exactly as it was typed, so any
normalisation below stays auditable.

## Sittings — 14, not 16

| Prefix | Chamber | Sittings |
|---|---|---|
| `dr_` | Dewan Rakyat | 2004-06-14, 2017-10-31, 2026-03-02, 2026-06-22 |
| `dn_` | Dewan Negara | 2019-05-06, 2020-09-22, 2026-02-26, 2026-03-02, 2026-07-20, 2026-08-03, 2026-08-04 |
| `kkdr_` | Kamar Khas Dewan Rakyat | 2026-07-14, 2026-07-15, 2026-07-16 |

**The design spec's "16 sittings" counts two references that are typos.** See
`kr_` below. The corpus to ingest for D1 is **14 Hansard sittings**, not 16 —
two fewer PDFs to collect.

## `kr_` — resolved: a typo of `kkdr_`

`kr_` is **not a sitting type**. It is a dropped-keystroke `kkdr_`, and the
parser corrects it (`DOC_TYPE_ALIASES` in `rag_eval/dataset/refs.py`).

Evidence — it occurs exactly twice, and both times *inside* a contiguous block
of `kkdr_` rows with the same owner, the same date and the same page:

```
Excel row 230  Syahir  kkdr_2026-7-14, ms 9
Excel row 231  Syahir  kkdr_2026-7-14, ms 9
Excel row 232  Syahir  kkdr_2026-7-14, ms 9
Excel row 233  Syahir  kr_2026-7-14,   ms 9   <-- tp-0231
Excel row 234  Syahir  kkdr_2026-7-14, ms 9
Excel row 235  Syahir  kkdr_2026-7-14, ms 9

Excel row 246  Syahir  kkdr_2026-7-15, ms 4
Excel row 247  Syahir  kkdr_2026-7-15, ms 5
Excel row 248  Syahir  kkdr_2026-7-15, ms 5
Excel row 249  Syahir  kr_2026-7-15,   ms 5   <-- tp-0247
Excel row 250  Syahir  kkdr_2026-7-15, ms 5
```

Both dates already have a `kkdr_` sitting. There is no third chamber sitting on
either day, and no `kr_` sitting exists anywhere else in the workbook.

Why this is corrected rather than merely recorded: left alone, `kr_2026-07-14`
is a sitting id no ingested chunk can ever carry, so `tp-0231` and `tp-0247`
would score a permanent zero on `hit_rate`, `recall` and
`page_citation_accuracy` — a fixed −0.5% on every scorecard, caused by a typo.
The correction is applied at parse time, reported by `convert-golden`, and
mirrored in `normalise_sitting_id` so a corrected golden reference still matches
a chunk whose metadata carries the uncorrected prefix.

**Fix the workbook** and this alias becomes dead code. Until then it is load-bearing.

## Other normalisations

All applied at parse time, all reported by `convert-golden`:

| In the workbook | Normalised to | Why |
|---|---|---|
| `kkdr_2026-7-14` | `kkdr_2026-07-14` | single-digit months would split one sitting into two retrieval targets |
| `ms 19` (no dot) | pages `(19,)` | hand-typed variation |
| `ms. 15-16` | pages `(15, 16)` | a range means *any of these pages* for citation matching |
| `ms 1-2, 5` | pages `(1, 2, 5)` | 33 references list multiple pages |
| `KKDR_2026-7-14.pdf` | `kkdr_2026-07-14` | applied to chunk metadata coming back from a RAG service |

## Known defects in the source workbook

Fix these in `TanyaParlimen_QnA.xlsx` and the framework's handling becomes
unnecessary — none of it is load-bearing once the sheet is clean.

| Defect | Where | Handled by |
|---|---|---|
| `kr_` typo | Excel rows 233, 249 | corrected to `kkdr_`, reported |
| `No` 200 used twice | Excel rows 201, 202 | second row gets id `tp-0200-r202`; two questions sharing an id would overwrite each other's trace |
| Empty reference | `tp-0100` | excluded from retrieval metrics, **not** scored as a miss |
| Trailing blank rows | Excel rows 373, 374 | dropped (371 of 373 rows convert) |

## Judge calibration set

`judge_calibration.jsonl` (created on first use) holds the human labels the
judge panel is measured against — the ~50-sample seed set plus every later
adjudication of a panel split. Append-only: a human label is evidence, and
replacing one would make the panel's agreement history unreproducible.

```bash
rag-eval label seed_labels.json     # seed it
rag-eval calibrate --run <run_id>   # panel-vs-human agreement per dimension and per judge
```
