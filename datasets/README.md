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

## `kr_` — resolved: a typo of `kkdr_`, now fixed at source

`kr_` was **not a sitting type**. It was a dropped-keystroke `kkdr_`.

**The workbook has since been corrected** — Excel rows 233 and 249 now read
`kkdr_`, and no `kr_` remains in the sheet. `DOC_TYPE_ALIASES` in
`rag_eval/dataset/refs.py` still maps `kr_` → `kkdr_`, but it is now a guard
against the typo reappearing rather than something the current data depends on;
`convert-golden` reports zero aliased references. The corpus directory is
independent confirmation: there are 14 Hansard PDFs and no `kr_*.pdf`.

The original evidence is kept below because it is why the corpus is 14 sittings
rather than the 16 the design spec assumes.

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
| ~~`kr_` typo~~ | ~~Excel rows 233, 249~~ | **fixed at source**; alias kept as a guard |
| `No` 200 used twice | Excel rows 201, 202 | second row gets id `tp-0200-r202`; two questions sharing an id would overwrite each other's trace |
| Empty reference | `tp-0100` | excluded from retrieval metrics, **not** scored as a miss |
| Trailing blank rows | Excel rows 373, 374 | dropped (371 of 373 rows convert) |

## Hansard PDF corpus — `hansard_pdfs/`

The 14 sitting PDFs the golden set cites, stored as
`datasets/hansard_pdfs/<sitting_id>.pdf`:

```
datasets/hansard_pdfs/
    manifest.json
    dr_2026-06-22.pdf      dn_2026-08-04.pdf      kkdr_2026-07-14.pdf
    …
```

**The PDFs are gitignored; `manifest.json` is not.** ~15 MB of source documents
do not belong in the repo, but their checksums do: the manifest is what lets
anyone prove a local copy is the same corpus a past scorecard was produced
against. A fresh clone therefore has the manifest and no documents, and
`corpus-check` says so in one line rather than reporting fourteen missing files:

```
FAIL: no Hansard PDFs in datasets/hansard_pdfs — corpus not fetched
```

Fetch the 14 documents into `datasets/hansard_pdfs/` (from the parliament corpus
share — ask Nicholas or the interns), named `<sitting_id>.pdf` exactly as
`manifest.json` lists them, then re-run `corpus-check` to confirm you have the
same bytes.

`manifest.json` is keyed by sitting id — `sitting_id → filename → sha256 →
page_count` — plus byte size, PDF version, and how many golden questions cite
the sitting and which pages they cite:

```json
{
  "directory": "datasets/hansard_pdfs",
  "document_count": 14,
  "golden_set": "golden_v1.jsonl",
  "golden_set_sha256": "138f682c…",
  "documents": {
    "dr_2026-06-22": {
      "filename": "dr_2026-06-22.pdf",
      "sha256": "c19dcc7d7d8c0394…",
      "page_count": 155,
      "chamber": "Dewan Rakyat",
      "bytes": 1245440,
      "golden_questions": 40,
      "referenced_pages": { "count": 27, "min": 3, "max": 70 }
    }
  }
}
```

One PDF per sitting is enforced, not assumed: two files resolving to the same
sitting id (`dr_2026-06-22.pdf` and `dr_2026-6-22.pdf`) is a hard error, because
the keyed manifest would otherwise let one of them overwrite the other and go
unchecksummed. A file whose name is not `<sitting_id>.pdf` verifies fine but
warns with the name to rename it to.

Page counts are read from each PDF's own page tree, cross-checking the
`/Type /Page` object count against the tree's `/Count`; when they disagree the
count is recorded as `null` rather than guessed, since a wrong number in a
manifest that later checks claim to have verified is worse than no number.

```bash
rag-eval corpus-check            # verify the PDFs against the manifest
rag-eval corpus-check --write    # (re)generate the manifest from disk
```

Why checksum the PDFs as well as the golden set: a silently re-downloaded,
truncated or re-paginated Hansard changes what the retriever can possibly find,
so every score compared across that change is meaningless. `corpus-check` turns
that into a loud failure instead of a slow mystery.

**Failures** (exit 1) — a score would be wrong or impossible:

| Check | Catches |
|---|---|
| sha256 / byte size | a PDF that changed since it was recorded |
| file missing | a manifested PDF no longer on disk, *and* the questions it orphans |
| file unmanifested | a PDF added without review |
| page count changed | a re-paginated or re-exported document |
| golden sitting with no PDF | questions that can never be answered from the corpus |
| `ms.` beyond the last page | a truncated download, or a reference to the wrong sitting |

**Warnings** — worth a look, but a run is still valid: a PDF nothing cites, an
unreadable page count, an unrecognised filename prefix, or a manifest whose
recorded `golden_set_sha256` no longer matches the golden set on disk.

### `ms.` is the printed page, not the PDF page — verified offsets

**This is the single most important fact about the corpus.** A Hansard `ms.`
number is the page number *printed on the page*. It is **not** the physical page
index in the PDF: every sitting has covers and front matter, so the two differ
by a per-document offset.

Spot-checked against three PDFs by locating golden answers' verbatim text and
comparing the physical page it lands on with the `ms.` the golden set records:

| Document | Offset | Content matches agreeing | Cross-check |
|---|---|---|---|
| `dr_2026-06-22` | **+6** | 22/29 (76%) | — |
| `dn_2026-08-04` | **+5** | 18/26 (69%) | running header also says 5 |
| `kkdr_2026-07-14` | **+2** | 12/16 (75%) | running header also says 2 |

So `physical_page = ms + offset`, and **the offset differs per document** — a
single global constant would be wrong for nearly every file.

Worked examples, each re-checkable by opening the PDF at the physical page:

```
tp-0003   golden: dr_2026-06-22, ms. 9   ->  PHYSICAL page 15   (offset 6)
  "…kita telah menambah 2,202 orang. Jadi, jumlah keseluruhan setakat ini
   adalah 3,404 orang…"
  golden answer: "Jumlah keseluruhan setakat ini adalah 3,404 orang."

tp-0001   golden: dr_2026-06-22, ms. 3   ->  PHYSICAL page 9    (offset 6)
  "…Yang Berhormat bagi kawasan Pandan dan Setiawangsa telah menghantar surat
   untuk melepaskan keahlian mereka sebagai Ahli Dewan Rakyat…"

tp-0207   golden: kkdr_2026-07-14, ms. 3 ->  PHYSICAL page 5    (offset 2)
  "KKDR.14.7.2026 3 Antara kesalahan yang direkodkan sepanjang tempoh
   berkempen…"
   ^ the running header carries the printed number 3 on physical page 5.
```

The remaining disagreements in the table are almost all ±1 — an answer that
spans a page break puts its quoted phrase on the following page — plus a few
short tokens that coincidentally match elsewhere in the document.

**Method caveat.** These offsets come from a stdlib PDF text reader, not a full
PDF library; 17–29% of pages in these documents yield no extractable text, and
those pages simply do not vote. Treat the offsets as well-evidenced, not
certified. They are *not* recorded in `manifest.json` for that reason — a
number a later check claims to have verified must be one we can actually
re-verify.

### What this means for ingestion

Chunk metadata should carry the **printed** `ms.` number as `page`, parsed from
the page's running header, because that is what the golden set records, what a
citation must say to be verifiable against the official Hansard, and what stays
stable when the corpus is re-exported. Keep the physical index alongside it
(`pdf_page`) for rendering and debugging, and flag whether the printed number
was read or inferred.

If ingestion instead stamps the physical index as `page`, the failure is silent
and specific: `hit_rate@k` and `mrr` stay healthy (they match on sitting only)
while `page_citation_accuracy` and `recall@k` collapse toward zero. That exact
signature — good hit-rate, near-zero citation accuracy — means a page-mapping
bug, not a bad retriever.

### What the page-count check does *not* prove

Page counts only **bound** the `ms.` references; the bounds check passing says
nothing about alignment. Confirming the ingestor actually preserved sitting id
and the right page number is `rag-eval ingest-check`, run against the live
collection.

Non-PDF files in the directory are ignored, including the `:Zone.Identifier`
streams Windows leaves beside downloaded files. Those are safe to delete.

## Judge calibration set

`judge_calibration.jsonl` (created on first use) holds the human labels the
judge panel is measured against — the ~50-sample seed set plus every later
adjudication of a panel split. Append-only: a human label is evidence, and
replacing one would make the panel's agreement history unreproducible.

```bash
rag-eval label seed_labels.json     # seed it
rag-eval calibrate --run <run_id>   # panel-vs-human agreement per dimension and per judge
```
