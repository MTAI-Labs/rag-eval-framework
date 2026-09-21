# rag-eval-framework

A reusable evaluation harness for any RAG service we build, so that when a RAG
stack changes we can say **with numbers** whether retrieval and answer quality
went up or down.

The framework is product-agnostic: each RAG service plugs in through a thin
adapter. Two ship today — the hosted NVIDIA RAG stack on the RTX6000 cluster and
TanyaParlimen's production RAG.

---

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"           # add ".[judge]" for the real judge panel
cp .env.example .env              # THOTH_BASE_URL + the two NVIDIA endpoints

rag-eval convert-golden           # TanyaParlimen QnA.xlsx -> datasets/golden_v1.jsonl
rag-eval dataset-check            # golden set: schema + checksum
rag-eval corpus-check             # Hansard PDFs: sha256, page counts, coverage
rag-eval ingest-check --adapter nvidia   # is the collection scoreable at all?
rag-eval page-map-check --adapter nvidia # does a chunk's page mean the golden ms.?
rag-eval eval --adapter nvidia    # run + judge + score + report
```

The last command prints a run id and writes `runs/<run_id>/scorecard.html`.

### Try it with no GPU, no gateway, no network

```bash
rag-eval eval --adapter mock \
  --option fixture=tests/data/mock_traces.jsonl \
  --offline-judge tests/data/judge_reply.json --limit 30
```

That replays 30 canned traces through the real pipeline — metrics, judge panel,
scorecard, regression diff — and opens in `runs/<run_id>/scorecard.html`. Run it
twice to see the diff. Without `--option fixture=…` the mock returns chunks with
no metadata, which is worth seeing once: it is exactly the ingestion failure the
scorecard warns about.

## The intended workflow

Run the suite **before** and **after** any change to a RAG stack. The diff is
the go/no-go evidence:

```bash
rag-eval eval --adapter nvidia --note "baseline"
#  … change chunking / retriever / prompt / model …
rag-eval eval --adapter nvidia --note "after chunk-size 512"
```

The second report diffs itself against the previous run **on the same adapter**
and lists both the moved metrics and the individual questions that moved, so a
regression is traceable to rows rather than to a number.

## Pipeline

The stages are separate commands over one run directory, because judging is the
expensive stage: a metric bug should be fixable with `score` alone, and a
judge-prompt change re-runnable without paying for retrieval again.

```
golden_v1.jsonl ──► run ──► traces.jsonl ──┬──► judge ──► judgments.jsonl ──┐
                                        │                               │
                                        └───────────────────────────────┴──► score
                                                                             │
                                            scorecard.json ◄─────────────────┘
                                                   │
                                                   └──► report ──► scorecard.html
                                                                   + regression diff
```

| Command | Does |
|---|---|
| `convert-golden` | Workbook → versioned `golden_v1.jsonl` + checksummed manifest |
| `dataset-check` | Validate a golden set's schema and checksum |
| `corpus-check` | Verify `datasets/hansard_pdfs/` against its sha256 manifest |
| `page-offsets` | Measure printed-`ms.` vs physical-page offset per document |
| `ingest` | Create the eval collection (if absent) and upload the PDFs |
| `ingest-check` | Does the collection exist, and do its chunks carry sitting id + page? |
| `page-map-check` | Does a retrieved chunk's page actually *mean* the golden set's `ms.`? |
| `run` | Golden set → `traces.jsonl` (resumable, error-tolerant) |
| `judge` | Traces → `judgments.jsonl` via the 3-model panel |
| `score` | Traces + judgments → `scorecard.json` |
| `report` | Scorecard → `scorecard.html` with the regression diff |
| `eval` | All four, in one shot |
| `adjudicate` | Export panel splits for a human; import their decisions |
| `calibrate` | How well does the panel agree with the human labels? |
| `diff` | Compare two scored runs |

Every run directory is append-only and self-describing:

```
runs/<run_id>/
    manifest.json     config, adapter, judge model ids, golden-set sha256, git rev
    traces.jsonl      one trace per question
    judgments.jsonl   one panel verdict per question
    scorecard.json    computed metrics
    diff.json         regression diff vs the previous run
    scorecard.html    the report a human reads
```

## Metrics

Every metric is a **pure function over traces** — no I/O, no network, no files —
so each is unit-tested against synthetic traces whose correct score is known by
construction.

Two rules apply throughout:

- **Questions with no golden reference are excluded from the denominator**, not
  scored as misses. One golden row (`tp-0100`) has no `Reference`. Every metric
  reports its own `scorable` count, and returns `None` rather than `0.0` when
  there is nothing to score — an absent metric and a metric that is genuinely
  zero are different facts.
- **An empty retrieval is a miss, not missing data.** If the RAG returns no
  chunks, or the adapter errored, the question still failed to find its
  evidence and scores zero.

### Retrieval — deterministic, no judge

Let `G` = the question's gold `SourceRef` (sitting + pages), `C₁..C_k` the top-k
retrieved chunks in rank order, and `N` the number of scorable questions.

| Metric | Formula |
|---|---|
| `hit_rate@k` | `|{q : ∃ i≤k, sitting(Cᵢ) = sitting(G)}| / N` |
| `recall@k` | `|{q : ∃ i≤k, sitting(Cᵢ) = sitting(G) ∧ page(Cᵢ) ∈ pages(G)}| / N` |
| `mrr` | `(1/N) · Σ 1/rank(first chunk whose sitting = sitting(G))`, 0 if none |
| `page_citation_accuracy` | `|{q : cited page ∈ pages(G) for the gold sitting}| / N` |
| `sitting_citation_accuracy` | `|{q : any cited source is the gold sitting}| / N` |
| `no_citation_rate` | `|{q : the answer cited nothing}| / N` |

**Worked example — `tp-0003`**, gold `dr_2026-06-22, ms. 9`:

```
rank 1  dn_2026-02-26  page 4    ← wrong sitting
rank 2  dr_2026-06-22  page 15   ← right sitting, wrong page
rank 3  dr_2026-06-22  page 9    ← right sitting, gold page
cited:  dr_2026-06-22, ms. 9
```

```
hit_rate@1  = 0     first chunk is the wrong sitting
hit_rate@3  = 1     a gold-sitting chunk appears within the top 3
recall@1    = 0     no gold page in the top 1
recall@3    = 1     rank 3 carries page 9
mrr         = 1/2   first gold-sitting chunk is at rank 2
page_citation_accuracy = 1   the cited ms. 9 is a gold page
```

**Page matching is an intersection test, not set equality.** The Excel reference
is often a range (`ms. 15-16`); an answer citing page 15 of it is correct, and
demanding the full range would score correct behaviour as a miss.

**`page(Cᵢ)` is the printed `ms.`, not the PDF page index.** The ingestor reports
a physical page; the golden set cites the number printed on the paper, and the two
differ by the front matter. The adapter converts at the point it builds the chunk,
using the `ms_offset` stamped onto the document at ingest:

```
page      = pdf_page + 1 - ms_offset     # the printed ms., what metrics compare
pdf_page  = what the ingestor reported   # kept for locating the chunk in the PDF
```

`dr_2004-06-14` prints its `ms. 1` on PDF page 13, so `ms_offset = 13` and physical
page 16 is printed page 4. Getting this wrong is not a rounding error: measured on
the live collection, comparing raw physical pages against the golden set scored
**10% recall on a corpus that retrieves at 94%**.

A document ingested without an `ms_offset` falls back to the raw index and records
`page_source = "physical"`. Those chunks cannot match a golden `ms.` except by
coincidence, so `metadata_health.chunks_without_ms_offset` counts them and the
scorecard prints a warning rather than letting the run report a quiet zero.

#### Citation outcomes

`page_citation_accuracy` alone cannot tell you *why* it is low, so every
question is also classified:

| status | meaning |
|---|---|
| `page_match` | right sitting **and** a gold page |
| `sitting_only` | right sitting, wrong page — retrieval or chunking |
| `wrong` | cited something, none of it the gold sitting |
| `none` | **cited nothing at all** |

`none` **counts as a miss** in `page_citation_accuracy` — failing to cite is a
failure to cite correctly, and excluding it would flatter a RAG that simply
stopped citing. But it is *also* reported as `no_citation_rate` and in
`citation_breakdown`, because it calls for a different fix: no retrieval tuning
repairs a RAG that is not emitting citations. The scorecard raises a warning
whenever it is non-zero.

```
page_citation_accuracy  0.781      289 / 370
no_citation_rate        0.032       12 / 370
citation_breakdown      {page_match: 289, sitting_only: 51, wrong: 18, none: 12}
```

The breakdown always sums to `scorable`.

#### When metadata is missing

A chunk whose metadata lost `sitting_id`/`page` can never match. That is an
**ingestion** defect, not a retrieval failure, so `metadata_health` reports it
separately and the scorecard warns below 95% coverage. Check `ingest-check`
before believing a bad hit-rate, and `page-map-check` before believing a bad
`page_citation_accuracy`.

### Generation — from the judge panel

Each dimension is scored 1–5 by every panel member and aggregated by majority
vote (see below). Let `J` = questions with a final score for that dimension.

| Metric | Formula |
|---|---|
| `faithfulness`, `correctness`, `completeness`, `citation_accuracy` | `mean(final score)` over `J` |
| `<dimension>_pass_rate` | `|{q : score ≥ pass_threshold}| / |J|`, default threshold 4 |
| `hallucination_rate` | `|{q : faithfulness < hallucination_threshold}| / |J|`, default threshold 3 |
| `empty_answer_rate` | `|{q : no answer text}| / |all questions|` |

**Worked example** — four judged questions with faithfulness `5, 4, 2, 1`:

```
faithfulness            = (5+4+2+1)/4 = 3.0
faithfulness_pass_rate  = |{5,4}| / 4 = 0.5     (≥ 4)
hallucination_rate      = |{2,1}| / 4 = 0.5     (< 3)
```

Human adjudication overrides the panel wherever it exists, and the aggregate
always reports `awaiting_human_review` so a mean cannot quietly hide 40 split
rows. Unjudged questions — an adapter error, or every judge failing — are
counted in `unjudged` and excluded from the means rather than scored as zero.

### Ops

| Metric | Formula |
|---|---|
| `latency_ms.p50 / p95 / p99` | linear-interpolated percentile over **successful** traces |
| `error_rate` | `|{q : trace.error}| / |all questions|` |
| `tokens.mean_per_query` | `(Σ prompt + Σ completion) / |successful|` |
| `cost_usd.rag_total` | `(Σ prompt/1000)·input_rate + (Σ completion/1000)·output_rate` |
| `cost_usd.rag_per_query` | `rag_total / |successful|` |

**Worked example** — two successful queries at 100 ms and 300 ms, each 100
prompt + 20 completion tokens, rates $1.00/1k input and $2.00/1k output, plus
one errored query:

```
error_rate   = 1/3   = 0.333
p50 latency  = 200 ms         the errored query has no latency and is excluded
rag_total    = (200/1000 × 1.00) + (40/1000 × 2.00) = $0.28
rag_per_query= 0.28 / 2 = $0.14
```

**Failed traces never pollute latency** — a 30-second timeout would otherwise
dominate p95 while telling you nothing about how fast the service answers.

**Cost is `None`, never `0`, when a service reports no token usage.** An unknown
cost rendered as free is worse than no number at all. Judge cost is accounted
separately under `ops.judge`, since the panel is roughly 3× a single judge.

## The judge panel

Three models — GLM-5.2, Qwen3.5-397B, Kimi-K2.6 — all served by the Thoth
gateway through the standard `openai` SDK against `${THOTH_BASE_URL}/v1`. Model
ids live in config, not code, so swapping a panel member is a config review.

All three see one identical prompt, so a disagreement is evidence about the
models rather than about the wording each saw. Each dimension is aggregated by
majority vote:

| Panel | Result |
|---|---|
| 4 / 4 / 4 | `unanimous` |
| 4 / 4 / 5 | `majority` — adjacent disagreement, no flag |
| 5 / 5 / 2 | `split` — a majority, but a dissenter two points out |
| 1 / 3 / 5 | `split` — no majority; the median is provisional |

Splits are the point of paying 3× for judging. They are flagged, excluded from
nothing, and surfaced in the scorecard as *awaiting human review*:

```bash
rag-eval adjudicate --run <id> --export flagged.json    # fill in human_scores
rag-eval adjudicate --run <id> --import flagged.json    # overrides + calibration
rag-eval score --run <id>                               # fold them in
rag-eval calibrate --run <id>                           # agreement per dimension and per judge
```

Each adjudication is appended to `datasets/judge_calibration.jsonl`, so the
calibration set grows as the framework is used and panel-vs-human agreement
becomes a tracked number. Seed it with the ~50 human-labelled samples before the
first real judged run:

```bash
rag-eval label seed_labels.json
```

## Adding a RAG service

Subclass `RagAdapter`, implement `answer()`, register it:

```python
from rag_eval.adapters import register
from rag_eval.adapters.base import RagAdapter

@register
class MyRagAdapter(RagAdapter):
    name = "my-rag"

    def answer(self, question: str, question_id: str = "") -> RagTrace:
        trace = self._trace(question, question_id)
        with self.timer() as sw:
            body = ...                       # call your service
        trace.latency_ms = sw.elapsed_ms
        trace.generated_answer = body["answer"]
        trace.retrieved_chunks = [
            self.make_chunk(i + 1, c["text"], sitting_id=c["doc"], page=c["page"])
            for i, c in enumerate(body["sources"])
        ]
        trace.cited_sources = self.citations_from_chunks(trace.retrieved_chunks)
        return trace
```

The adapter's one real job is **populating `sitting_id` and `page`** — every
retrieval and citation metric depends on that metadata surviving the trip out of
the service. `make_chunk` normalises the usual variants (`KKDR_2026-7-14.pdf`,
`"ms. 12"`). Record a service failure as `trace.error` rather than raising, so
one bad question does not end a 371-question run.

The contract tests in `tests/test_adapters.py` are what a new adapter must pass.

## Layout

```
rag_eval/
    types.py        RagTrace, RetrievedChunk, GoldenItem, SourceRef
    config.py       judge model ids, adapter endpoints, cost tables
    dataset/        Excel reader, reference parser, checksummed loader
    adapters/       RagAdapter ABC + nvidia, tanyaparlimen, mock
    judges/         rubric, prompts, gateway client, panel vote, calibration
    metrics/        retrieval, generation, ops, scorecard
    reporting/      HTML scorecard + regression diff
    runner.py       the run stage and ingest-check
    store.py        append-only run artifacts
    cli/            rag-eval
datasets/           golden_v1.jsonl + manifest, calibration set, source workbook
    hansard_pdfs/   the 14 Hansard PDFs, one per sitting + manifest.json (sha256 per document)
                    (see datasets/README.md for the schema and data caveats)
runs/               run artifacts (gitignored)
```

The core is **stdlib-only** on purpose. An eval framework whose golden set can
only be rebuilt when pandas and openpyxl happen to be installed is a framework
that stops being rebuildable; the `.xlsx` reader is ~60 lines of `zipfile` and
`ElementTree`. Only the real judge panel needs a dependency (`openai`), so the
CI smoke test runs the whole pipeline with no GPU, network or gateway key.

## Verifying the page mapping

`ingest-check` proves a chunk *carries* a page number. `page-map-check` proves
that number *means* what the golden set says — a different claim, and the one
`page_citation_accuracy` actually rests on.

```bash
rag-eval page-map-check --adapter nvidia --sample 14 \
  --offsets datasets/hansard_pdfs/offsets.json
```

It retrieves for a golden question and compares three independent values:

| source | |
|---|---|
| **header** | the printed page read out of the chunk's own running header (`DN 4.8.2026 129`) — ground truth |
| **computed** | `page_number + 1 - ms_offset`, what our arithmetic predicts |
| **excel** | the `ms.` the golden set records |

`header` vs `computed` tests the formula, including whether nv-ingest's
`page_number` is 0-based. `header` vs `excel` tests the golden set itself. They
fail differently, so the verdicts are separate — `formula-mismatch` means our
offset is wrong, `golden-mismatch` means the Excel reference is.

The base is **derived, not assumed**: the report states the mapping the data
implies, and warns if it disagrees with the one in use.

```
observed mapping: ms = page_number + BASE - ms_offset, BASE = 1 (4x)
PASS: 4/4 comparable probe(s) map correctly
```

**`no-header` is not a failure.** The running header sits at the top of a page,
and each page becomes ~3.4 chunks, so roughly 70% of retrieved chunks are
mid-page prose with no header to read. Those probes are unverifiable rather
than wrong. Raise `--sample` to get more comparable probes.

A retrieval miss reports `no-hit` — that is a statement about retrieval, not
about page mapping, and is not counted as a mismatch.

Re-run this after **every** re-ingest, including a migration to another cluster:
the offsets are per-document and the page base is a property of the ingestor.

## Data integrity

Two artifacts gate every score, and both are checksummed and verifiable:

```bash
rag-eval dataset-check    # golden_v1.jsonl matches its manifest
rag-eval corpus-check     # every Hansard PDF matches its sha256, and covers the golden set
```

Both exit non-zero on a mismatch, so CI catches a quietly edited golden set or a
re-downloaded Hansard before it silently moves a scorecard. `corpus-check` also
cross-checks the two against each other: a golden sitting with no PDF, or a
citation pointing past the last page of its document, is a failure.

## Testing

```bash
pytest -q
```

Unit tests per metric with synthetic traces (known-good → known-score), adapter
contract tests against a mock service, golden-set schema and checksum
enforcement, judge panel voting and failure handling, and an offline end-to-end
smoke over the whole pipeline including the run-over-run diff.

## Known caveats in the source data

Surfaced by `convert-golden`, all handled rather than papered over:

- **371 Q&A pairs across 14 sittings, 2004–2026** — not the 16 the design spec
  assumes; see `kr_` below. The parliament corpus holds mostly 2025–2026 Hansard,
  so collecting the older PDFs is still a real task.
- **Reference formats vary**: page ranges (`ms. 15-16`), missing dots (`ms 19`),
  page lists (`ms 1-2, 5`), single-digit months (`kkdr_2026-7-14`). All parse to
  a normalised `SourceRef`; a range means "any of these pages" for citation
  matching, since citing one page of a range is correct behaviour.
- **1 row has no reference** (`tp-0100`) — excluded from retrieval metrics.
- **Row 202 reuses `No` 200**, so its id is `tp-0200-r202`. Fix the workbook and
  ids become stable; two questions sharing an id would silently overwrite each
  other's trace.
- **`kr_` was a typo of `kkdr_`, not a sitting type** — resolved, and since
  fixed in the workbook. That is why the corpus is 14 sittings, not the 16 the
  design spec assumes. `DOC_TYPE_ALIASES` still maps it as a guard against
  reappearance. Full evidence in [datasets/README.md](datasets/README.md).

## Not in v1

Automated CI gating on score thresholds (a human reads the scorecard first —
`rag-eval diff --fail-on-regression` exists if you want to opt in), non-parliament
datasets, fine-tuned judges, and a UI dashboard.
