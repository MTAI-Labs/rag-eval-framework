# rag-eval-framework

A reusable evaluation harness for any RAG service we build, so that when a RAG
stack changes we can say **with numbers** whether retrieval and answer quality
went up or down.

The framework is product-agnostic: each RAG service plugs in through a thin
adapter. Two ship today — the hosted NVIDIA RAG stack on the RTX6000 cluster and
TanyaParlimen's production RAG.

Implements the approved design in `2026-09-02-rag-eval-framework-design.md`.

---

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"           # add ".[judge]" for the real judge panel
cp .env.example .env              # fill in THOTH_BASE_URL, endpoints, keys

rag-eval convert-golden           # TanyaParlimen QnA.xlsx -> datasets/golden_v1.jsonl
rag-eval dataset-check            # schema + checksum
rag-eval ingest-check --adapter nvidia   # is the collection scoreable at all?
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
| `ingest-check` | Probe the collection: do chunks carry sitting id + page? |
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

**Retrieval** — deterministic, no judge:

| Metric | Definition |
|---|---|
| `hit_rate@k` | ≥1 of the top-k chunks comes from the correct sitting |
| `recall@k` | a top-k chunk carries a **gold page** of the correct sitting |
| `mrr` | mean reciprocal rank of the first correct-sitting chunk |
| `page_citation_accuracy` | the answer's cited `ms.` matches the Excel page |

Two rules that change how the numbers read:

1. Questions with **no golden reference** are excluded from the denominator, not
   scored as misses. Every metric reports its own `scorable` count.
2. A chunk whose metadata lost `sitting_id`/`page` can never match. That is an
   **ingestion** defect, not a retrieval failure, so `metadata_health` reports it
   separately and the scorecard raises a warning below 95% coverage. Check
   `ingest-check` before believing a bad hit-rate.

**Generation** — from the judge panel: `faithfulness`, `correctness`,
`completeness`, `citation_accuracy` (1–5 each), plus `hallucination_rate`
(fraction with faithfulness < 3).

**Ops** — `p50/p95/p99` latency, tokens, estimated cost per query, error rate.
Cost is `None`, never `0`, when a service reports no token usage: an unknown
cost that renders as free is worse than no number.

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
                    (see datasets/README.md for the schema and data caveats)
runs/               run artifacts (gitignored)
```

The core is **stdlib-only** on purpose. An eval framework whose golden set can
only be rebuilt when pandas and openpyxl happen to be installed is a framework
that stops being rebuildable; the `.xlsx` reader is ~60 lines of `zipfile` and
`ElementTree`. Only the real judge panel needs a dependency (`openai`), so the
CI smoke test runs the whole pipeline with no GPU, network or gateway key.

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
- **`kr_` is a typo of `kkdr_`, not a sitting type** — resolved. Both occurrences
  sit inside contiguous `kkdr_` blocks with the same owner, date and page, on
  dates that already have a `kkdr_` sitting. The parser corrects it and
  `convert-golden` reports it. That is why the corpus is 14 sittings, not 16 —
  two fewer PDFs to collect. Full evidence in
  [datasets/README.md](datasets/README.md).

## Not in v1

Automated CI gating on score thresholds (a human reads the scorecard first —
`rag-eval diff --fail-on-regression` exists if you want to opt in), non-parliament
datasets, fine-tuned judges, and a UI dashboard.
