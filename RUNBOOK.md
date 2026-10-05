# RAG evaluation runbook

The standing gate for RAG work. **Run this every time the RAG service changes** —
a new model, a new chunker, a new retriever, a reindex, a prompt edit, a new
collection. A change that is not measured against this yardstick is not shippable.

Six workflows, in the order you will need them:

1. [Re-run the benchmark after a RAG change](#1-re-run-the-benchmark-after-a-rag-change)
2. [Read the scorecard and the regression diff](#2-read-the-scorecard-and-the-regression-diff)
3. [The go/no-go decision rule](#3-the-gono-go-decision-rule)
4. [Adjudicate flagged rows](#4-adjudicate-flagged-rows)
5. [Extend the golden set from a UAT round](#5-extend-the-golden-set-from-a-uat-round)
6. [Add a new RAG service adapter](#6-add-a-new-rag-service-adapter)

Setup, metric formulas and architecture live in [README.md](README.md); this
document is the operating manual.

---

## The current baseline

Every comparison is against this run until it is superseded. Supersede it only
by a run that passed the go/no-go rule below.

| | |
|---|---|
| **Run** | `20260923-034301-nvidia` |
| Adapter / collection | `nvidia` / `parliament_hansard_eval` |
| RAG LLM | `zai-org/GLM-5.3-Flash` |
| Golden set | 371 questions, sha256 `138f682c731d…` |
| Framework revision | `b6f0e88` |

**Retrieval** — hit_rate@5 **0.989** · recall@5 **0.930** · MRR **0.972** ·
page_citation_accuracy **0.930** · no_citation_rate 0.000

**Generation** (1–5 scale) — faithfulness **4.94** · correctness **4.72** ·
citation_accuracy **4.71** · completeness **4.31** · hallucination_rate
**0.008** · empty_answer_rate 0.000

**Ops** — 371/371 succeeded, error_rate **0.000** · latency p50 **1.41s**,
p95 **121.9s**, p99 **123.5s**, max **130.7s**

Two caveats that belong with these numbers wherever they are quoted:

- **Latency is bimodal, not a long tail.** p50 is 1.4s but p95/p99/max cluster
  at 122–131s (p95 is 86x p50). That shape is a fixed upstream
  timeout-and-retry, not compute. Do not quote p95 as steady-state cost, and do
  not treat a p95 regression as a retrieval-quality signal. The scorecard now
  detects this automatically and says so in its caveats.
- **The 6pp gap between hit_rate (0.989) and recall (0.930) is a chunk-boundary
  artefact, not a retrieval failure.** Of the 23 questions that found the right
  sitting but not the right page, **19 are off by only 1–2 pages** — the answer
  spans a page break, so the chunk carrying it is labelled with the adjacent
  page. Only 4 are genuinely far off.
- **`faithfulness` cannot discriminate.** The panel returns 5 on essentially
  every row; human labelling found 12 of 50 scored lower, 5 at the maximum gap.
  Treat 4.94 as a ceiling artefact. `completeness` and `correctness` carry the
  real generation signal.

---

## 1. Re-run the benchmark after a RAG change

Four composable stages. Each consumes and produces versioned artifacts under
`runs/<run-id>/`, so you can stop after any stage and resume later.

```bash
# 0. pre-flight — never spend an hour to discover the service was down
rag-eval dataset-check                       # golden set intact
rag-eval smoke --adapter nvidia --limit 5    # service answers at all

# 1-4. the pipeline
rag-eval run    --adapter nvidia --concurrency 4 --note "what changed"
rag-eval judge  --run <run-id>
rag-eval score  --run <run-id>
rag-eval report --run <run-id>
```

`rag-eval eval` chains all four, but prefer the separate stages for a real
benchmark: if judging fails you keep the traces rather than re-paying for
retrieval.

**Always pass `--note`** saying what changed ("switched RAG LLM to GLM-5.3-Flash",
"chunk size 512→1024"). It lands in the manifest and is the only thing that
explains a delta six weeks later.

Expect roughly 30 minutes for `run` and 8 minutes for `judge` at 371 questions,
but treat that as weather, not climate — the same run took 8 hours on a degraded
day. An interrupted run resumes with `rag-eval run --resume <run-id>`.

### Before you trust the traces

```bash
rag-eval ingest-check  --adapter nvidia   # chunks carry sitting_id + page
rag-eval page-map-check --adapter nvidia  # a chunk's page means the golden ms.
```

Run both after **any reindex or re-ingest**. `page-map-check` is the one that
catches the expensive failure: chunks carrying physical PDF pages instead of
printed `ms.` pages silently drove recall to 10% when it was really 93%.

---

## 2. Read the scorecard and the regression diff

Open `runs/<run-id>/scorecard.html` — self-contained, no network needed. Column
headers sort; the filter box narrows to flagged, missed or errored questions.

Read it in this order:

**1. The comparability banner, before any number.** If the diff says
`NOT COMPARABLE`, the stack changed underneath (LLM, embedding model, retrieval
mode, embedding profile) and the deltas describe a *different system*, not an
improvement to this one. The baseline diff says exactly this, because the RAG
LLM changed between runs. A metric that "improved" across an incomparable
boundary tells you nothing.

**2. Ops, to decide whether the run is even valid.** A non-zero `error_rate`
means questions failed at the adapter; those are excluded from generation means,
so a scorecard with 20% errors is measuring the 80% that happened to work.

**3. Retrieval, which is deterministic and needs no judge.**
- `hit_rate@k` — found the right *sitting*
- `recall@k` — found the right *page* within it
- A wide gap between them is a **chunking** problem, not a retriever problem.
  The baseline's 0.989 vs 0.930 is that gap at its healthy size.
- `metadata_health.chunks_without_ms_offset` above zero means some documents
  were ingested without an offset and their page metrics are under-reporting.
  Fix the ingest, don't interpret the number.

**4. Generation, weighted by what each dimension is worth.** `completeness` and
`correctness` discriminate. `citation_accuracy` runs ~0.6 more generous than
humans. `faithfulness` is a ceiling — see the caveat above.

**5. Per-question deltas, sorted by the metric that moved.** Sorting reads the
*current* value in a `0.80 → 0.40` cell, so sorting ascending surfaces the
questions that fell furthest.

---

## 3. The go/no-go decision rule

Apply in order. The first rule that fires decides.

**Stop — the run is not evidence:**

| Condition | Meaning |
|---|---|
| Diff says `NOT COMPARABLE` | You are comparing two different systems. Re-baseline deliberately or revert the stack change. |
| `ops.error_rate > 0.02` | Too many questions failed to be a fair sample. Fix the service, re-run. |
| `metadata_health.chunks_without_ms_offset > 0` | Page metrics are structurally wrong. Re-ingest with `--offsets`. |
| `empty_answer_rate > 0.02` | The service is answering nothing. Not a quality question yet. |

**No-go — a real regression:**

| Condition | Rationale |
|---|---|
| `recall@5` drops **> 2pp** | The evidence is no longer being found. The single most important number. |
| `hit_rate@5` drops **> 2pp** | Retrieval lost the document entirely. |
| `hallucination_rate` rises **> 1pp** | Answers asserting things the context does not support. |
| `correctness` or `completeness` drops **> 0.2** | Human-visible answer quality. |
| `ops.latency_ms.p50` **doubles** | Use p50, never p95 — p95 measures the retry ceiling. |

**Go, with a note:** everything else, including improvements. Record the run id
and the `--note` in the change's PR or ticket.

**Go requires one more thing:** a re-run of `page-map-check` if the change
touched ingestion, chunking or the collection. A reindex that quietly drops
`ms_offset` passes every threshold above while making the page metrics
meaningless.

> The thresholds are deliberately blunt. With 371 questions, a 2pp move in
> recall is about 7 questions — large enough to be real, small enough to catch a
> genuine regression. Tighten them only with a bigger golden set.

---

## 4. Adjudicate flagged rows

When the three judges disagree beyond `flag_spread_threshold` (default 2), the
row is flagged and its score is a provisional median. The baseline run flagged
25 of 371. The scorecard says so in its caveats.

```bash
# export what needs a human
rag-eval adjudicate --run <run-id> --export flagged.json

# fill in the human scores, then
rag-eval adjudicate --run <run-id> --import flagged.json --labelled-by <name>
rag-eval score  --run <run-id>      # re-score with human scores overriding
rag-eval report --run <run-id>
```

Human scores override the panel wherever they exist (`PanelVerdict.final_scores`).
Adjudicated decisions also feed the calibration set, so adjudication improves the
panel's measured agreement over time rather than being throwaway work.

**When to bother:** if flagged rows are under ~5% and the decision is a clear go,
adjudication rarely changes it. Adjudicate when the decision is marginal, when
flagged rows exceed ~10%, or when a specific dimension is what the change was
meant to improve.

---

## 5. Extend the golden set from a UAT round

New Q&A pairs arrive as a workbook with columns
`No | Question | Answer | Owner | Reference` (Malay headers `Soalan`, `Jawapan`,
`Rujukan` also work).

```bash
rag-eval convert-golden --xlsx datasets/<new>.xlsx \
  --output datasets/golden_v2.jsonl --report conversion-report.json
rag-eval dataset-check --dataset datasets/golden_v2.jsonl
rag-eval corpus-check --dataset datasets/golden_v2.jsonl   # cites real pages?
```

**References must parse** as `<doc_type>_<yyyy-mm-dd>[, ms. N]` — e.g.
`dr_2026-06-22, ms. 9`. Ranges (`ms. 15-16`) and lists (`ms 1-2, 5`) are fine.
`--strict` fails the conversion on any unparseable or unknown reference; use it
for a UAT round you control, and read `conversion-report.json` either way — it
names every dropped row.

**A new sitting means new work.** If the round cites a document not in the
corpus, you must add the PDF as `<sitting_id>.pdf`, regenerate the manifest and
offsets, and re-ingest:

```bash
rag-eval corpus-check --write
rag-eval page-offsets --set <sitting>=<offset> --output datasets/hansard_pdfs/offsets.json
rag-eval ingest --offsets datasets/hansard_pdfs/offsets.json
```

**Version the new set rather than editing the old one.** `golden_v1.jsonl` is
the baseline's checksum; changing it in place makes every prior scorecard
unreproducible. Bump to `golden_v2.jsonl` and re-baseline deliberately — a run
against a different golden set is not comparable to one against this baseline,
and the manifest records the checksum so the diff can tell.

Offsets that automatic measurement cannot determine must be passed with `--set`;
they are stored in the file's `verified` block and always win over re-measurement.

---

## 6. Add a new RAG service adapter

One class, one registry entry, no changes to metrics or reporting.

```python
# rag_eval/adapters/myservice.py
from rag_eval.adapters import register
from rag_eval.adapters.base import RagAdapter
from rag_eval.types import RagTrace

@register
class MyServiceAdapter(RagAdapter):
    name = "myservice"

    def answer(self, question: str, question_id: str = "") -> RagTrace:
        trace = self._trace(question, question_id)
        ...
        return trace
```

**The contract, which the shared test suite enforces:**

- `answer()` returns a `RagTrace` **always**. A service failure is
  `trace.error`, never an exception — one broken question must not kill a
  371-question run.
- Build chunks with `RagAdapter.make_chunk(...)` and pass `ms_offset`. It
  normalises sitting ids, parses page strings, and converts physical pages to
  printed `ms.` Skipping it is how page metrics silently break.
- Read chunk metadata through configurable field paths, not hardcoded keys.
  Settings merge shallowly (`{**DEFAULTS, **options}`), so an override in
  config **replaces** the whole list — copy the full default list when
  overriding.

Then:

```bash
pytest tests/test_adapter_contract.py      # 16 shared tests, all adapters
rag-eval smoke --adapter myservice --limit 5   # live, 5 golden questions
```

Both must pass before the adapter is used for a benchmark. Register config under
`adapters.myservice.options` in `rag-eval.config.json`.

> **Scope limit worth knowing:** the adapter layer is service-agnostic, but the
> *evidence model* is not — the framework assumes citations are `sitting + printed
> page`. A corpus whose citations are shaped differently needs more than an
> adapter.

---

## When things break

| Symptom | Cause | Fix |
|---|---|---|
| Ingest: all files `submitted`, `0 elements` | nv-ingest frozen | Restart `nv-ingest-ms-runtime` |
| recall ≈ 0.10 with healthy hit_rate | chunks carry physical pages | Re-ingest with `--offsets` |
| `calibrate` reports 0 overlapping samples | labels and panel scores from different runs | Use `--sheet`, or judge the run first |
| p95 latency ≈ 122s, p50 ≈ 1.4s | upstream timeout-and-retry | GPU-side; not a quality regression |
| Judge: "no JSON object in reply" | reasoning consumed `max_tokens` | `enable_thinking: false`, raise `max_tokens` |
| Answers begin "The user asks…" | RAG LLM leaking chain-of-thought | Fix server-side; scores are invalid until then |
