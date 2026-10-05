# RAG Evaluation Framework — how we measure any RAG change

**Owner:** Aiman Faris · MTAI Labs  ·  **Status:** in use, standing gate
**Repo:** `rag-eval-framework`  ·  **Runbook:** `RUNBOOK.md` in the repo root

## What it is

A product-agnostic harness that measures any RAG service on three axes —
**retrieval**, **generation** (a 3-model LLM-judge panel), and **ops** — and
diffs every run against the previous one. It exists so that a change to a RAG
service can be *shown* to be an improvement rather than asserted.

It is not tied to one RAG product: services plug in behind a small adapter
interface. Today it measures the NVIDIA RAG Blueprint over the Malaysian
Parliament Hansard corpus.

## Why you would use it

Run it whenever the RAG service changes — a new LLM, a new chunker, a different
retriever, a reindex, a prompt edit, a new collection. It answers, with
evidence:

- Did retrieval still find the right document and the right page?
- Did answer quality move, and on which dimension?
- Did anything get slower or start failing?
- **Is this change safe to ship?** — there is an explicit go/no-go rule.

## Baseline (2026-09-23)

Run `20260923-034301-nvidia` · 371 golden questions · collection
`parliament_hansard_eval` · RAG LLM `zai-org/GLM-5.3-Flash` · golden set sha256
`138f682c731d…`

| Axis | Numbers |
|---|---|
| **Retrieval** | hit_rate@5 **0.989** · recall@5 **0.930** · MRR **0.972** · page_citation_accuracy **0.930** |
| **Generation** (1–5) | faithfulness 4.94 · correctness **4.72** · citation_accuracy 4.71 · completeness **4.31** · hallucination_rate **0.008** |
| **Ops** | 371/371 succeeded · error_rate **0.000** · latency p50 **1.41s** / p95 121.9s |

**Judge panel:** 3 models via the Thoth gateway (GLM-5.2, Qwen3.5-397B,
Kimi-K2.6), majority vote, disagreement flagged for human adjudication. 1,113
judge calls, 0 failures. 25 of 371 rows flagged.

**Panel–human agreement:** 86.5% (within-1) against 50 stratified samples
hand-labelled by five reviewers, clearing the 0.8 target.

### Read these caveats with the numbers

- **Latency is bimodal.** p50 1.4s, but p95/p99/max cluster at 122–131s. That
  shape is a fixed upstream timeout-and-retry, not compute cost. Quote p50.
- **`faithfulness` 4.94 is a ceiling artefact**, not a quality finding — the
  panel returns 5 on essentially every row, while human labellers scored 12 of
  50 lower. `completeness` and `correctness` carry the real signal.
- **The 0.930 recall depends on a page-offset conversion.** Hansard cites the
  *printed* `ms.` page; the ingestor reports the *physical* PDF page. Before
  that conversion was applied, the same corpus measured 10% recall. Any reindex
  must re-verify it with `rag-eval page-map-check`.

## The standing workflow

```bash
rag-eval dataset-check                    # golden set intact
rag-eval smoke --adapter nvidia --limit 5 # service alive
rag-eval run   --adapter nvidia --note "what changed"
rag-eval judge  --run <run-id>
rag-eval score  --run <run-id>
rag-eval report --run <run-id>            # -> scorecard.html + regression diff
```

Each stage writes versioned artifacts under `runs/<run-id>/`, and the manifest
records config, adapter, collection, judge model ids, golden-set checksum, git
revision and timestamp — so any scorecard is reproducible and attributable.

**Go/no-go, in short.** Stop if the diff says `NOT COMPARABLE` (the stack
changed, so the deltas describe a different system), if error rate exceeds 2%,
or if any chunk lost its page offset. Otherwise it is a no-go if recall@5 or
hit_rate@5 drops more than 2pp, hallucination_rate rises more than 1pp,
correctness or completeness drops more than 0.2, or p50 latency doubles. Full
rule in `RUNBOOK.md`.

## Scope and limits

**Reusable today:** the adapter layer, the judge panel, generation and ops
metrics, run storage and the regression diff — all corpus-agnostic. A new RAG
service is one subclass plus a config entry, validated by a 16-test shared
contract suite.

**Not yet reusable:** the *evidence model*. The framework assumes a citation is
`sitting + printed page`, which is baked into the golden-set format, the ingest
metadata schema and the retrieval metrics. A corpus citing "§4.2" or "clause
7(b)" needs more than a new adapter. Generalising this is the next piece of
work.

**Also worth knowing:** the collection's metadata schema is fixed at creation —
getting it wrong means dropping and re-ingesting, not editing config.

## Where things are

- **Runbook** (the operating manual, six workflows) — `RUNBOOK.md`
- **Metric formulas with worked examples** — `README.md`
- **Baseline scorecard** — `runs/20260923-034301-nvidia/scorecard.html`
- **Judge calibration report** — `datasets/calibration_report.json`
- **Golden set** — `datasets/golden_v1.jsonl` (371 questions, checksummed)

## Keywords for search

RAG evaluation, RAG benchmark, retrieval metrics, recall@k, hit rate, MRR,
LLM-as-judge, judge panel, hallucination rate, regression diff, scorecard,
golden set, Hansard, parliament, NVIDIA RAG Blueprint, nv-ingest, Milvus,
Thoth gateway, go/no-go, RAG regression testing, rag-eval
