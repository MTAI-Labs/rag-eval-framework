"""End-to-end smoke: the whole pipeline on a 10-question slice, offline.

No GPU, no network, no gateway -- a mock adapter and a scripted judge client.
This is the test that must stay green in CI, because it is what proves the
stages still compose after any one of them changes.
"""

from __future__ import annotations

import json

import pytest
from conftest import chunk, ref, trace

from rag_eval.adapters import get_adapter
from rag_eval.config import Config, JudgeConfig, JudgeModel
from rag_eval.dataset.loader import load_golden_set, write_golden_set
from rag_eval.judges.client import ScriptedChatClient
from rag_eval.judges.panel import JudgePanel
from rag_eval.metrics.scorecard import build_scorecard
from rag_eval.reporting import compare, write_report
from rag_eval.runner import ingest_check, run_dataset
from rag_eval.store import RunManifest, RunStore, previous_run
from rag_eval.types import GoldenItem

SITTINGS = ["dr_2026-06-22", "dn_2026-08-04", "kkdr_2026-07-14"]


def slice_of_ten() -> list[GoldenItem]:
    items = []
    for i in range(10):
        doc, _, date = SITTINGS[i % len(SITTINGS)].partition("_")
        items.append(
            GoldenItem(
                id=f"tp-{i:04d}",
                question=f"Question {i}?",
                expected_answer=f"Golden answer {i}.",
                reference=ref(doc, date, i + 1),
            )
        )
    return items


def fixture_traces(items: list[GoldenItem], *, good_through: int) -> str:
    """Traces where the first ``good_through`` questions retrieve correctly."""
    lines = []
    for index, item in enumerate(items):
        gold = item.reference
        if index < good_through:
            chunks = [chunk(1, gold.sitting_id, gold.pages[0]), chunk(2, "dn_2019-05-06", 2)]
            cited = [ref(gold.doc_type, gold.sitting_date, gold.pages[0])]
        else:
            chunks = [chunk(1, "dn_2019-05-06", 2)]
            cited = [ref("dn", "2019-05-06", 2)]
        lines.append(json.dumps(
            trace(item.id, chunks, cited=cited, answer=f"Answer {index}.").to_dict()
        ))
    return "\n".join(lines) + "\n"


def judge_reply(faithfulness: int = 5) -> str:
    return json.dumps({"scores": {"faithfulness": faithfulness, "correctness": 4,
                                  "completeness": 4, "citation_accuracy": 4},
                       "rationale": "supported by the context"})


@pytest.fixture
def workspace(tmp_path):
    items = slice_of_ten()
    dataset = tmp_path / "golden_v1.jsonl"
    write_golden_set(dataset, items, source="smoke")
    (tmp_path / "runs").mkdir()
    return tmp_path, dataset, items


def make_config(tmp_path) -> Config:
    config = Config(runs_dir=tmp_path / "runs")
    config.judge = JudgeConfig(models=[JudgeModel(id=m) for m in ("glm", "qwen", "kimi")])
    return config


def execute(tmp_path, dataset, *, good_through: int, judge_scores: int = 5) -> RunStore:
    """One full pipeline pass; returns the completed run store."""
    config = make_config(tmp_path)
    items = load_golden_set(dataset)

    fixture = tmp_path / f"traces-{good_through}.jsonl"
    fixture.write_text(fixture_traces(items, good_through=good_through), encoding="utf-8")

    adapter = get_adapter("mock", fixture=str(fixture), top_k=5)
    store = RunStore.create(config.runs_dir, "mock")
    store.write_manifest(RunManifest(
        run_id=store.run_id, adapter="mock", created_at="2026-09-02T00:00:00+00:00",
        dataset=str(dataset), questions_run=len(items),
        judge_models=[m.id for m in config.judge.models],
    ))

    traces = run_dataset(adapter, items, store)

    panel = JudgePanel(ScriptedChatClient({}, default=judge_reply(judge_scores)),
                       config.judge, max_workers=1)
    verdicts = panel.judge_all(zip(items, sorted(traces, key=lambda t: t.question_id)))
    store.write_judgments(verdicts)

    card = build_scorecard(store.run_id, "mock", items, traces, verdicts,
                           config=config, dataset=str(dataset))
    store.write_scorecard(card.to_dict())

    baseline = previous_run(config.runs_dir, "mock", before=store.run_id)
    diff = compare(card.to_dict(), baseline.read_scorecard() if baseline else None)
    write_report(store.report_path, card.to_dict(), diff, store.read_manifest().to_dict())
    return store


def test_full_pipeline_produces_every_artifact(workspace):
    tmp_path, dataset, items = workspace
    store = execute(tmp_path, dataset, good_through=8)

    assert store.traces_path.exists()
    assert store.judgments_path.exists()
    assert store.scorecard_path.exists()
    assert store.report_path.exists()
    assert len(store.read_traces()) == 10
    assert len(store.read_judgments()) == 10


def test_metrics_match_the_fixture_by_construction(workspace):
    tmp_path, dataset, _ = workspace
    card = execute(tmp_path, dataset, good_through=8).read_scorecard()

    # 8 of 10 questions retrieve and cite their gold sitting and page.
    assert card["retrieval"]["hit_rate@5"] == 0.8
    assert card["retrieval"]["recall@5"] == 0.8
    assert card["retrieval"]["page_citation_accuracy"] == 0.8
    assert card["retrieval"]["mrr"] == 0.8  # correct chunk is always rank 1
    assert card["generation"]["faithfulness"] == 5.0
    assert card["generation"]["hallucination_rate"] == 0.0
    assert card["ops"]["error_rate"] == 0.0
    assert len(card["per_question"]) == 10


def test_a_second_run_diffs_against_the_first(workspace):
    tmp_path, dataset, _ = workspace
    first = execute(tmp_path, dataset, good_through=8)
    second = execute(tmp_path, dataset, good_through=4, judge_scores=2)

    diff = compare(second.read_scorecard(), first.read_scorecard())
    assert diff["baseline_run_id"] == first.run_id
    assert diff["has_baseline"]

    regressed = {r["metric"] for r in diff["regressed"]}
    assert "hit_rate@5" in regressed
    assert "page_citation_accuracy" in regressed
    assert "hallucination_rate" in regressed  # faithfulness 2 -> every answer hallucinates

    # And the diff names the specific questions to go and debug: 4-7 stopped
    # retrieving their sitting. 8 and 9 were already missing in the baseline, so
    # they are not "changes" -- a diff that relisted them every run would be noise.
    moved = {c["question_id"] for c in diff["per_question"]}
    assert moved == {f"tp-{i:04d}" for i in range(4, 8)}
    assert all(c["direction"] == "regressed" for c in diff["per_question"])


def test_the_html_report_opens_without_a_network(workspace):
    tmp_path, dataset, _ = workspace
    store = execute(tmp_path, dataset, good_through=8)
    html = store.report_path.read_text(encoding="utf-8")

    assert html.startswith("<!doctype html>")
    assert "src=\"http" not in html and "href=\"http" not in html
    assert "tp-0000" in html


def test_an_interrupted_run_resumes_where_it_stopped(workspace):
    tmp_path, dataset, items = workspace
    config = make_config(tmp_path)
    fixture = tmp_path / "traces.jsonl"
    fixture.write_text(fixture_traces(items, good_through=10), encoding="utf-8")

    store = RunStore.create(config.runs_dir, "mock")
    adapter = get_adapter("mock", fixture=str(fixture))
    run_dataset(adapter, items[:4], store)
    assert len(store.read_traces()) == 4

    run_dataset(adapter, items, store, resume=True)
    trace_ids = [t.question_id for t in store.read_traces()]
    assert len(trace_ids) == 10
    assert len(set(trace_ids)) == 10  # resumed, not duplicated


def test_ingest_check_fails_loudly_when_metadata_is_missing(workspace):
    tmp_path, dataset, items = workspace
    # The default mock adapter returns chunks with no sitting id or page --
    # exactly the failure mode design spec §6 warns about.
    report = ingest_check(get_adapter("mock", top_k=3), items, sample=6)

    assert report["probes"] == 6
    assert report["sitting_id_coverage"] == 0.0
    assert report["page_citation_accuracy_computable"] is False
    assert report["verdict"].startswith("FAIL: no chunk carries a sitting id")


def test_ingest_check_passes_on_a_healthy_collection(workspace):
    tmp_path, dataset, items = workspace
    fixture = tmp_path / "traces.jsonl"
    fixture.write_text(fixture_traces(items, good_through=10), encoding="utf-8")
    report = ingest_check(get_adapter("mock", fixture=str(fixture)), items, sample=6)

    assert report["sitting_id_coverage"] == 1.0
    assert report["page_coverage"] == 1.0
    assert report["verdict"].startswith("PASS")


def test_ingest_check_probes_spread_across_sittings(workspace):
    tmp_path, dataset, items = workspace
    fixture = tmp_path / "traces.jsonl"
    fixture.write_text(fixture_traces(items, good_through=10), encoding="utf-8")
    report = ingest_check(get_adapter("mock", fixture=str(fixture)), items, sample=3)

    probed = {r["gold_sitting"] for r in report["probe_results"]}
    assert probed == set(SITTINGS), "sampling one sitting proves nothing about the other 15"


def test_adapter_errors_flow_through_to_the_scorecard(workspace):
    tmp_path, dataset, items = workspace
    config = make_config(tmp_path)
    adapter = get_adapter("mock", fail_ids=[items[0].id, items[1].id])
    store = RunStore.create(config.runs_dir, "mock")
    traces = run_dataset(adapter, items, store)

    panel = JudgePanel(ScriptedChatClient({}, default=judge_reply()), config.judge, max_workers=1)
    verdicts = panel.judge_all(zip(items, sorted(traces, key=lambda t: t.question_id)))
    card = build_scorecard(store.run_id, "mock", items, traces, verdicts, config=config)

    assert card.ops["error_rate"] == 0.2
    assert card.generation["unjudged"] == 2
    assert panel.usage.calls == 24  # 8 judged questions x 3 judges, failures never sent
    assert any("error rate" in w for w in card.warnings)


def test_every_compared_metric_exists_in_a_real_scorecard(workspace):
    # A metric listed in the wrong block renders as "missing" forever and the
    # regression it was meant to catch goes unnoticed.
    from rag_eval.metrics.scorecard import HEADLINE
    from rag_eval.reporting.diff import EXTRA_COMPARISONS

    tmp_path, dataset, _ = workspace
    card = execute(tmp_path, dataset, good_through=8).read_scorecard()

    for block, metric, _ in tuple(HEADLINE) + EXTRA_COMPARISONS:
        assert metric in card[block], f"{block}.{metric} is not produced by the scorecard"
