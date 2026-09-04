"""Self-contained HTML scorecard.

No template engine, no CDN: the report has to open from a file share or a
ticket attachment months after the run, on a laptop with no network. Everything
(styles, the small amount of filtering JS, the data) is inlined.
"""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rag_eval.judges.rubric import DIMENSIONS

_CSS = """
:root{--bg:#f7f7f8;--card:#fff;--ink:#1b1c1e;--muted:#6b6f76;--line:#e3e5e9;
--good:#1a7f4b;--bad:#b3261e;--warn:#8a5a00;--warnbg:#fff6e5;--accent:#2b4c7e}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:32px 20px 72px}
h1{font-size:22px;margin:0 0 4px}
h2{font-size:16px;margin:36px 0 12px;padding-bottom:6px;border-bottom:1px solid var(--line)}
.sub{color:var(--muted);margin:0 0 24px}
.sub code{background:#eceef1;padding:1px 5px;border-radius:4px}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.card .k{font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted)}
.card .v{font-size:26px;font-weight:600;margin-top:4px;font-variant-numeric:tabular-nums}
.card .d{font-size:12px;margin-top:2px;font-variant-numeric:tabular-nums}
.up{color:var(--good)}.down{color:var(--bad)}.flat{color:var(--muted)}
.warn{background:var(--warnbg);border:1px solid #f0dcb0;border-left:4px solid var(--warn);
border-radius:8px;padding:12px 16px;margin:18px 0;color:#5b4200}
.stop{background:#fbe9e7;border:1px solid #f3c0ba;border-left:4px solid var(--bad);
border-radius:8px;padding:14px 18px;margin:18px 0;color:#7a1b12}
.stop h3{margin:0 0 6px;font-size:15px}
.stop table{margin-top:8px;background:transparent;border:none}
.stop th,.stop td{border-bottom:1px solid #f3c0ba;padding:5px 10px}
.warn ul{margin:6px 0 0;padding-left:20px}
table{width:100%;border-collapse:collapse;background:var(--card);
border:1px solid var(--line);border-radius:10px;overflow:hidden}
th,td{text-align:left;padding:9px 12px;border-bottom:1px solid var(--line);vertical-align:top}
th{background:#f0f1f4;font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted)}
tr:last-child td{border-bottom:none}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.pill{display:inline-block;padding:1px 8px;border-radius:99px;font-size:11px;font-weight:600}
.pill.improved{background:#e6f4ec;color:var(--good)}
.pill.regressed{background:#fbe9e7;color:var(--bad)}
.pill.flat{background:#eceef1;color:var(--muted)}
.pill.new{background:#e8eefb;color:var(--accent)}
.pill.missing{background:#eceef1;color:var(--muted)}
.pill.split{background:#fff1cc;color:var(--warn)}
.scroll{overflow-x:auto}
.controls{display:flex;gap:10px;align-items:center;margin:12px 0}
.controls input,.controls select{padding:7px 10px;border:1px solid var(--line);
border-radius:8px;font:inherit;background:#fff}
.controls input{flex:1}
details{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:10px 14px;margin-top:10px}
summary{cursor:pointer;font-weight:600}
.q{max-width:420px}
.q .qa{color:var(--muted);font-size:12.5px;margin-top:4px;white-space:pre-wrap}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}
.foot{margin-top:40px;color:var(--muted);font-size:12px}
"""

_JS = """
const rows=[...document.querySelectorAll('#questions tbody tr')];
const q=document.getElementById('filter'),f=document.getElementById('only');
function apply(){const t=q.value.toLowerCase(),m=f.value;
for(const r of rows){const okT=!t||r.dataset.text.includes(t);
const okM=m==='all'||r.dataset[m]==='1';r.hidden=!(okT&&okM);}
document.getElementById('shown').textContent=rows.filter(r=>!r.hidden).length;}
q.addEventListener('input',apply);f.addEventListener('change',apply);apply();
"""


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}".rstrip("0").rstrip(".")
    return str(value)


def _pct(value: Any) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def _delta_html(row: dict[str, Any]) -> str:
    verdict = row["verdict"]
    if verdict in ("new", "missing"):
        return f'<span class="flat">{verdict}</span>'
    if row["delta"] is None:
        return '<span class="flat">—</span>'
    cls = {"improved": "up", "regressed": "down", "flat": "flat"}[verdict]
    sign = "+" if row["delta"] > 0 else ""
    pct = f" ({sign}{row['pct_change']:.1f}%)" if row.get("pct_change") is not None else ""
    return f'<span class="{cls}">{sign}{_fmt(row["delta"], 4)}{pct}</span>'


def _headline_cards(scorecard: dict[str, Any], diff: dict[str, Any] | None) -> str:
    by_metric = {(d["block"], d["metric"]): d for d in (diff or {}).get("metrics", [])}
    cards = []
    for row in scorecard.get("headline", []):
        key = (row["block"], row["metric"])
        value = row["value"]
        shown = _pct(value) if _is_rate(row["metric"]) else _fmt(value)
        delta = by_metric.get(key)
        delta_html = f'<div class="d">{_delta_html(delta)} vs baseline</div>' if delta else ""
        cards.append(
            f'<div class="card"><div class="k">{_esc(row["metric"])}</div>'
            f'<div class="v">{shown}</div>{delta_html}</div>'
        )
    return f'<div class="cards">{"".join(cards)}</div>'


def _is_rate(metric: str) -> bool:
    return (
        metric.startswith(("hit_rate", "recall"))
        or metric.endswith(("_rate", "_accuracy"))
    )


def _warnings_block(warnings: list[str]) -> str:
    if not warnings:
        return ""
    items = "".join(f"<li>{_esc(w)}</li>" for w in warnings)
    return f'<div class="warn"><strong>Read this before the numbers</strong><ul>{items}</ul></div>'


def _comparability_banner(diff: dict[str, Any] | None) -> str:
    """Say plainly when the two runs are not the same system.

    This sits above the headline numbers because a reader who sees "+0.12
    hit_rate@5" first has already drawn the wrong conclusion.
    """
    drift = (diff or {}).get("environment_drift") or {}
    if not diff or diff.get("comparable", True):
        return ""
    rows = "".join(
        f"<tr><td class='mono'>{_esc(d['field'])}</td>"
        f"<td class='mono'>{_esc(d['baseline'])}</td>"
        f"<td class='mono'>{_esc(d['current'])}</td></tr>"
        for d in drift.get("differences", [])
    )
    return (
        '<div class="stop"><h3>These two runs are not the same system</h3>'
        "The RAG stack itself changed between the baseline and this run, so the "
        "differences below measure a different system rather than a change to the same "
        "one. Do <strong>not</strong> read them as improvement or regression."
        f"<table><thead><tr><th>Field</th><th>Baseline</th><th>Current</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></div>"
    )


def _environment_block(env: dict[str, Any]) -> str:
    if not env:
        return ""
    rows = "".join(
        f"<tr><td class='mono'>{_esc(k)}</td><td class='mono'>{_esc(v)}</td></tr>"
        for k, v in env.items()
    )
    return (
        "<h2>RAG environment (as reported by the stack at run time)</h2>"
        f'<div class="scroll"><table><thead><tr><th>Field</th><th>Value</th></tr></thead>'
        f"<tbody>{rows}</tbody></table></div>"
    )


def _diff_table(diff: dict[str, Any] | None) -> str:
    if not diff:
        return ""
    if not diff.get("has_baseline"):
        return (
            '<h2>Regression diff</h2><p class="sub">First run for this adapter — '
            "no baseline to compare against. The next run will diff against this one.</p>"
        )
    rows = "".join(
        f"<tr><td class='mono'>{_esc(r['block'])}.{_esc(r['metric'])}</td>"
        f"<td class='num'>{_fmt(r['baseline'], 4)}</td>"
        f"<td class='num'>{_fmt(r['current'], 4)}</td>"
        f"<td class='num'>{_delta_html(r)}</td>"
        f"<td><span class='pill {r['verdict']}'>{r['verdict']}</span></td></tr>"
        for r in diff.get("metrics", [])
    )
    summary = diff.get("summary", {})
    counts = " · ".join(f"{v} {k}" for k, v in sorted(summary.items()))
    return (
        f"<h2>Regression diff vs <span class='mono'>{_esc(diff.get('baseline_run_id'))}</span></h2>"
        f'<p class="sub">{_esc(counts)}</p>'
        f'<div class="scroll"><table><thead><tr><th>Metric</th><th class="num">Baseline</th>'
        f'<th class="num">Current</th><th class="num">Δ</th><th>Verdict</th></tr></thead>'
        f"<tbody>{rows}</tbody></table></div>"
    )


def _moved_questions(diff: dict[str, Any] | None) -> str:
    changes = (diff or {}).get("per_question") or []
    if not changes:
        return ""
    rows = "".join(
        f"<tr><td class='mono'>{_esc(c['question_id'])}</td>"
        f"<td>{_esc(c['question'])}</td>"
        f"<td class='mono'>{_esc(c['gold_sitting'])}</td>"
        f"<td class='num'>{_fmt(c['hit@5']['baseline'])} → {_fmt(c['hit@5']['current'])}</td>"
        f"<td class='num'>{_fmt(c['correctness']['baseline'])} → {_fmt(c['correctness']['current'])}</td>"
        f"<td><span class='pill {c['direction'] if c['direction'] in ('improved','regressed') else 'flat'}'>"
        f"{c['direction']}</span></td></tr>"
        for c in changes[:100]
    )
    more = (
        f'<p class="sub">Showing 100 of {len(changes)} changed questions.</p>'
        if len(changes) > 100
        else ""
    )
    return (
        f"<h2>Questions that moved ({len(changes)})</h2>{more}"
        f'<div class="scroll"><table><thead><tr><th>ID</th><th>Question</th><th>Gold sitting</th>'
        f'<th class="num">hit@5</th><th class="num">correctness</th><th>Direction</th></tr></thead>'
        f"<tbody>{rows}</tbody></table></div>"
    )


def _judges_table(judges: dict[str, Any], generation: dict[str, Any]) -> str:
    if not judges:
        return ""
    rows = "".join(
        f"<tr><td class='mono'>{_esc(model)}</td>"
        f"<td class='num'>{stats.get('calls', 0)}</td>"
        f"<td class='num'>{_pct(stats.get('failure_rate'))}</td>"
        + "".join(
            f"<td class='num'>{_fmt(stats.get('mean_scores', {}).get(d), 2)}</td>"
            for d in DIMENSIONS
        )
        + "</tr>"
        for model, stats in sorted(judges.items())
    )
    disagreement = generation.get("panel_disagreement", {})
    by_dim = disagreement.get("by_dimension", {})
    dims = ", ".join(f"{d}: {by_dim.get(d, 0)}" for d in DIMENSIONS)
    heads = "".join(f'<th class="num">{d}</th>' for d in DIMENSIONS)
    return (
        "<h2>Judge panel</h2>"
        f'<p class="sub">{disagreement.get("flagged_questions", 0)} question(s) split the panel '
        f'({disagreement.get("awaiting_human_review", 0)} awaiting human adjudication). '
        f"Splits by dimension — {_esc(dims)}.</p>"
        f'<div class="scroll"><table><thead><tr><th>Judge</th><th class="num">Calls</th>'
        f'<th class="num">Failures</th>{heads}</tr></thead><tbody>{rows}</tbody></table></div>'
    )


def _metric_block(title: str, data: dict[str, Any]) -> str:
    rows = []
    for key, value in data.items():
        if isinstance(value, dict):
            for sub, sub_value in value.items():
                if isinstance(sub_value, dict):
                    continue
                rows.append(
                    f"<tr><td class='mono'>{_esc(key)}.{_esc(sub)}</td>"
                    f"<td class='num'>{_fmt(sub_value, 4)}</td></tr>"
                )
        else:
            shown = _pct(value) if _is_rate(key) and isinstance(value, float) else _fmt(value, 4)
            rows.append(f"<tr><td class='mono'>{_esc(key)}</td><td class='num'>{shown}</td></tr>")
    return (
        f"<h2>{_esc(title)}</h2><div class='scroll'><table><thead><tr><th>Metric</th>"
        f"<th class='num'>Value</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def _questions_table(per_question: list[dict[str, Any]]) -> str:
    if not per_question:
        return ""
    rows = []
    for q in per_question:
        retrieval = q.get("retrieval") or {}
        scores = q.get("scores") or {}
        flagged = q.get("flagged") or []
        hit5 = retrieval.get("hit_at_k", {}).get("5")
        text = " ".join(
            str(x) for x in (q.get("question_id"), q.get("question"), q.get("gold_sitting"))
        ).lower()
        rationale = "\n".join(f"{m}: {r}" for m, r in (q.get("rationales") or {}).items())
        flags = (
            f"<span class='pill split'>{_esc(', '.join(flagged))}</span>" if flagged else ""
        )
        error = "<span class='pill regressed'>error</span>" if q.get("error") else ""
        rows.append(
            f"<tr data-text=\"{_esc(text)}\" data-flagged=\"{1 if flagged else 0}\" "
            f"data-miss=\"{0 if hit5 else 1}\" data-error=\"{1 if q.get('error') else 0}\">"
            f"<td class='mono'>{_esc(q.get('question_id'))}</td>"
            f"<td class='q'>{_esc(q.get('question'))}"
            f"<div class='qa'>{_esc((q.get('generated_answer') or q.get('error') or '')[:300])}</div></td>"
            f"<td class='mono'>{_esc(q.get('gold_sitting'))}<br>ms. "
            f"{_esc(', '.join(str(p) for p in q.get('gold_pages', [])) or '—')}</td>"
            f"<td class='num'>{_fmt(hit5)}</td>"
            f"<td class='num'>{_fmt(retrieval.get('citation_page_match'))}</td>"
            + "".join(f"<td class='num'>{_fmt(scores.get(d))}</td>" for d in DIMENSIONS)
            + f"<td>{flags}{error}<div class='qa' title=\"{_esc(rationale)}\">"
            f"{_esc(rationale[:120])}</div></td></tr>"
        )
    heads = "".join(f'<th class="num">{d[:4]}</th>' for d in DIMENSIONS)
    return (
        f"<h2>Per-question results</h2>"
        f'<div class="controls"><input id="filter" type="search" '
        f'placeholder="Filter by question id, text or sitting…">'
        f'<select id="only"><option value="all">All questions</option>'
        f'<option value="flagged">Panel split only</option>'
        f'<option value="miss">Retrieval miss @5 only</option>'
        f'<option value="error">Adapter errors only</option></select>'
        f'<span class="sub" style="margin:0"><span id="shown">0</span> shown</span></div>'
        f'<div class="scroll"><table id="questions"><thead><tr><th>ID</th><th>Question / answer</th>'
        f'<th>Gold ref</th><th class="num">hit@5</th><th class="num">cite</th>{heads}'
        f"<th>Notes</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def render(
    scorecard: dict[str, Any],
    diff: dict[str, Any] | None = None,
    manifest: dict[str, Any] | None = None,
) -> str:
    """Render one run's scorecard, with its regression diff, as a single page."""
    manifest = manifest or {}
    run_id = scorecard.get("run_id", "")
    adapter = scorecard.get("adapter", "")
    dataset = scorecard.get("dataset", "")
    created = scorecard.get("created_at") or manifest.get("created_at", "")
    judge_models = ", ".join(manifest.get("judge_models", [])) or "(not judged)"
    questions = (scorecard.get("retrieval") or {}).get("questions", 0)

    provenance = (
        f"Adapter <code>{_esc(adapter)}</code> · {questions} questions · "
        f"dataset <code>{_esc(Path(dataset).name if dataset else '—')}</code>"
        f"<br>Judges: {_esc(judge_models)} · framework "
        f"<code>{_esc(manifest.get('framework_version', ''))}</code>"
        f"{' @ ' + _esc(manifest.get('git_revision')) if manifest.get('git_revision') else ''}"
        f" · run at {_esc(created)}"
    )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RAG eval — {_esc(run_id)}</title><style>{_CSS}</style></head>
<body><div class="wrap">
<h1>RAG evaluation scorecard</h1>
<p class="sub"><span class="mono">{_esc(run_id)}</span><br>{provenance}</p>
{_comparability_banner(diff)}
{_warnings_block(scorecard.get("warnings", []))}
{_headline_cards(scorecard, diff)}
{_diff_table(diff)}
{_moved_questions(diff)}
{_metric_block("Retrieval", scorecard.get("retrieval", {}))}
{_metric_block("Generation", scorecard.get("generation", {}))}
{_metric_block("Ops", scorecard.get("ops", {}))}
{_judges_table(scorecard.get("judges", {}), scorecard.get("generation", {}))}
{_environment_block(scorecard.get("rag_environment", {}))}
{_questions_table(scorecard.get("per_question", []))}
<details><summary>Run manifest</summary>
<pre class="mono">{_esc(json.dumps(manifest, indent=2, ensure_ascii=False))}</pre></details>
<p class="foot">Generated by rag-eval on
{datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}. Retrieval and ops numbers are
deterministic; generation scores come from an LLM judge panel and carry the caveats above.</p>
</div><script>{_JS}</script></body></html>
"""


def write_report(
    path: str | Path,
    scorecard: dict[str, Any],
    diff: dict[str, Any] | None = None,
    manifest: dict[str, Any] | None = None,
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(scorecard, diff, manifest), encoding="utf-8")
    return out
