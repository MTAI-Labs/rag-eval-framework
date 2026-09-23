"""Calibration set: the human-labelled subset the panel is measured against.

Design spec §4.1 -- judges are calibrated on ~50 human-labelled samples before
the first full run, and every later human adjudication of a split row is
appended, so the calibration set grows as the framework is used and the panel's
agreement with humans becomes a tracked number rather than an assumption.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from rag_eval.judges.rubric import DIMENSIONS, JudgeVerdict, PanelVerdict, clamp
from rag_eval.types import GoldenItem

DEFAULT_PATH = Path("datasets/judge_calibration.jsonl")


@dataclass
class CalibrationSample:
    question_id: str
    human_scores: dict[str, int]
    labelled_by: str = ""
    labelled_at: str = ""
    source: str = "manual"          # manual | adjudication
    run_id: str = ""
    notes: str = ""
    panel_scores: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "human_scores": dict(self.human_scores),
            "panel_scores": dict(self.panel_scores),
            "labelled_by": self.labelled_by,
            "labelled_at": self.labelled_at,
            "source": self.source,
            "run_id": self.run_id,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CalibrationSample":
        return cls(
            question_id=str(d["question_id"]),
            human_scores={k: int(v) for k, v in (d.get("human_scores") or {}).items()},
            panel_scores={k: int(v) for k, v in (d.get("panel_scores") or {}).items()},
            labelled_by=d.get("labelled_by", ""),
            labelled_at=d.get("labelled_at", ""),
            source=d.get("source", "manual"),
            run_id=d.get("run_id", ""),
            notes=d.get("notes", ""),
        )


#: Column order for the labelling sheet.
#:
#: The material to read comes first, then the human's own scores, and the
#: panel's scores last. That ordering is deliberate: a labeller who sees the
#: judges' answers before forming their own is anchored by them, and the whole
#: point of calibration is an independent opinion to measure the panel against.
CSV_COLUMNS = [
    "question_id", "owner", "sitting_id", "golden_ms",
    "question", "expected_answer", "generated_answer", "retrieved_context",
    "human_faithfulness", "human_correctness", "human_completeness",
    "human_citation_accuracy", "labelled_by", "notes",
    "panel_faithfulness", "panel_correctness", "panel_completeness",
    "panel_citation_accuracy", "flagged_dimensions", "judges_responded",
    "judge_rationales", "adapter_error",
]


def sheet_to_rows(sheet: dict[str, Any]) -> list[dict[str, str]]:
    """Flatten a calibration sheet into CSV-shaped rows."""
    out = []
    for r in sheet.get("rows", []):
        context = "\n\n".join(
            f"[{c['rank']}] {c.get('sitting_id')} p.{c.get('page')}: {c.get('text', '')}"
            for c in r.get("retrieved", [])
        )
        rationales = "\n\n".join(
            f"{m}: {t}" for m, t in (r.get("judge_rationales") or {}).items()
        )
        panel = r.get("panel_scores") or {}
        human = r.get("human_scores") or {}
        row = {
            "question_id": r.get("question_id", ""),
            "owner": r.get("owner", ""),
            "sitting_id": r.get("sitting_id") or "",
            "golden_ms": ", ".join(str(p) for p in (r.get("golden_ms") or [])),
            "question": r.get("question", ""),
            "expected_answer": r.get("expected_answer", ""),
            "generated_answer": r.get("generated_answer", ""),
            "retrieved_context": context,
            "labelled_by": r.get("labelled_by", ""),
            "notes": r.get("notes", ""),
            "flagged_dimensions": ", ".join(r.get("flagged_dimensions") or []),
            "judges_responded": str(len(r.get("judge_scores") or {})),
            "judge_rationales": rationales,
            "adapter_error": r.get("adapter_error") or "",
        }
        for d in DIMENSIONS:
            row[f"human_{d}"] = "" if human.get(d) is None else str(human[d])
            row[f"panel_{d}"] = "" if panel.get(d) is None else str(panel[d])
        out.append(row)
    return out


def _normalise_keys(row: dict[str, Any]) -> dict[str, Any]:
    """Accept both ``human_faithfulness`` and ``human faithfulness``.

    The .xlsx workbook prettifies headers for people, so a CSV exported back out
    of Excel carries spaces where the machine-written CSV has underscores.
    Tolerating both means a sheet labelled in a spreadsheet imports without
    anyone having to rename columns — and without silently finding no labels.
    """
    return {str(k).strip().lower().replace(" ", "_"): v for k, v in row.items() if k}


def rows_to_samples(rows: Sequence[dict[str, Any]], labelled_by: str = "") -> list[CalibrationSample]:
    """Read human scores back out of a filled-in CSV.

    A row with no scores is skipped rather than recorded as blank — an
    unlabelled row must not look like a label.
    """
    samples = []
    for raw_row in rows:
        row = _normalise_keys(raw_row)
        scores: dict[str, int] = {}
        for d in DIMENSIONS:
            raw = str(row.get(f"human_{d}") or "").strip()
            if raw:
                value = clamp(raw)
                if value is not None:
                    scores[d] = value
        if not scores:
            continue
        samples.append(CalibrationSample(
            question_id=str(row.get("question_id", "")).strip(),
            human_scores=scores,
            panel_scores={d: int(row[f"panel_{d}"]) for d in DIMENSIONS
                          if str(row.get(f"panel_{d}") or "").strip().isdigit()},
            labelled_by=str(row.get("labelled_by") or labelled_by).strip(),
            notes=str(row.get("notes") or "").strip(),
            source="manual",
        ))
    return samples


def load_calibration(path: str | Path = DEFAULT_PATH) -> list[CalibrationSample]:
    p = Path(path)
    if not p.exists():
        return []
    samples = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            samples.append(CalibrationSample.from_dict(json.loads(line)))
    return samples


def append_calibration(
    samples: Iterable[CalibrationSample], path: str | Path = DEFAULT_PATH
) -> int:
    """Append samples, skipping question ids already labelled.

    Append-only on purpose: a human label is evidence, and silently replacing
    one would make the panel's agreement history unreproducible.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    existing = {s.question_id for s in load_calibration(p)}
    written = 0
    with open(p, "a", encoding="utf-8") as fh:
        for sample in samples:
            if sample.question_id in existing:
                continue
            sample.labelled_at = sample.labelled_at or datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            )
            fh.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")
            existing.add(sample.question_id)
            written += 1
    return written


def samples_from_adjudications(
    verdicts: Iterable[PanelVerdict], *, run_id: str = "", labelled_by: str = ""
) -> list[CalibrationSample]:
    """Turn human adjudications of split rows into calibration samples."""
    return [
        CalibrationSample(
            question_id=v.question_id,
            human_scores=dict(v.human_scores),
            panel_scores=dict(v.scores),
            source="adjudication",
            run_id=run_id,
            labelled_by=labelled_by,
        )
        for v in verdicts
        if v.human_scores
    ]


def stratified_sample(
    items: Sequence[GoldenItem], size: int, *, seed: int = 0
) -> list[GoldenItem]:
    """Pick ``size`` questions spread across owners *and* sittings.

    Calibration measures whether the panel agrees with humans in general, so a
    sample drawn from one reviewer or one sitting would calibrate against that
    reviewer's habits rather than the rubric. Cells are visited round-robin so
    small strata are represented rather than swamped by large ones, and the
    order is seeded for reproducibility -- the same corpus yields the same
    sheet, which matters when a calibration result is challenged later.
    """
    cells: dict[tuple[str, str], list[GoldenItem]] = {}
    for item in items:
        owner = (item.owner or "(unassigned)").strip()
        sitting = item.reference.sitting_id if item.reference else "(no reference)"
        cells.setdefault((owner, sitting), []).append(item)

    rng = random.Random(seed)
    for bucket in cells.values():
        rng.shuffle(bucket)

    chosen: list[GoldenItem] = []
    order = sorted(cells)
    depth = 0
    while len(chosen) < size and any(len(cells[k]) > depth for k in order):
        for key in order:
            if depth < len(cells[key]):
                chosen.append(cells[key][depth])
                if len(chosen) >= size:
                    break
        depth += 1
    return chosen


def sample_coverage(items: Sequence[GoldenItem]) -> dict[str, Any]:
    """How a sample is spread, so a skewed one is visible before anyone labels it."""
    owners: dict[str, int] = {}
    sittings: dict[str, int] = {}
    for item in items:
        owners[(item.owner or "(unassigned)")] = owners.get(item.owner or "(unassigned)", 0) + 1
        key = item.reference.sitting_id if item.reference else "(no reference)"
        sittings[key] = sittings.get(key, 0) + 1
    return {"size": len(items), "owners": dict(sorted(owners.items())),
            "sittings": dict(sorted(sittings.items()))}


def inter_judge_agreement(verdicts: Sequence[PanelVerdict]) -> dict[str, Any]:
    """How much the judges agree with *each other*, independent of any human.

    Distinct from panel-human agreement and worth reading alongside it: judges
    that agree closely with each other but poorly with humans indicate a rubric
    problem shared by all three, which is the case the spec says to fix by
    tuning the prompt rather than swapping judges.
    """
    pairs: dict[str, dict[str, int]] = {}
    per_dimension: dict[str, dict[str, Any]] = {}

    for dimension in DIMENSIONS:
        exact = within1 = total = 0
        for v in verdicts:
            scored = [(j.model, j.scores[dimension]) for j in v.verdicts
                      if j.ok and dimension in j.scores]
            for i in range(len(scored)):
                for k in range(i + 1, len(scored)):
                    (m1, s1), (m2, s2) = scored[i], scored[k]
                    key = " vs ".join(sorted((m1, m2)))
                    cell = pairs.setdefault(key, {"compared": 0, "exact": 0, "within_1": 0})
                    cell["compared"] += 1
                    cell["exact"] += s1 == s2
                    cell["within_1"] += abs(s1 - s2) <= 1
                    total += 1
                    exact += s1 == s2
                    within1 += abs(s1 - s2) <= 1
        per_dimension[dimension] = {
            "pairs_compared": total,
            "exact_agreement": round(exact / total, 4) if total else None,
            "within_1": round(within1 / total, 4) if total else None,
        }

    for key, cell in pairs.items():
        n = cell.pop("compared")
        pairs[key] = {
            "compared": n,
            "exact_agreement": round(cell["exact"] / n, 4) if n else None,
            "within_1": round(cell["within_1"] / n, 4) if n else None,
        }

    flagged = sum(1 for v in verdicts if v.flagged_dimensions)
    return {
        "questions": len(verdicts),
        "flagged_questions": flagged,
        "flagged_rate": round(flagged / len(verdicts), 4) if verdicts else None,
        "per_dimension": per_dimension,
        "per_pair": dict(sorted(pairs.items())),
    }


def verdicts_from_sheet(sheet: dict[str, Any]) -> list[PanelVerdict]:
    """Rebuild panel verdicts from a calibration sheet.

    ``agreement_report`` takes ``PanelVerdict``s, and the only other source is a
    run store. A sheet built by ``calibrate-sample`` holds the same judgement --
    per-judge scores, the panel's aggregate, what was flagged -- but never
    passed through a run, so without this the numbers can only be recomputed by
    hand and the report is not reproducible.

    Only judges the sheet recorded are rebuilt: ``calibrate-sample`` stores
    ``judge_scores`` for panel members that answered, so a row judged 2/3 yields
    two verdicts rather than a third with invented scores.
    """
    out = []
    for row in sheet.get("rows", []):
        rationales = row.get("judge_rationales") or {}
        verdicts = [
            JudgeVerdict(model=model, scores=dict(scores),
                         rationale=str(rationales.get(model, "")))
            for model, scores in (row.get("judge_scores") or {}).items()
        ]
        out.append(PanelVerdict(
            question_id=str(row.get("question_id", "")),
            verdicts=verdicts,
            scores={k: v for k, v in (row.get("panel_scores") or {}).items()
                    if isinstance(v, int)},
            flagged_dimensions=list(row.get("flagged_dimensions") or []),
            error=row.get("adapter_error") or None,
        ))
    return out


def agreement_report(
    verdicts: Sequence[PanelVerdict], samples: Sequence[CalibrationSample]
) -> dict[str, Any]:
    """How well the panel matches the humans, per dimension and per judge.

    ``exact`` is agreement on the same 1-5 score; ``within_1`` is the more
    forgiving reading that matters in practice, since a 4-vs-5 disagreement
    rarely changes a go/no-go decision.
    """
    by_id = {v.question_id: v for v in verdicts}
    labelled = [s for s in samples if s.question_id in by_id]

    per_dimension: dict[str, dict[str, Any]] = {}
    per_judge: dict[str, dict[str, Any]] = {}

    for dimension in DIMENSIONS:
        exact = within1 = total = 0
        deltas: list[int] = []
        for sample in labelled:
            human = clamp(sample.human_scores.get(dimension))
            panel = by_id[sample.question_id].scores.get(dimension)
            if human is None or panel is None:
                continue
            total += 1
            delta = panel - human
            deltas.append(delta)
            exact += delta == 0
            within1 += abs(delta) <= 1
        per_dimension[dimension] = {
            "labelled": total,
            "exact_agreement": round(exact / total, 4) if total else None,
            "within_1": round(within1 / total, 4) if total else None,
            "mean_bias": round(sum(deltas) / len(deltas), 4) if deltas else None,
        }

    for sample in labelled:
        for judge in by_id[sample.question_id].verdicts:
            if not judge.ok:
                continue
            stats = per_judge.setdefault(
                judge.model, {"compared": 0, "exact": 0, "within_1": 0, "bias_sum": 0}
            )
            for dimension in DIMENSIONS:
                human = clamp(sample.human_scores.get(dimension))
                if human is None or dimension not in judge.scores:
                    continue
                delta = judge.scores[dimension] - human
                stats["compared"] += 1
                stats["exact"] += delta == 0
                stats["within_1"] += abs(delta) <= 1
                stats["bias_sum"] += delta

    for model, stats in per_judge.items():
        n = stats.pop("compared")
        bias = stats.pop("bias_sum")
        per_judge[model] = {
            "compared": n,
            "exact_agreement": round(stats["exact"] / n, 4) if n else None,
            "within_1": round(stats["within_1"] / n, 4) if n else None,
            "mean_bias": round(bias / n, 4) if n else None,
        }

    overall = [
        (d, per_dimension[d]["within_1"]) for d in DIMENSIONS
        if per_dimension[d]["within_1"] is not None
    ]
    headline = round(sum(v for _, v in overall) / len(overall), 4) if overall else None

    return {
        "labelled_samples": len(labelled),
        "calibration_set_size": len(samples),
        "panel_human_agreement": headline,
        "target": 0.8,
        "meets_target": (headline is not None and headline >= 0.8),
        "per_dimension": per_dimension,
        "per_judge": per_judge,
        "inter_judge": inter_judge_agreement([by_id[s.question_id] for s in labelled]),
    }
