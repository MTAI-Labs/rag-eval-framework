"""Calibration set: the human-labelled subset the panel is measured against.

Design spec §4.1 -- judges are calibrated on ~50 human-labelled samples before
the first full run, and every later human adjudication of a split row is
appended, so the calibration set grows as the framework is used and the panel's
agreement with humans becomes a tracked number rather than an assumption.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from rag_eval.judges.rubric import DIMENSIONS, PanelVerdict, clamp

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

    return {
        "labelled_samples": len(labelled),
        "calibration_set_size": len(samples),
        "per_dimension": per_dimension,
        "per_judge": per_judge,
    }
