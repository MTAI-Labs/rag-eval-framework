"""Generation metrics: does the RAG stack answer well?

Faithfulness, correctness and completeness come from the judge panel
(design spec §4.2); ``hallucination_rate`` is the fraction of answers whose
faithfulness falls below the configured threshold.

Human adjudications override panel scores wherever they exist, and the
aggregate always reports how much of it is still unadjudicated -- a mean that
hides 40 split rows is not evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from rag_eval.judges.rubric import DIMENSIONS, PanelVerdict
from rag_eval.types import RagTrace

JUDGED_DIMENSIONS = ("faithfulness", "correctness", "completeness")

#: Phrases that mark a model narrating its own instructions rather than
#: answering. A reasoning model with thinking left on emits its chain-of-thought
#: as the response; judges then score that near-perfect on faithfulness because
#: it contradicts nothing, so nothing downstream catches it. Cheap to detect and
#: worth catching before a run, not after.
_REASONING_MARKERS = (
    "we need", "need answer", "the user is asking", "the user asks",
    "user wants", "let me ", "i need to", "i should", "context only",
    "need follow instructions", "okay, ", "first, i", "let's ",
    "the question asks", "we must", "i'll ",
)


def looks_like_reasoning(answer: str, *, window: int = 300) -> bool:
    """Whether an answer is the model thinking out loud instead of answering.

    Checked against the opening of the text: a genuine answer may later discuss
    what a speaker said they needed to do, but it does not *begin* by narrating
    the task. Deliberately conservative -- a false positive would wrongly
    discredit a real answer.
    """
    head = (answer or "").strip().lower()[:window]
    return any(marker in head for marker in _REASONING_MARKERS)


def answer_quality(traces: Sequence[RagTrace]) -> dict[str, Any]:
    """How many answers are real answers, before anyone scores them.

    A RAG emitting reasoning is not a low-quality RAG -- it is a misconfigured
    one, and no judge score computed over it means anything.
    """
    answered = [t for t in traces if t.ok and (t.generated_answer or "").strip()]
    leaked = [t for t in answered if looks_like_reasoning(t.generated_answer)]
    return {
        "traces": len(traces),
        "answered": len(answered),
        "reasoning_leaked": len(leaked),
        "reasoning_leak_rate": (
            round(len(leaked) / len(answered), 4) if answered else None
        ),
        "examples": [t.question_id for t in leaked[:5]],
    }


@dataclass
class GenerationResult:
    question_id: str
    scores: dict[str, int]
    flagged: list[str]
    adjudicated: bool
    judged: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "scores": dict(self.scores),
            "flagged": list(self.flagged),
            "adjudicated": self.adjudicated,
            "judged": self.judged,
        }


def score_question(verdict: PanelVerdict) -> GenerationResult:
    return GenerationResult(
        question_id=verdict.question_id,
        scores=verdict.final_scores(),
        flagged=list(verdict.flagged_dimensions),
        adjudicated=verdict.adjudicated,
        judged=verdict.error is None or bool(verdict.scores),
    )


def _mean(values: Sequence[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def aggregate(
    verdicts: Sequence[PanelVerdict],
    traces: Sequence[RagTrace] | None = None,
    *,
    hallucination_threshold: int = 3,
    pass_threshold: int = 4,
) -> dict[str, Any]:
    """Aggregate panel verdicts into the scorecard's generation block."""
    results = [score_question(v) for v in verdicts]
    judged = [r for r in results if r.scores]

    metrics: dict[str, Any] = {
        "questions": len(results),
        "judged": len(judged),
        "unjudged": len(results) - len(judged),
    }

    for dimension in DIMENSIONS:
        values = [r.scores[dimension] for r in judged if dimension in r.scores]
        metrics[dimension] = _mean(values)
        metrics[f"{dimension}_pass_rate"] = (
            round(sum(1 for v in values if v >= pass_threshold) / len(values), 4)
            if values
            else None
        )

    faithfulness = [r.scores["faithfulness"] for r in judged if "faithfulness" in r.scores]
    metrics["hallucination_rate"] = (
        round(sum(1 for v in faithfulness if v < hallucination_threshold) / len(faithfulness), 4)
        if faithfulness
        else None
    )
    metrics["hallucination_threshold"] = hallucination_threshold
    metrics["pass_threshold"] = pass_threshold

    flagged = [r for r in results if r.flagged]
    metrics["panel_disagreement"] = {
        "flagged_questions": len(flagged),
        "flagged_rate": round(len(flagged) / len(results), 4) if results else None,
        "adjudicated": sum(1 for r in flagged if r.adjudicated),
        "awaiting_human_review": sum(1 for r in flagged if not r.adjudicated),
        "by_dimension": {
            d: sum(1 for r in results if d in r.flagged) for d in DIMENSIONS
        },
    }

    if traces is not None:
        answered = [t for t in traces if t.ok and t.generated_answer.strip()]
        metrics["empty_answer_rate"] = (
            round(1 - len(answered) / len(traces), 4) if traces else None
        )
        metrics["mean_answer_chars"] = _mean([len(t.generated_answer) for t in answered])

    return metrics


def per_judge_scores(verdicts: Sequence[PanelVerdict]) -> dict[str, dict[str, Any]]:
    """Each panel member's own mean per dimension, plus its failure rate.

    A judge drifting away from the panel shows up here first, which is the
    signal for re-calibration or for swapping a panel member in config.
    """
    stats: dict[str, dict[str, Any]] = {}
    for verdict in verdicts:
        for judge in verdict.verdicts:
            entry = stats.setdefault(
                judge.model, {"calls": 0, "failures": 0, "_sums": {}, "_counts": {}}
            )
            entry["calls"] += 1
            if not judge.ok:
                entry["failures"] += 1
                continue
            for dimension, score in judge.scores.items():
                entry["_sums"][dimension] = entry["_sums"].get(dimension, 0) + score
                entry["_counts"][dimension] = entry["_counts"].get(dimension, 0) + 1

    out: dict[str, dict[str, Any]] = {}
    for model, entry in stats.items():
        means = {
            d: round(entry["_sums"][d] / entry["_counts"][d], 4)
            for d in entry["_counts"]
        }
        out[model] = {
            "calls": entry["calls"],
            "failures": entry["failures"],
            "failure_rate": round(entry["failures"] / entry["calls"], 4) if entry["calls"] else None,
            "mean_scores": means,
        }
    return out
