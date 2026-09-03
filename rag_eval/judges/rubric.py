"""The judge rubric (design spec §4.1) and the verdict records it produces.

Four dimensions, each scored 1-5 by every panel member:

    faithfulness       is every claim supported by the retrieved context?
    correctness        does it agree with the golden answer (facts, numbers, names)?
    completeness       does it cover what the golden answer covers?
    citation_accuracy  does the cited sitting/page match the Excel reference?

``citation_accuracy`` is also computed deterministically in
``rag_eval.metrics.retrieval``; the judge's opinion of it is kept as a
cross-check on the deterministic metric, never as a replacement for it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

SCALE_MIN, SCALE_MAX = 1, 5

DIMENSIONS: tuple[str, ...] = (
    "faithfulness",
    "correctness",
    "completeness",
    "citation_accuracy",
)

DIMENSION_PROMPTS: dict[str, str] = {
    "faithfulness": (
        "Is every factual claim in the answer supported by the retrieved context? "
        "5 = every claim is directly supported; 3 = mostly supported with one "
        "unsupported detail; 1 = substantially fabricated or contradicted by the context."
    ),
    "correctness": (
        "Does the answer agree with the golden answer on facts, numbers, names and "
        "dates? 5 = agrees on every checkable fact; 3 = right in substance but wrong "
        "or missing on a number/name; 1 = contradicts the golden answer."
    ),
    "completeness": (
        "Does the answer cover the parts of the question that the golden answer "
        "covers? 5 = covers everything; 3 = covers the main point but omits "
        "secondary parts; 1 = leaves the question essentially unanswered."
    ),
    "citation_accuracy": (
        "Do the sources cited by the answer point at the sitting and page given in "
        "the golden reference? 5 = exact sitting and page; 3 = right sitting, wrong "
        "page; 1 = wrong sitting or no usable citation."
    ),
}


@dataclass
class JudgeVerdict:
    """One panel member's scores for one (question, answer) pair."""

    model: str
    scores: dict[str, int] = field(default_factory=dict)
    rationale: str = ""
    error: str | None = None
    latency_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    raw: str = ""

    @property
    def ok(self) -> bool:
        return self.error is None and all(d in self.scores for d in DIMENSIONS)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "scores": dict(self.scores),
            "rationale": self.rationale,
            "error": self.error,
            "latency_ms": self.latency_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "JudgeVerdict":
        return cls(
            model=d["model"],
            scores={k: int(v) for k, v in (d.get("scores") or {}).items()},
            rationale=d.get("rationale", ""),
            error=d.get("error"),
            latency_ms=d.get("latency_ms"),
            prompt_tokens=d.get("prompt_tokens"),
            completion_tokens=d.get("completion_tokens"),
        )


@dataclass
class PanelVerdict:
    """The panel's aggregated judgement for one question."""

    question_id: str
    verdicts: list[JudgeVerdict] = field(default_factory=list)
    scores: dict[str, int] = field(default_factory=dict)          # final, per dimension
    flagged_dimensions: list[str] = field(default_factory=list)   # need human adjudication
    agreement: dict[str, str] = field(default_factory=dict)       # unanimous|majority|split
    human_scores: dict[str, int] = field(default_factory=dict)    # filled by adjudication
    error: str | None = None

    @property
    def needs_human_review(self) -> bool:
        return bool(self.flagged_dimensions) or self.error is not None

    @property
    def adjudicated(self) -> bool:
        return bool(self.human_scores)

    def final_scores(self) -> dict[str, int]:
        """Human adjudication overrides the panel wherever it exists."""
        return {**self.scores, **self.human_scores}

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "scores": dict(self.scores),
            "final_scores": self.final_scores(),
            "flagged_dimensions": list(self.flagged_dimensions),
            "agreement": dict(self.agreement),
            "human_scores": dict(self.human_scores),
            "needs_human_review": self.needs_human_review,
            "error": self.error,
            "verdicts": [v.to_dict() for v in self.verdicts],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PanelVerdict":
        return cls(
            question_id=str(d["question_id"]),
            verdicts=[JudgeVerdict.from_dict(v) for v in d.get("verdicts", [])],
            scores={k: int(v) for k, v in (d.get("scores") or {}).items()},
            flagged_dimensions=list(d.get("flagged_dimensions", [])),
            agreement=dict(d.get("agreement", {})),
            human_scores={k: int(v) for k, v in (d.get("human_scores") or {}).items()},
            error=d.get("error"),
        )


def clamp(score: Any) -> int | None:
    """Coerce a judge's raw score onto the 1-5 integer scale."""
    try:
        value = int(round(float(score)))
    except (TypeError, ValueError):
        return None
    return max(SCALE_MIN, min(SCALE_MAX, value))


def majority_vote(values: Sequence[int], *, spread_threshold: int = 2) -> tuple[int | None, str]:
    """Aggregate one dimension across the panel.

    Returns ``(score, agreement)`` where agreement is:

    ``unanimous``  every judge gave the same score
    ``majority``   at least two agree and the panel's spread is tolerable
    ``split``      no two judges agree, or a majority exists but a dissenter is
                   ``spread_threshold`` or more away (e.g. 5/5/2) -- either way
                   the row is flagged for human adjudication

    A split still returns the median as a provisional score so the scorecard is
    computable; the flag is what tells a human to go look.
    """
    scores = [s for s in values if s is not None]
    if not scores:
        return None, "split"
    if len(set(scores)) == 1:
        return scores[0], "unanimous"

    counts: dict[int, int] = {}
    for s in scores:
        counts[s] = counts.get(s, 0) + 1
    best = max(counts.items(), key=lambda kv: (kv[1], kv[0]))
    spread = max(scores) - min(scores)

    if best[1] >= 2 and spread < spread_threshold:
        return best[0], "majority"
    if best[1] >= 2:
        return best[0], "split"

    ordered = sorted(scores)
    median = ordered[len(ordered) // 2]
    return median, "split"
