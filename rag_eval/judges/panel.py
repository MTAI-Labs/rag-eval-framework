"""The judge panel: three models, one rubric, majority vote (design spec §5.2).

Three judges cost roughly 3x a single judge; the design accepted that for
robustness. What the panel buys is not a better average -- it is the
*disagreement signal*: rows where the panel splits are exactly the rows a human
should look at, and their adjudications feed the calibration set.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Iterable, Sequence

from rag_eval.config import JudgeConfig, JudgeModel
from rag_eval.judges.client import ChatClient
from rag_eval.judges.prompts import build_messages
from rag_eval.judges.rubric import (
    DIMENSIONS,
    JudgeVerdict,
    PanelVerdict,
    clamp,
    majority_vote,
)
from rag_eval.types import GoldenItem, RagTrace, Stopwatch

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


class JudgeParseError(ValueError):
    """The model did not return a usable rubric object."""


def parse_verdict(text: str, model: str) -> JudgeVerdict:
    """Parse a judge reply into a verdict.

    Tolerates a JSON object wrapped in prose or a ``json`` code fence -- not
    every gateway model honours ``response_format`` -- but not a missing
    dimension: a partial rubric is a retry, never a silent zero.
    """
    raw = (text or "").strip()
    candidate = raw
    if not candidate.startswith("{"):
        match = _JSON_BLOCK_RE.search(raw)
        if not match:
            raise JudgeParseError(f"{model}: no JSON object in reply: {raw[:200]!r}")
        candidate = match.group()

    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise JudgeParseError(f"{model}: invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise JudgeParseError(f"{model}: expected a JSON object, got {type(data).__name__}")

    # Accept both {"scores": {...}} and a flat {dimension: score} object.
    scores_raw = data.get("scores") if isinstance(data.get("scores"), dict) else data
    scores: dict[str, int] = {}
    missing: list[str] = []
    for dimension in DIMENSIONS:
        value = clamp(scores_raw.get(dimension))
        if value is None:
            missing.append(dimension)
        else:
            scores[dimension] = value
    if missing:
        raise JudgeParseError(f"{model}: missing or unscoreable dimension(s): {', '.join(missing)}")

    return JudgeVerdict(
        model=model,
        scores=scores,
        rationale=str(data.get("rationale", "") or "")[:2000],
        raw=raw[:4000],
    )


@dataclass
class PanelUsage:
    """Token and call accounting, so judging cost lands in the run manifest."""

    calls: int = 0
    failed_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0

    def to_dict(self) -> dict[str, float | int]:
        return {
            "calls": self.calls,
            "failed_calls": self.failed_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": round(self.cost_usd, 6),
        }


class JudgePanel:
    """Scores traces with a panel of models and aggregates by majority vote."""

    def __init__(
        self,
        client: ChatClient,
        config: JudgeConfig | None = None,
        *,
        max_workers: int = 3,
    ) -> None:
        self.client = client
        self.config = config or JudgeConfig()
        self.max_workers = max_workers
        self.usage = PanelUsage()

    # -- one question ------------------------------------------------------
    def judge_one(self, item: GoldenItem, trace: RagTrace) -> PanelVerdict:
        verdict = PanelVerdict(question_id=item.id)

        if not trace.ok:
            # A failed RAG call is a retrieval/ops failure, not something to ask
            # three models about; scoring it would burn judge budget on nothing.
            verdict.error = f"adapter error, not judged: {trace.error}"
            verdict.flagged_dimensions = list(DIMENSIONS)
            verdict.agreement = {d: "split" for d in DIMENSIONS}
            return verdict

        messages = build_messages(item, trace, context_top_k=self.config.context_top_k)

        if self.max_workers > 1 and len(self.config.models) > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                verdict.verdicts = list(
                    pool.map(lambda m: self._ask(m, messages), self.config.models)
                )
        else:
            verdict.verdicts = [self._ask(m, messages) for m in self.config.models]

        return self.aggregate(verdict)

    def aggregate(self, verdict: PanelVerdict) -> PanelVerdict:
        """Majority-vote each dimension and flag the ones that split."""
        usable = [v for v in verdict.verdicts if v.ok]
        if not usable:
            verdict.error = "every judge failed; see per-judge errors"
            verdict.flagged_dimensions = list(DIMENSIONS)
            verdict.agreement = {d: "split" for d in DIMENSIONS}
            return verdict

        for dimension in DIMENSIONS:
            values = [v.scores[dimension] for v in usable if dimension in v.scores]
            score, agreement = majority_vote(
                values, spread_threshold=self.config.flag_spread_threshold
            )
            if score is not None:
                verdict.scores[dimension] = score
            verdict.agreement[dimension] = agreement
            if agreement == "split":
                verdict.flagged_dimensions.append(dimension)

        if len(usable) < len(verdict.verdicts):
            verdict.error = (
                f"{len(verdict.verdicts) - len(usable)} of {len(verdict.verdicts)} judges failed; "
                f"aggregated over {len(usable)}"
            )
        return verdict

    # -- a whole run -------------------------------------------------------
    def judge_all(
        self,
        pairs: Iterable[tuple[GoldenItem, RagTrace]],
        *,
        on_result=None,
    ) -> list[PanelVerdict]:
        results: list[PanelVerdict] = []
        for item, trace in pairs:
            verdict = self.judge_one(item, trace)
            results.append(verdict)
            if on_result:
                on_result(verdict)
        return results

    # -- internals ---------------------------------------------------------
    def _ask(self, model: JudgeModel, messages: list[dict[str, str]]) -> JudgeVerdict:
        last_error = ""
        for attempt in range(1, self.config.max_retries + 1):
            self.usage.calls += 1
            try:
                with Stopwatch() as sw:
                    response = self.client.complete(
                        model.id,
                        messages,
                        temperature=model.temperature,
                        max_tokens=model.max_tokens,
                    )
                verdict = parse_verdict(response.text, model.id)
            except Exception as exc:  # noqa: BLE001 - a bad judge call is retried, not fatal
                self.usage.failed_calls += 1
                last_error = f"attempt {attempt}: {exc}"
                continue

            verdict.latency_ms = sw.elapsed_ms
            verdict.prompt_tokens = response.prompt_tokens
            verdict.completion_tokens = response.completion_tokens
            self._account(model, response.prompt_tokens, response.completion_tokens)
            return verdict

        return JudgeVerdict(model=model.id, error=last_error or "unknown judge failure")

    def _account(self, model: JudgeModel, prompt: int | None, completion: int | None) -> None:
        self.usage.prompt_tokens += prompt or 0
        self.usage.completion_tokens += completion or 0
        self.usage.cost_usd += (
            (prompt or 0) / 1000.0 * model.input_cost_per_1k
            + (completion or 0) / 1000.0 * model.output_cost_per_1k
        )


def flagged_rows(verdicts: Sequence[PanelVerdict]) -> list[PanelVerdict]:
    """The rows a human must adjudicate before the scorecard is trustworthy."""
    return [v for v in verdicts if v.needs_human_review and not v.adjudicated]
