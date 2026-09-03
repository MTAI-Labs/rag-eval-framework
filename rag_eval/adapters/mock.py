"""In-process adapter used by the contract tests and the CI end-to-end smoke.

Backed by a fixture file so a full pipeline run -- traces, judging (with a
scripted judge), metrics, scorecard -- needs no GPU and no network.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from rag_eval.adapters.base import RagAdapter
from rag_eval.types import RagTrace


class MockAdapter(RagAdapter):
    """Replays canned traces keyed by question id, or synthesises one.

    Options:
        ``fixture``      path to a JSONL of ``RagTrace`` dicts (keyed by ``question_id``)
        ``fail_ids``     question ids to return as errors, to exercise error-rate handling
        ``latency_ms``   fixed latency stamped on synthesised traces
        ``seed``         seed for the synthetic retrieval scores
    """

    name = "mock"

    def __init__(self, **options: Any) -> None:
        super().__init__(**options)
        self.traces: dict[str, dict[str, Any]] = {}
        fixture = options.get("fixture")
        if fixture:
            for line in Path(fixture).read_text(encoding="utf-8").splitlines():
                if line.strip():
                    raw = json.loads(line)
                    self.traces[str(raw["question_id"])] = raw
        self.by_question = {
            raw.get("question", ""): raw for raw in self.traces.values() if raw.get("question")
        }
        self.fail_ids = set(options.get("fail_ids", ()))
        self.latency_ms = float(options.get("latency_ms", 120.0))
        self._rng = random.Random(options.get("seed", 0))

    def answer(self, question: str, question_id: str = "") -> RagTrace:
        if question_id and question_id in self.fail_ids:
            trace = self._trace(question, question_id)
            trace.error = "mock adapter: scripted failure"
            trace.latency_ms = self.latency_ms
            return trace

        raw = self.traces.get(question_id) or self.by_question.get(question)
        if raw is not None:
            data = dict(raw)
            data.setdefault("adapter", self.name)
            data["question_id"] = question_id or data.get("question_id", "")
            data["question"] = question
            return RagTrace.from_dict(data)

        trace = self._trace(question, question_id)
        trace.generated_answer = f"[mock answer] {question}"
        trace.model = "mock-1"
        trace.latency_ms = self.latency_ms
        trace.prompt_tokens = 400
        trace.completion_tokens = 80
        trace.retrieved_chunks = [
            self.make_chunk(
                rank=i + 1,
                text=f"[mock chunk {i + 1}] {question}",
                chunk_id=f"mock-{i + 1}",
                score=round(1.0 - i * 0.1 + self._rng.random() * 1e-6, 6),
                sitting_id=None,
                page=None,
            )
            for i in range(self.top_k)
        ]
        return trace

    def health(self) -> tuple[bool, str]:
        return True, f"mock adapter, {len(self.traces)} canned trace(s)"
