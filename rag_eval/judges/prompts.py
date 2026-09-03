"""Judge prompt construction.

One prompt, shared by all three panel members, so a disagreement is evidence
about the models rather than about the wording they each saw. Output is a
strict JSON object; anything else is a parse failure and gets retried.
"""

from __future__ import annotations

import json

from rag_eval.judges.rubric import DIMENSION_PROMPTS, DIMENSIONS
from rag_eval.types import GoldenItem, RagTrace

SYSTEM_PROMPT = """\
You are an impartial evaluator of a Malaysian parliamentary question-answering \
system. The corpus is Hansard (Dewan Rakyat, Dewan Negara and their Kamar Khas \
sittings); answers may be in Malay or English, and a Malay answer to an English \
question is not itself an error.

You score one answer at a time against a golden answer written by a human \
reviewer and against the context the system retrieved. You do not rewrite the \
answer, and you do not use knowledge outside the golden answer and the \
retrieved context: if the answer asserts something that neither source \
supports, that counts against faithfulness even if you believe it is true.

Reply with a single JSON object and nothing else."""

_SCHEMA_HINT = {
    "scores": {d: "<integer 1-5>" for d in DIMENSIONS},
    "rationale": "<one or two sentences, citing the specific fact that decided the lowest score>",
}


def build_user_prompt(item: GoldenItem, trace: RagTrace, *, context_top_k: int = 5) -> str:
    """Assemble the judged payload: question, golden answer, reference, context, answer."""
    reference = str(item.reference) if item.reference else "(no reference recorded)"
    cited = ", ".join(str(s) for s in trace.cited_sources) or "(none cited)"
    context = trace.context_text(context_top_k) or "(the system retrieved nothing)"

    rubric_lines = "\n".join(f"- {d}: {DIMENSION_PROMPTS[d]}" for d in DIMENSIONS)

    return f"""\
## Rubric — score each dimension from 1 (worst) to 5 (best)
{rubric_lines}

## Question
{item.question}

## Golden answer (written by a human reviewer)
{item.expected_answer}

## Golden reference (the Hansard sitting and page the answer came from)
{reference}

## Sources cited by the system
{cited}

## Context the system retrieved
{context}

## Answer produced by the system
{trace.generated_answer or "(the system produced no answer)"}

## Required output
{json.dumps(_SCHEMA_HINT, indent=2)}
"""


def build_messages(item: GoldenItem, trace: RagTrace, *, context_top_k: int = 5) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(item, trace, context_top_k=context_top_k)},
    ]
