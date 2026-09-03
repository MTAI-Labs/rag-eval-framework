"""LLM-as-judge panel: rubric, prompts, gateway client, majority vote, calibration."""

from rag_eval.judges.rubric import (
    DIMENSIONS,
    DIMENSION_PROMPTS,
    JudgeVerdict,
    PanelVerdict,
    majority_vote,
)
from rag_eval.judges.panel import JudgePanel, JudgeParseError, PanelUsage, flagged_rows, parse_verdict
from rag_eval.judges.client import ChatClient, ChatResponse, ScriptedChatClient, ThothChatClient, build_client
from rag_eval.judges.calibration import (
    CalibrationSample,
    agreement_report,
    append_calibration,
    load_calibration,
    samples_from_adjudications,
)

__all__ = [
    "DIMENSIONS",
    "DIMENSION_PROMPTS",
    "JudgeVerdict",
    "PanelVerdict",
    "majority_vote",
    "JudgePanel",
    "JudgeParseError",
    "PanelUsage",
    "flagged_rows",
    "parse_verdict",
    "ChatClient",
    "ChatResponse",
    "ScriptedChatClient",
    "ThothChatClient",
    "build_client",
    "CalibrationSample",
    "agreement_report",
    "append_calibration",
    "load_calibration",
    "samples_from_adjudications",
]
