"""HTML scorecard and run-over-run regression diff."""

from rag_eval.reporting.diff import MetricDelta, compare, per_question_changes
from rag_eval.reporting.html import render, write_report

__all__ = ["MetricDelta", "compare", "per_question_changes", "render", "write_report"]
