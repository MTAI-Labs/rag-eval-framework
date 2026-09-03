"""Reusable evaluation framework for RAG services.

Every RAG service plugs in through a thin adapter; the harness measures
retrieval, generation and ops quality so that a change to a RAG stack can be
judged with numbers rather than vibes.
"""

__version__ = "0.1.0"
