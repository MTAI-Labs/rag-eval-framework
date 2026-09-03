"""``python -m rag_eval`` — same entry point as the ``rag-eval`` console script."""

from rag_eval.cli.main import main

if __name__ == "__main__":
    raise SystemExit(main())
