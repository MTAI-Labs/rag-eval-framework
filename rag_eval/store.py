"""Append-only run artifacts under ``runs/``.

Every stage of the pipeline reads and writes files in one run directory, which
is what makes the stages composable (``run`` -> ``judge`` -> ``score`` ->
``report``) and what makes a score reproducible months later:

    runs/<run_id>/
        manifest.json     config, adapter description, judge model ids, dataset checksum
        traces.jsonl      one RagTrace per question
        judgments.jsonl   one PanelVerdict per question
        scorecard.json    computed metrics
        scorecard.html    the human-readable report

Nothing here overwrites a completed run: a re-run gets a new id. The manifest
is the provenance record, so it captures *what was true at run time*, including
the judge model ids and the golden-set checksum.
"""

from __future__ import annotations

import json
import os
import platform
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from rag_eval import __version__
from rag_eval.judges.rubric import PanelVerdict
from rag_eval.types import RagTrace

RUN_ID_RE = re.compile(r"^\d{8}-\d{6}-[a-z0-9-]+$")


def new_run_id(adapter: str, now: datetime | None = None) -> str:
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%d-%H%M%S")
    slug = re.sub(r"[^a-z0-9-]+", "-", adapter.lower()).strip("-") or "run"
    return f"{stamp}-{slug}"


def git_revision() -> str:
    """The framework revision that produced a run, when available."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=Path(__file__).resolve().parent.parent,
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - environment dependent
        return ""


@dataclass
class RunManifest:
    run_id: str
    adapter: str
    created_at: str = ""
    completed_at: str = ""
    dataset: str = ""
    dataset_sha256: str = ""
    dataset_count: int = 0
    questions_run: int = 0
    adapter_config: dict[str, Any] = field(default_factory=dict)
    judge_models: list[str] = field(default_factory=list)
    judge_usage: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    framework_version: str = __version__
    git_revision: str = field(default_factory=git_revision)
    environment: dict[str, str] = field(
        default_factory=lambda: {"python": platform.python_version(), "platform": platform.platform()}
    )
    stages: dict[str, str] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RunManifest":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


class RunStore:
    """Filesystem access to one run directory."""

    def __init__(self, run_dir: str | Path) -> None:
        self.dir = Path(run_dir)
        self.run_id = self.dir.name

    # -- construction ------------------------------------------------------
    @classmethod
    def create(cls, runs_dir: str | Path, adapter: str) -> "RunStore":
        """Claim a fresh run directory.

        Run ids are second-granular, so two runs started in the same second get
        a numeric suffix rather than one silently appending to the other's
        traces. ``exist_ok=False`` is what makes the claim atomic.
        """
        base = new_run_id(adapter)
        for attempt in range(1, 100):
            run_id = base if attempt == 1 else f"{base}-{attempt}"
            store = cls(Path(runs_dir) / run_id)
            try:
                store.dir.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                continue
            return store
        raise RuntimeError(f"cannot allocate a run directory under {runs_dir}")

    @classmethod
    def open(cls, runs_dir: str | Path, run_id: str) -> "RunStore":
        path = Path(runs_dir) / run_id
        if not path.is_dir():
            raise FileNotFoundError(f"no such run: {run_id} (looked in {runs_dir})")
        return cls(path)

    # -- paths -------------------------------------------------------------
    @property
    def manifest_path(self) -> Path:
        return self.dir / "manifest.json"

    @property
    def traces_path(self) -> Path:
        return self.dir / "traces.jsonl"

    @property
    def judgments_path(self) -> Path:
        return self.dir / "judgments.jsonl"

    @property
    def scorecard_path(self) -> Path:
        return self.dir / "scorecard.json"

    @property
    def report_path(self) -> Path:
        return self.dir / "scorecard.html"

    # -- manifest ----------------------------------------------------------
    def write_manifest(self, manifest: RunManifest) -> None:
        self.manifest_path.write_text(
            json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    def read_manifest(self) -> RunManifest:
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"run {self.run_id} has no manifest.json")
        return RunManifest.from_dict(json.loads(self.manifest_path.read_text(encoding="utf-8")))

    def stamp_stage(self, stage: str) -> None:
        """Record when a pipeline stage completed, so a half-run run is obvious."""
        manifest = self.read_manifest()
        manifest.stages[stage] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        manifest.completed_at = manifest.stages[stage]
        self.write_manifest(manifest)

    # -- traces ------------------------------------------------------------
    def append_trace(self, trace: RagTrace) -> None:
        with open(self.traces_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def write_traces(self, traces: Iterable[RagTrace]) -> None:
        with open(self.traces_path, "w", encoding="utf-8") as fh:
            for trace in traces:
                fh.write(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n")

    def read_traces(self) -> list[RagTrace]:
        return [RagTrace.from_dict(d) for d in _read_jsonl(self.traces_path)]

    # -- judgments ---------------------------------------------------------
    def append_judgment(self, verdict: PanelVerdict) -> None:
        with open(self.judgments_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(verdict.to_dict(), ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def write_judgments(self, verdicts: Iterable[PanelVerdict]) -> None:
        with open(self.judgments_path, "w", encoding="utf-8") as fh:
            for verdict in verdicts:
                fh.write(json.dumps(verdict.to_dict(), ensure_ascii=False) + "\n")

    def read_judgments(self) -> list[PanelVerdict]:
        return [PanelVerdict.from_dict(d) for d in _read_jsonl(self.judgments_path)]

    # -- scorecard ---------------------------------------------------------
    def write_scorecard(self, scorecard: dict[str, Any]) -> None:
        self.scorecard_path.write_text(
            json.dumps(scorecard, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    def read_scorecard(self) -> dict[str, Any]:
        if not self.scorecard_path.exists():
            raise FileNotFoundError(f"run {self.run_id} has no scorecard.json; run 'rag-eval score'")
        return json.loads(self.scorecard_path.read_text(encoding="utf-8"))


def list_runs(runs_dir: str | Path, adapter: str | None = None) -> list[RunStore]:
    """All runs, oldest first. Run ids sort chronologically by construction."""
    root = Path(runs_dir)
    if not root.is_dir():
        return []
    stores = [
        RunStore(p)
        for p in sorted(root.iterdir())
        if p.is_dir() and RUN_ID_RE.match(p.name)
    ]
    if adapter:
        pattern = re.compile(rf"^\d{{8}}-\d{{6}}-{re.escape(adapter)}(-\d+)?$")
        stores = [s for s in stores if pattern.match(s.run_id)]
    return stores


def previous_run(
    runs_dir: str | Path, adapter: str, before: str | None = None
) -> RunStore | None:
    """The last scored run for the same adapter -- the regression baseline.

    Comparing across adapters would be comparing two different products, so the
    baseline is always same-adapter.
    """
    candidates = [s for s in list_runs(runs_dir, adapter) if s.scorecard_path.exists()]
    if before:
        candidates = [s for s in candidates if s.run_id < before]
    return candidates[-1] if candidates else None


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return iter(())

    def gen() -> Iterator[dict[str, Any]]:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)

    return gen()
