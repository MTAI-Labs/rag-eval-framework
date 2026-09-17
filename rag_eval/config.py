"""Configuration: judge model ids, adapter endpoints and cost tables.

Deliberately data, not code -- §5.2 of the design spec requires judge model ids
to live in config so swapping a panel member is a config review, not a diff.

Resolution order (later wins): built-in defaults -> config file (JSON) ->
environment variables -> CLI overrides.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "rag-eval.config.json"
DEFAULT_RUNS_DIR = REPO_ROOT / "runs"
DEFAULT_DATASETS_DIR = REPO_ROOT / "datasets"


@dataclass
class JudgeModel:
    """One member of the judge panel, served by the Thoth gateway."""

    id: str
    label: str = ""
    temperature: float = 0.0
    #: Generous because the panel models are *reasoning* models: they spend
    #: most of the budget on hidden reasoning before emitting the rubric. At
    #: 1024 a judge burns the entire allowance thinking and returns empty
    #: content with finish_reason="length" — which looks exactly like a model
    #: refusing to answer. Measured: ~1570 reasoning tokens for one rubric.
    max_tokens: int = 4096
    # USD per 1k tokens, used for the ops cost estimate.
    input_cost_per_1k: float = 0.0
    output_cost_per_1k: float = 0.0

    def __post_init__(self) -> None:
        self.label = self.label or self.id


@dataclass
class JudgeConfig:
    base_url_env: str = "THOTH_BASE_URL"
    api_key_env: str = "THOTH_API_KEY"
    #: The design spec's panel — GLM-5.2, Qwen3.5-397B, Kimi-K2.6 — under the
    #: ids the Thoth gateway actually serves. These are vendor-prefixed; the
    #: bare names ("glm-5.2") resolve to nothing and every call 404s, which only
    #: shows up when judging runs. Check against ${THOTH_BASE_URL}/v1/models
    #: before changing them.
    models: list[JudgeModel] = field(
        default_factory=lambda: [
            JudgeModel(id="zai-org/GLM-5.2", label="GLM-5.2"),
            JudgeModel(id="Qwen/Qwen3.5-397B-A17B", label="Qwen3.5-397B"),
            JudgeModel(id="moonshotai/Kimi-K2.6", label="Kimi-K2.6"),
        ]
    )
    # Context handed to the judge, in chunks.
    context_top_k: int = 5
    # A dimension is flagged for human adjudication when the panel produces no
    # majority, or when it agrees but spreads this far apart (e.g. 5/5/2).
    flag_spread_threshold: int = 2
    max_retries: int = 3
    timeout_s: float = 120.0

    @property
    def base_url(self) -> str:
        raw = os.environ.get(self.base_url_env, "").rstrip("/")
        if not raw:
            raise RuntimeError(
                f"{self.base_url_env} is not set -- the judge panel needs the Thoth gateway. "
                f"Copy .env.example and export it."
            )
        return raw if raw.endswith("/v1") else f"{raw}/v1"

    @property
    def api_key(self) -> str:
        return os.environ.get(self.api_key_env, "")


@dataclass
class AdapterConfig:
    """Free-form per-adapter settings; each adapter validates what it needs."""

    name: str
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricsConfig:
    # k values reported for hit_rate@k / recall@k / MRR.
    k_values: list[int] = field(default_factory=lambda: [1, 3, 5, 10])
    # faithfulness below this counts as a hallucination (spec §4.2).
    hallucination_threshold: int = 3
    # Judge scores at or above this count as "passing" in the scorecard.
    pass_threshold: int = 4


@dataclass
class Config:
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    adapters: dict[str, AdapterConfig] = field(default_factory=dict)
    runs_dir: Path = DEFAULT_RUNS_DIR
    datasets_dir: Path = DEFAULT_DATASETS_DIR

    def adapter_options(self, name: str) -> dict[str, Any]:
        cfg = self.adapters.get(name)
        return dict(cfg.options) if cfg else {}

    def judge_model(self, model_id: str) -> JudgeModel | None:
        return next((m for m in self.judge.models if m.id == model_id), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "judge": {
                "base_url_env": self.judge.base_url_env,
                "models": [asdict(m) for m in self.judge.models],
                "context_top_k": self.judge.context_top_k,
                "flag_spread_threshold": self.judge.flag_spread_threshold,
            },
            "metrics": asdict(self.metrics),
            "adapters": {k: asdict(v) for k, v in self.adapters.items()},
        }


def load_config(path: str | Path | None = None) -> Config:
    """Load config from JSON, falling back to built-in defaults.

    A missing file is not an error: the defaults are the approved design.
    """
    cfg = Config()
    candidate = Path(path) if path else DEFAULT_CONFIG_PATH
    if not candidate.exists():
        if path:
            raise FileNotFoundError(f"config file not found: {candidate}")
        return cfg

    raw = json.loads(candidate.read_text(encoding="utf-8"))

    judge_raw = raw.get("judge", {})
    if "models" in judge_raw:
        cfg.judge.models = [JudgeModel(**m) for m in judge_raw["models"]]
    for key in ("base_url_env", "api_key_env", "context_top_k",
                "flag_spread_threshold", "max_retries", "timeout_s"):
        if key in judge_raw:
            setattr(cfg.judge, key, judge_raw[key])

    for key, value in raw.get("metrics", {}).items():
        if hasattr(cfg.metrics, key):
            setattr(cfg.metrics, key, value)

    for name, options in raw.get("adapters", {}).items():
        opts = options.get("options", options) if isinstance(options, dict) else {}
        cfg.adapters[name] = AdapterConfig(name=name, options=opts)

    if "runs_dir" in raw:
        cfg.runs_dir = Path(raw["runs_dir"])
    if "datasets_dir" in raw:
        cfg.datasets_dir = Path(raw["datasets_dir"])
    return cfg


def load_dotenv(path: str | Path = REPO_ROOT / ".env") -> None:
    """Minimal .env loader so the CLI works without a python-dotenv dependency.

    Existing environment variables always win.
    """
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
