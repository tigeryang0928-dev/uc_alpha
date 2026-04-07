"""Load training settings from YAML (see config.yaml)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
_XGB_SHAP_ROOT = Path(__file__).resolve().parent
DEFAULT_FAMOSE_CONFIG_PATH = _XGB_SHAP_ROOT / "famose" / "config.yaml"


def _resolve_path(p: str) -> str:
    path = Path(p)
    if path.is_absolute():
        return str(path)
    return str((_REPO_ROOT / path).resolve())


def _pick(d: dict[str, Any] | None, cls: type) -> dict[str, Any]:
    if not d:
        return {}
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in d.items() if k in names}


@dataclass
class DataConfig:
    handler_dir: str = "backtest_2/data"
    start_date: str | None = None


@dataclass
class StocksConfig:
    include: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)


@dataclass
class FiltersConfig:
    min_value_dollars: float = 20_000_000
    min_listing_days: int = 5
    use_value_filter: bool = True
    use_listing_day_filter: bool = True
    use_opening_limit_filter: bool = True
    use_day_trade_filter: bool = True
    use_non_limit_up_filter: bool = True


@dataclass
class SplitsConfig:
    train_frac: float = 0.7
    val_frac: float = 0.15


@dataclass
class OutputConfig:
    dir: str = "xgb_shap/output"


@dataclass
class ShapConfig:
    sample: int = 5000
    background_max: int = 2000
    conditional_correlation_threshold: float = 0.8
    conditional_min_slice_size: int = 30
    conditional_max_unique_for_filter: int = 12
    conditional_filter_columns: list[str] = field(default_factory=list)
    conditional_factor_columns: list[str] = field(default_factory=list)
    # Role recommendation (conditional_monotonic_factors.json)
    role_primary_abs_rho: float = 0.8
    role_noise_abs_rho: float = 0.5
    role_anchor_min_abs_rho_spread: float = 0.35
    role_high_shap_quantile: float = 0.5
    # Auto-discretization scan (all continuous factors; conditional_monotonic_factors.json)
    discretization_scan_strong_abs_rho: float = 0.8
    discretization_scan_quantiles: list[float] = field(
        default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    )
    # Each [q_lo, q_hi] with 0 < q_lo < q_hi < 1: scan Spearman inside anchor values between those quantiles.
    discretization_scan_band_pairs: list[list[float]] = field(default_factory=list)


@dataclass
class FamoseConfig:
    """FAMOSE-style DSL discovery; see xgb_shap/famose/."""

    max_rounds: int = 5
    max_steps_per_round: int = 6
    min_rel_improvement: float = 0.01
    early_stop_rounds_no_gain: int = 3
    run_subdir: str = "famose_runs"
    # openai: OpenAI-compatible POST {llm_base_url}/chat/completions + OPENAI_API_KEY
    # gemini: Google AI Studio REST v1beta + GEMINI_API_KEY (or GOOGLE_API_KEY)
    # cursor: see famose/cursor_cli.py — agent --print --force --model auto; optional env FAMOSE_CURSOR_CLI
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.8
    llm_base_url: str = "https://api.openai.com/v1"
    llm_cursor_cli: str = "cursor"
    llm_cursor_workspace: str | None = None
    llm_cursor_timeout_sec: float = 600.0
    llm_cursor_extra_args: list[str] = field(default_factory=list)
    mrmr_max_features: int | None = None
    date_sample_frac: float = 1.0


@dataclass
class TrainConfig:
    data: DataConfig = field(default_factory=DataConfig)
    stocks: StocksConfig = field(default_factory=StocksConfig)
    filters: FiltersConfig = field(default_factory=FiltersConfig)
    target: str = "intraday"
    splits: SplitsConfig = field(default_factory=SplitsConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    random_state: int = 42
    xgboost: dict[str, Any] = field(default_factory=dict)
    shap: ShapConfig = field(default_factory=ShapConfig)
    famose: FamoseConfig = field(default_factory=FamoseConfig)
    features: list[dict[str, str]] = field(default_factory=list)

    def resolved_handler_dir(self) -> str:
        return _resolve_path(self.data.handler_dir)

    def resolved_output_dir(self) -> str:
        return _resolve_path(self.output.dir)

    def feature_specs(self) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        for row in self.features:
            name = row.get("name")
            expr = row.get("expr")
            if not name or expr is None:
                continue
            out.append((str(name), str(expr)))
        return out


def load_famose_config(path: str | Path | None = None) -> FamoseConfig:
    """
    Load ``FamoseConfig`` from the FAMOSE-only YAML (default: ``xgb_shap/famose/config.yaml``).

    The file may be either a flat mapping of Famose fields or a single top-level ``famose:`` mapping.
    If the path is missing, returns ``FamoseConfig()`` defaults.
    """
    p = Path(path) if path is not None else DEFAULT_FAMOSE_CONFIG_PATH
    if not p.is_file():
        return FamoseConfig()
    with p.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    nested = raw.get("famose")
    if isinstance(nested, dict):
        famose_raw = nested
    else:
        famose_raw = raw
    return FamoseConfig(**_pick(famose_raw, FamoseConfig))


def load_train_config(
    path: str | Path,
    *,
    famose_config_path: str | Path | None = None,
) -> TrainConfig:
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    xgb_defaults = {
        "n_estimators": 800,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 50,
        "reg_lambda": 1.0,
        "n_jobs": -1,
        "early_stopping_rounds": 80,
    }
    xgb_user = raw.get("xgboost") or {}
    xgb_merged = {**xgb_defaults, **xgb_user}

    return TrainConfig(
        data=DataConfig(**_pick(raw.get("data"), DataConfig)),
        stocks=StocksConfig(**_pick(raw.get("stocks"), StocksConfig)),
        filters=FiltersConfig(**_pick(raw.get("filters"), FiltersConfig)),
        target=str(raw.get("target", "intraday")),
        splits=SplitsConfig(**_pick(raw.get("splits"), SplitsConfig)),
        output=OutputConfig(**_pick(raw.get("output"), OutputConfig)),
        random_state=int(raw.get("random_state", 42)),
        xgboost=xgb_merged,
        shap=ShapConfig(**_pick(raw.get("shap"), ShapConfig)),
        famose=load_famose_config(famose_config_path),
        features=list(raw.get("features") or []),
    )


def save_train_config_yaml(cfg: TrainConfig, path: str | Path) -> None:
    """Write the in-memory TrainConfig (after CLI overrides) as YAML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            asdict(cfg),
            f,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )
