"""Build Handler + panel stacks aligned with train_xgboost_shap."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent.parent
for _p in (_REPO / "xgb_shap", _REPO / "backtest_2"):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)

from method import funcs_methods  # noqa: E402
from panel_ml import row_date_masks, stack_panel  # noqa: E402
from train_config import TrainConfig  # noqa: E402
from train_xgboost_shap import (  # noqa: E402
    DEFAULT_FEATURE_SPECS,
    _apply_stock_universe,
    _build_returns_and_filters,
    _eval_features,
    _winsorize_panel_returns_masked,
)
from utils.data_bridge import Handler  # noqa: E402


@dataclass
class FamosePanelContext:
    handler: Handler
    exp_ret: pd.DataFrame
    universe: pd.DataFrame
    seed_specs: list[tuple[str, str]]
    funcs_methods: dict
    cfg: TrainConfig


def build_panel_context(cfg: TrainConfig) -> FamosePanelContext:
    data_dir = cfg.resolved_handler_dir()
    handler = Handler(data_dir)
    if cfg.data.start_date:
        handler.start_date = cfg.data.start_date

    adj_close = handler["Close"] * handler["Adjust_Factor"]
    adj_open = handler["Open"] * handler["Adjust_Factor"]
    targets = _build_returns_and_filters(handler, adj_close, adj_open, cfg)
    if cfg.target not in targets:
        raise ValueError(f"target must be one of {list(targets.keys())}")
    exp_ret, universe = targets[cfg.target]
    universe = _apply_stock_universe(universe, cfg)
    exp_ret = exp_ret.loc[:, universe.columns]
    if cfg.target == "winsorize_intraday":
        exp_ret = _winsorize_panel_returns_masked(exp_ret, universe, 0.01, 0.99)

    seed_specs = cfg.feature_specs() or list(DEFAULT_FEATURE_SPECS)
    return FamosePanelContext(
        handler=handler,
        exp_ret=exp_ret,
        universe=universe,
        seed_specs=seed_specs,
        funcs_methods=funcs_methods,
        cfg=cfg,
    )


def eval_specs(ctx: FamosePanelContext, specs: list[tuple[str, str]]) -> dict[str, pd.DataFrame]:
    feats = _eval_features(ctx.handler, specs)
    return {k: v.loc[:, ctx.universe.columns] for k, v in feats.items()}


def stack_from_specs(
    ctx: FamosePanelContext, specs: list[tuple[str, str]]
) -> tuple[pd.DataFrame, pd.Series]:
    feats = eval_specs(ctx, specs)
    return stack_panel(feats, ctx.exp_ret, mask=ctx.universe)


def train_val_masks(idx: pd.MultiIndex, cfg: TrainConfig) -> tuple[np.ndarray, np.ndarray]:
    tr_m, va_m, _ = row_date_masks(
        idx, train_frac=cfg.splits.train_frac, val_frac=cfg.splits.val_frac
    )
    return tr_m, va_m


def maybe_subsample_by_date(
    X: pd.DataFrame, y: pd.Series, frac: float, rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.Series]:
    if frac >= 1.0 or len(X) < 500:
        return X, y
    dates = X.index.get_level_values(0).unique()
    k = max(3, int(len(dates) * frac))
    pick = rng.choice(len(dates), size=k, replace=False)
    chosen = set(dates[np.sort(pick)])
    lev0 = X.index.get_level_values(0)
    m = np.array([d in chosen for d in lev0], dtype=bool)
    return X.loc[m], y.loc[m]
