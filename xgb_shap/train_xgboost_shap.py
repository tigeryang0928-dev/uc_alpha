"""
Train XGBoost on backtest_2 Handler data + method.py feature expressions;
evaluate with held-out metrics and SHAP (TreeExplainer).

Settings: edit xgb_shap/config.yaml (stocks, filters, features, XGBoost params).

Entry point: run `python demo.py` from the repository root.
"""

from __future__ import annotations

import sys
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from scipy import stats

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PKG_ROOT = Path(__file__).resolve().parent
_BACKTEST_2 = _REPO_ROOT / "backtest_2"
for _p in (_PKG_ROOT, _BACKTEST_2):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)

from method import funcs_methods  # noqa: E402
from utils.data_bridge import Handler  # noqa: E402
from utils.evaluate import factors_analyze  # noqa: E402

from panel_ml import row_date_masks, stack_panel  # noqa: E402
from shap_conditional import (  # noqa: E402
    compute_conditional_monotonic_report,
    plot_discovered_monotonic_thresholds,
    plot_mean_signed_shap_bar,
    print_conditional_summary,
    save_conditional_monotonic_json,
)
from train_config import TrainConfig, save_train_config_yaml  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)

DEFAULT_FEATURE_SPECS: list[tuple[str, str]] = [
    ("cs_rank_turnover", "cs_rank(Turnover)"),
    ("cs_zscore_turnover", "cs_zscore(Turnover)"),
    ("ts_mean_close_5", "ts_mean(Close * Adjust_Factor, 5)"),
    ("ts_std_close_10", "ts_std(Close * Adjust_Factor, 10)"),
]


def _true_df(template: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(True, index=template.index, columns=template.columns)


def _eval_features(handler: Handler, specs: list[tuple[str, str]]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for name, expr in specs:
        out[name] = eval(expr, funcs_methods, handler)
    return out


def _build_returns_and_filters(
    handler: Handler,
    adj_close: pd.DataFrame,
    adj_open: pd.DataFrame,
    flt: TrainConfig,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    template = handler["Close"]
    f = flt.filters

    exp_overnight_ret = adj_open.shift(-2) / adj_close.shift(-1) - 1
    exp_intrday_ret = adj_close.shift(-1) / adj_open.shift(-1) - 1
    exp_day_ret = adj_open.shift(-2) / adj_open.shift(-1) - 1

    base_value = _true_df(template)
    if f.use_value_filter:
        base_value &= handler["Value_Dollars"] >= f.min_value_dollars

    listing = _true_df(template)
    if f.use_listing_day_filter and f.min_listing_days > 0:
        n = int(f.min_listing_days)
        listing = handler["Close"].notna().astype(int).rolling(n).sum() == n

    opening_ok = _true_df(template)
    if f.use_opening_limit_filter:
        opening_ok = ~(
            (handler["Limit_Up_or_Down_in_Opening_Fg"] == "Y") & (adj_open > adj_close.shift(1))
        )

    day_trade_ok = _true_df(template)
    if f.use_day_trade_filter:
        day_trade_ok = handler["Suspension_of_buy_After_Day_Trading_Fg"] != "Y"

    non_limit_up = _true_df(template)
    if f.use_non_limit_up_filter:
        non_limit_up = handler["Limit_Up_or_Down"] != "+"

    濾網 = listing & base_value & opening_ok.shift(-1)
    隔夜濾網 = base_value & non_limit_up.shift(-1)
    日內濾網 = base_value & day_trade_ok.shift(-1)

    return {
        "overnight": (exp_overnight_ret, 隔夜濾網),
        "intraday": (exp_intrday_ret, 日內濾網),
        "day": (exp_day_ret, 濾網),
    }


def _apply_stock_universe(universe: pd.DataFrame, cfg: TrainConfig) -> pd.DataFrame:
    """Restrict to include list (if set) and remove exclude list."""
    cols = universe.columns
    allowed = pd.Series(0, index=cols, dtype=np.int64)
    if cfg.stocks.include:
        for sym in cfg.stocks.include:
            if sym in allowed.index:
                allowed.loc[sym] = 1
        col_ok = allowed.astype(bool)
    else:
        col_ok = pd.Series(True, index=cols)
    for sym in cfg.stocks.exclude:
        if sym in col_ok.index:
            col_ok.loc[sym] = False
    return universe.loc[:, col_ok]


def _spearman_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 3:
        return float("nan")
    r, _ = stats.spearmanr(y_true[mask], y_pred[mask])
    return float(r)


def run(cfg: TrainConfig) -> Path:
    """
    Each run writes under ``output.dir / <YYYY-mm-dd_HH-MM-SS> /`` (see config).
    Writes a single ``config.yaml`` there: the effective settings after YAML + CLI overrides.
    """
    base_out = Path(cfg.resolved_output_dir())
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = base_out / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    save_train_config_yaml(cfg, out_dir / "config.yaml")

    print(f"Run output directory: {out_dir}")

    data_dir = cfg.resolved_handler_dir()
    handler = Handler(data_dir)
    if cfg.data.start_date:
        handler.start_date = cfg.data.start_date

    adj_close = handler["Close"] * handler["Adjust_Factor"]
    adj_open = handler["Open"] * handler["Adjust_Factor"]

    targets = _build_returns_and_filters(handler, adj_close, adj_open, cfg)
    target_name = cfg.target
    if target_name not in targets:
        raise ValueError(f"target must be one of {list(targets.keys())}, got {target_name!r}")
    exp_ret, universe = targets[target_name]
    universe = _apply_stock_universe(universe, cfg)

    feature_specs = cfg.feature_specs() or DEFAULT_FEATURE_SPECS
    features = _eval_features(handler, feature_specs)
    features = {k: v.loc[:, universe.columns] for k, v in features.items()}
    exp_ret = exp_ret.loc[:, universe.columns]

    X_df, y_series = stack_panel(features, exp_ret, mask=universe)

    if len(X_df) < 1_000:
        raise RuntimeError(f"too few rows after stacking: {len(X_df)}; check data paths and filters")

    idx = X_df.index
    assert isinstance(idx, pd.MultiIndex)
    tr_m, va_m, te_m = row_date_masks(
        idx, train_frac=cfg.splits.train_frac, val_frac=cfg.splits.val_frac
    )

    X_tr, y_tr = X_df.loc[tr_m], y_series.loc[tr_m]
    X_va, y_va = X_df.loc[va_m], y_series.loc[va_m]
    X_te, y_te = X_df.loc[te_m], y_series.loc[te_m]

    xgb_kw = dict(cfg.xgboost)
    xgb_kw["random_state"] = cfg.random_state
    model = xgb.XGBRegressor(**xgb_kw)
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False,
    )

    pred_all = model.predict(X_df)
    pred_all_s = pd.Series(pred_all, index=X_df.index)
    factor_full = pred_all_s.unstack()
    factor_full = factor_full.reindex(index=exp_ret.index, columns=exp_ret.columns)
    factor_full = factor_full.where(universe)
    exp_for_fa = exp_ret.where(universe)

    factors_analyze(
        factor_full,
        exp_for_fa,
        one_side=False,
        rank_range_n=10,
        quantile_metric="mean",
        name="XGBoost_score",
        output_dir=plots_dir,
        silent=True,
    )
    print(
        "Wrote factors_analyze: quantile_bar + cumulative (HTML; PNG if kaleido installed), "
        "quantile_stats.csv, stats_summary.csv"
    )

    pred_te = model.predict(X_te)
    rmse_te = float(np.sqrt(np.mean((pred_te - y_te.to_numpy()) ** 2)))
    ic_te = _spearman_ic(y_te.to_numpy(), pred_te)

    print(f"Test rows: {len(y_te)}")
    print(f"Test RMSE: {rmse_te:.6f}")
    print(f"Test Spearman IC (pred vs realized): {ic_te:.4f}")
    print("Feature importance (gain):")
    for name, imp in sorted(
        zip(X_df.columns, model.feature_importances_, strict=True),
        key=lambda x: -x[1],
    ):
        print(f"  {name}: {imp:.4f}")

    fi_path = out_dir / "xgb_feature_importance.csv"
    pd.DataFrame({"feature": X_df.columns, "importance": model.feature_importances_}).to_csv(
        str(fi_path), index=False
    )
    print(f"Wrote {fi_path}")

    rng = np.random.default_rng(cfg.random_state)
    n_bg = min(cfg.shap.background_max, len(X_tr))
    n_ex = min(cfg.shap.sample, len(X_te))
    bg_idx = rng.choice(len(X_tr), size=n_bg, replace=False)
    ex_idx = rng.choice(len(X_te), size=n_ex, replace=False)
    X_bg = X_tr.iloc[bg_idx]
    X_ex = X_te.iloc[ex_idx]

    explainer = shap.TreeExplainer(model, data=X_bg)
    shap_values_ex = explainer.shap_values(X_ex)
    shap_values_te = explainer.shap_values(X_te)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values_ex,
        X_ex,
        feature_names=list(X_ex.columns),
        show=False,
        plot_size=(10, 6),
    )
    p1 = plots_dir / "shap_summary_beeswarm.png"
    plt.tight_layout()
    plt.savefig(str(p1), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {p1}")

    p2 = plots_dir / "shap_summary_bar.png"
    plot_mean_signed_shap_bar(
        shap_values_ex,
        list(X_ex.columns),
        p2,
        figsize=(8, 5),
    )
    print(f"Wrote {p2}")

    mean_signed_ex = np.mean(shap_values_ex, axis=0)
    mean_abs_ex = np.mean(np.abs(shap_values_ex), axis=0)
    shap_df = pd.DataFrame(
        {
            "feature": X_ex.columns,
            "mean_shap": mean_signed_ex,
            "mean_abs_shap": mean_abs_ex,
        }
    ).sort_values("mean_abs_shap", ascending=False)
    p3 = out_dir / "shap_feature_summary.csv"
    shap_df.to_csv(str(p3), index=False)
    print(f"Wrote {p3}")

    sc = cfg.shap
    try:
        cond_report = compute_conditional_monotonic_report(
            X_te.reset_index(drop=True),
            np.asarray(shap_values_te),
            correlation_threshold=sc.conditional_correlation_threshold,
            min_slice_size=sc.conditional_min_slice_size,
            manual_filters=list(sc.conditional_filter_columns),
            manual_factors=list(sc.conditional_factor_columns),
            max_unique_for_filter=sc.conditional_max_unique_for_filter,
            role_primary_abs_rho=sc.role_primary_abs_rho,
            role_noise_abs_rho=sc.role_noise_abs_rho,
            role_anchor_min_abs_rho_spread=sc.role_anchor_min_abs_rho_spread,
            role_high_shap_quantile=sc.role_high_shap_quantile,
            discretization_scan_strong_abs_rho=sc.discretization_scan_strong_abs_rho,
            discretization_scan_quantiles=list(sc.discretization_scan_quantiles),
        )
    except ValueError as e:
        print(f"conditional monotonic analysis skipped: {e}")
        cond_report = {
            "meta": {"error": str(e)},
            "findings": [],
            "pair_analyses": [],
            "discovered_anchors": [],
        }
    cond_path = out_dir / "conditional_monotonic_factors.json"
    save_conditional_monotonic_json(cond_report, cond_path)
    print(f"Wrote {cond_path}")
    threshold_plot_path = plots_dir / "monotonic_threshold_map.png"
    if plot_discovered_monotonic_thresholds(cond_report, threshold_plot_path):
        print(f"Wrote {threshold_plot_path}")
    print_conditional_summary(cond_report)
    return out_dir
