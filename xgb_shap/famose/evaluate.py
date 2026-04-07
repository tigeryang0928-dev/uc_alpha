"""XGBoost validation RMSE: model with vs without a candidate feature."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb


def val_rmse(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_va: pd.DataFrame,
    y_va: pd.Series,
    xgb_kw: dict,
    random_state: int,
) -> float:
    kw = {**xgb_kw, "random_state": random_state}
    model = xgb.XGBRegressor(**kw)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    pred = model.predict(X_va)
    e = pred - y_va.to_numpy()
    return float(np.sqrt(np.mean(e**2)))


def relative_improvement(rmse_with: float, rmse_without: float) -> float:
    if not np.isfinite(rmse_without) or rmse_without <= 0:
        return 0.0
    return (rmse_without - rmse_with) / rmse_without


def score_candidate(
    X_tr_base: pd.DataFrame,
    y_tr: pd.Series,
    X_va_base: pd.DataFrame,
    y_va: pd.Series,
    cand_name: str,
    cand_tr: pd.Series,
    cand_va: pd.Series,
    xgb_kw: dict,
    random_state: int,
) -> tuple[float, float, float]:
    """Returns (rmse_with, rmse_without, rel_improve)."""
    Xw_tr = X_tr_base.copy()
    Xw_tr[cand_name] = cand_tr
    Xw_va = X_va_base.copy()
    Xw_va[cand_name] = cand_va
    rmse_with = val_rmse(Xw_tr, y_tr, Xw_va, y_va, xgb_kw, random_state)
    rmse_without = val_rmse(X_tr_base, y_tr, X_va_base, y_va, xgb_kw, random_state + 1)
    rel = relative_improvement(rmse_with, rmse_without)
    return rmse_with, rmse_without, rel
