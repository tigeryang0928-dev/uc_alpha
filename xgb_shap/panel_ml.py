"""Panel (date × instrument) helpers for tabular ML."""

from __future__ import annotations

import numpy as np
import pandas as pd


def stack_panel(
    features: dict[str, pd.DataFrame],
    target: pd.DataFrame,
    mask: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Align feature matrices and target to the same index/columns, stack to long format,
    and drop rows with missing target, missing any feature, mask False, or non-finite values.

    Returns
    -------
    X : DataFrame
        One row per (date, instrument); columns are feature names.
    y : Series
        Target aligned to X.index (MultiIndex: date, instrument).
    """
    idx = target.index
    cols = target.columns
    y = target.reindex(index=idx, columns=cols)

    parts: dict[str, pd.Series] = {}
    for name, f in features.items():
        parts[name] = f.reindex(index=idx, columns=cols).stack()

    X_long = pd.DataFrame(parts)
    y_long = y.stack()
    if mask is not None:
        m_long = mask.reindex(index=idx, columns=cols).stack().astype(bool)
    else:
        m_long = pd.Series(True, index=y_long.index)

    m_long = m_long.reindex(y_long.index).fillna(False)
    comb = X_long.join(y_long.rename("__target__")).join(m_long.rename("__mask__"), how="inner")
    comb = comb[comb["__mask__"]].drop(columns=["__mask__"])
    comb = comb.replace([np.inf, -np.inf], np.nan).dropna()
    y_out = comb["__target__"].astype(np.float64)
    X_out = comb.drop(columns=["__target__"]).astype(np.float64)
    return X_out, y_out


def row_date_masks(
    index: pd.MultiIndex, train_frac: float, val_frac: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split unique level-0 dates into train / validation / test by contiguous blocks
    (chronological, no shuffling).
    """
    if not isinstance(index, pd.MultiIndex) or index.nlevels < 1:
        raise ValueError("index must be a MultiIndex with date as first level")
    dates = index.get_level_values(0).unique().sort_values()
    n = len(dates)
    if n < 3:
        raise ValueError("need at least 3 distinct dates for train/val/test split")
    i1 = max(1, int(n * train_frac))
    i2 = max(i1 + 1, int(n * (train_frac + val_frac)))
    if i2 >= n:
        i2 = n - 1
    train_d = set(dates[:i1])
    val_d = set(dates[i1:i2])
    test_d = set(dates[i2:])
    lev0 = index.get_level_values(0)
    return (
        np.array([d in train_d for d in lev0], dtype=bool),
        np.array([d in val_d for d in lev0], dtype=bool),
        np.array([d in test_d for d in lev0], dtype=bool),
    )
