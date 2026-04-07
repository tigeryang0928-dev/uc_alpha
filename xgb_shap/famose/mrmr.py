"""Greedy mRMR-style selection using mutual information and correlation redundancy."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression


def mrmr_select_features(
    X: pd.DataFrame, y: pd.Series, max_k: int | None, random_state: int = 42
) -> list[str]:
    cols = list(X.columns)
    if not cols:
        return []
    k = len(cols) if max_k is None else min(max_k, len(cols))
    if k <= 0:
        return []
    Xf = X.astype(np.float64).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    yv = y.to_numpy(dtype=np.float64)
    mi = mutual_info_regression(Xf, yv, random_state=random_state)
    rel = dict(zip(cols, mi, strict=True))
    selected: list[str] = []
    remaining = set(cols)
    while len(selected) < k and remaining:
        best_n: str | None = None
        best_s = -np.inf
        for name in remaining:
            r = rel[name]
            if selected:
                cors = []
                for s in selected:
                    a, b = Xf[name], Xf[s]
                    if float(a.std()) > 0 and float(b.std()) > 0:
                        c = a.corr(b)
                        if np.isfinite(c):
                            cors.append(abs(float(c)))
                red = float(np.mean(cors)) if cors else 0.0
            else:
                red = 0.0
            s = r - red
            if s > best_s:
                best_s = s
                best_n = name
        if best_n is None:
            break
        selected.append(best_n)
        remaining.remove(best_n)
    return selected
