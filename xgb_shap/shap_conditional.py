"""
Conditional slices: Spearman(raw feature, SHAP for that feature) per filter state.
See conditional_monotonic_factors.json output (findings + pair_analyses + recommended_role).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

ROLE_PRIMARY_FACTOR = "primary_factor"
ROLE_ANCHOR_FILTER = "anchor_filter"
ROLE_NOISE_COMPLEX = "noise_complex_model_only"
ROLE_UNCLASSIFIED = "unclassified"


def _state_key(v: Any) -> tuple:
    return (str(type(v).__name__), str(v))


def _json_safe_state(state: Any) -> Any:
    if isinstance(state, (bool, np.bool_)):
        return bool(state)
    if isinstance(state, (int, np.integer)):
        return int(state)
    if isinstance(state, (float, np.floating)):
        return float(state)
    return str(state)


def split_filter_vs_factor_columns(
    X: pd.DataFrame,
    manual_filters: list[str],
    manual_factors: list[str],
    max_unique_for_filter: int,
) -> tuple[list[str], list[str], dict[str, Any]]:
    """
    Classify columns into filter-like (low-cardinality) vs factor-like (continuous).
    Manual lists override auto for those names; remaining columns are auto-classified.
    """
    columns = list(X.columns)
    mf = set(manual_filters)
    mk = set(manual_factors)
    overlap = mf & mk
    if overlap:
        raise ValueError(f"conditional_filter_columns and conditional_factor_columns overlap: {overlap}")

    filters_auto: list[str] = []
    factors_auto: list[str] = []
    filters: list[str] = []
    factors: list[str] = []

    for c in columns:
        if c in mf:
            filters.append(c)
            continue
        if c in mk:
            factors.append(c)
            continue
        s = X[c].dropna()
        if len(s) == 0:
            factors.append(c)
            factors_auto.append(c)
            continue
        nu = int(s.nunique(dropna=True))
        if 2 <= nu <= max_unique_for_filter:
            filters.append(c)
            filters_auto.append(c)
        else:
            factors.append(c)
            factors_auto.append(c)

    meta = {
        "filters": filters,
        "factors": factors,
        "filters_auto_assigned": filters_auto,
        "factors_auto_assigned": factors_auto,
        "manual_filters": sorted(mf),
        "manual_factors": sorted(mk),
    }
    return filters, factors, meta


def _spearman_raw_vs_shap_slice(
    x_raw: np.ndarray,
    shap_col: np.ndarray,
    min_slice_size: int,
) -> tuple[float | None, int, str | None]:
    """
    Return (rho or None, n_valid, skip_reason).
    skip_reason is set when rho is None (insufficient data or degenerate).
    """
    valid = np.isfinite(x_raw) & np.isfinite(shap_col)
    nv = int(valid.sum())
    if nv < min_slice_size:
        return None, nv, "insufficient_n"
    x_v = x_raw[valid]
    s_v = shap_col[valid]
    if np.unique(x_v).size < 2 or np.unique(s_v).size < 2:
        return None, nv, "degenerate_values"
    rho, _ = stats.spearmanr(x_v, s_v)
    if rho is None or not np.isfinite(rho):
        return None, nv, "non_finite_rho"
    return float(rho), nv, None


def _recommend_role_for_pair(
    state_rows: list[dict[str, Any]],
    *,
    mean_abs_shap_factor: float,
    high_shap_cutoff: float,
    primary_abs_rho: float,
    noise_abs_rho: float,
    anchor_min_spread: float,
) -> tuple[str, str]:
    """
    Priority: primary_factor > noise_complex_model_only > anchor_filter > unclassified.
    Returns (role, short rationale).
    """
    evaluated = [r for r in state_rows if r.get("spearman_correlation") is not None]
    abs_rhos = [abs(float(r["spearman_correlation"])) for r in evaluated]

    if any(abs(float(r["spearman_correlation"])) > primary_abs_rho for r in evaluated):
        return ROLE_PRIMARY_FACTOR, f"|rho| > {primary_abs_rho} in at least one filter state"

    shap_high = mean_abs_shap_factor >= high_shap_cutoff
    if (
        shap_high
        and evaluated
        and all(abs(float(r["spearman_correlation"])) < noise_abs_rho for r in evaluated)
    ):
        return (
            ROLE_NOISE_COMPLEX,
            f"high global |SHAP| (>= {high_shap_cutoff:.6g}) but |rho| < {noise_abs_rho} in all evaluated states",
        )

    if len(abs_rhos) >= 2:
        spread = max(abs_rhos) - min(abs_rhos)
        if spread >= anchor_min_spread:
            return (
                ROLE_ANCHOR_FILTER,
                f"filter slices change monotonicity (spread of |rho| = {spread:.4f} >= {anchor_min_spread})",
            )

    return ROLE_UNCLASSIFIED, "does not meet primary, noise, or anchor criteria"


def _ordinal_percentile_label(q: float) -> str:
    p = int(round(float(q) * 100))
    p = max(0, min(100, p))
    if 10 <= (p % 100) <= 13:
        suffix = "th"
    elif p % 10 == 1:
        suffix = "st"
    elif p % 10 == 2:
        suffix = "nd"
    elif p % 10 == 3:
        suffix = "rd"
    else:
        suffix = "th"
    return f"{p}{suffix}"


def _global_raw_shap_self_spearman(
    x_col: np.ndarray,
    shap_col: np.ndarray,
    min_slice_size: int,
) -> float | None:
    """Spearman between raw feature and its own SHAP column on all finite rows."""
    return _spearman_raw_vs_shap_slice(
        np.asarray(x_col, dtype=np.float64),
        np.asarray(shap_col, dtype=np.float64),
        min_slice_size,
    )[0]


def scan_discretization_thresholds_for_feature(
    X: pd.DataFrame,
    shap_values: np.ndarray,
    *,
    col_list: list[str],
    factors: list[str],
    f_scan: str,
    min_slice_size: int,
    strong_abs_rho: float,
    quantiles: list[float],
) -> dict[str, Any]:
    """
    Try quantile cuts on ``f_scan`` (binary: value > threshold vs <=).
    In each slice, Spearman(raw_j, shap_j) for every other continuous factor j.
    Pick the cut that maximizes how many other factors reach |rho| > strong_abs_rho in at least one slice.
    """
    if f_scan not in col_list:
        return {"skipped": True, "reason": "feature_not_in_columns", "summary": None}

    other_factors = [j for j in factors if j != f_scan]
    if not other_factors:
        return {"skipped": True, "reason": "no_other_factors", "summary": None}

    x_scan = X[f_scan].to_numpy(dtype=np.float64, copy=False)
    finite_scan = np.isfinite(x_scan)
    if int(finite_scan.sum()) < min_slice_size * 2:
        return {"skipped": True, "reason": "insufficient_finite_scan_feature_rows", "summary": None}

    x_valid = x_scan[finite_scan]
    if np.unique(x_valid).size < 3:
        return {"skipped": True, "reason": "scan_feature_near_constant", "summary": None}

    best: dict[str, Any] | None = None
    best_key: tuple[int, float, float] = (-1, -1.0, -1.0)

    for q in quantiles:
        qf = float(q)
        if not (0.0 < qf < 1.0):
            continue
        t = float(np.quantile(x_valid, qf))
        mask_hi = finite_scan & (x_scan > t)
        mask_lo = finite_scan & (x_scan <= t)
        n_hi = int(mask_hi.sum())
        n_lo = int(mask_lo.sum())
        if n_hi < min_slice_size or n_lo < min_slice_size:
            continue

        strong_factors: list[str] = []
        detail_rows: list[dict[str, Any]] = []
        secondary = 0.0
        tertiary = 0.0

        for j in other_factors:
            strong_here = False
            j_idx = col_list.index(j)
            rho_hi, nv_hi, _ = _spearman_raw_vs_shap_slice(
                X.loc[mask_hi, j].to_numpy(dtype=np.float64, copy=False),
                shap_values[mask_hi, j_idx],
                min_slice_size,
            )
            rho_lo, nv_lo, _ = _spearman_raw_vs_shap_slice(
                X.loc[mask_lo, j].to_numpy(dtype=np.float64, copy=False),
                shap_values[mask_lo, j_idx],
                min_slice_size,
            )

            abs_hi = abs(float(rho_hi)) if rho_hi is not None else 0.0
            abs_lo = abs(float(rho_lo)) if rho_lo is not None else 0.0
            secondary += max(abs_hi, abs_lo)
            tertiary = max(tertiary, abs_hi, abs_lo)

            strong_here = False
            if rho_hi is not None and abs(float(rho_hi)) >= strong_abs_rho:
                strong_here = True
                detail_rows.append(
                    {
                        "other_factor": j,
                        "slice": "greater_than_threshold",
                        "spearman_correlation": round(float(rho_hi), 6),
                        "n_samples": nv_hi,
                    }
                )
            if rho_lo is not None and abs(float(rho_lo)) >= strong_abs_rho:
                strong_here = True
                detail_rows.append(
                    {
                        "other_factor": j,
                        "slice": "less_or_equal_threshold",
                        "spearman_correlation": round(float(rho_lo), 6),
                        "n_samples": nv_lo,
                    }
                )
            if strong_here and j not in strong_factors:
                strong_factors.append(j)

        n_strong = len(strong_factors)
        key = (n_strong, secondary, tertiary)
        if key > best_key:
            best_key = key
            best = {
                "quantile": round(qf, 6),
                "threshold_value": round(t, 8),
                "direction": "greater_than",
                "n_samples_greater_than": n_hi,
                "n_samples_less_or_equal": n_lo,
                "strong_abs_rho_threshold": strong_abs_rho,
                "strongly_monotonic_other_factor_count": n_strong,
                "strongly_monotonic_other_factors": list(strong_factors),
                "per_slice_details": list(detail_rows),
                "score_secondary_sum_abs_rho": round(secondary, 6),
            }

    if best is None:
        return {
            "skipped": True,
            "reason": "no_valid_quantile_cut_for_slice_sizes",
            "summary": None,
        }

    pct_label = _ordinal_percentile_label(best["quantile"])
    n_s = int(best["strongly_monotonic_other_factor_count"])
    tv = best["threshold_value"]
    if n_s == 0:
        summ = (
            f"Cut at {pct_label} percentile (value = {tv:.4g}). When > {tv:.4g}, "
            f"no other factors reached |rho| >= {strong_abs_rho}; "
            f"best cut by composite Spearman mass (sum max |rho| per factor = {best['score_secondary_sum_abs_rho']:.4f})."
        )
    else:
        summ = (
            f"Cut at {pct_label} percentile (value = {tv:.4g}). When > {tv:.4g}, "
            f"{n_s} other factor(s) become highly monotonic (|rho| >= {strong_abs_rho})."
        )
    best["summary"] = summ
    best["skipped"] = False
    return best


def _build_discretization_scan_results(
    X: pd.DataFrame,
    shap_values: np.ndarray,
    *,
    factors: list[str],
    col_list: list[str],
    min_slice_size: int,
    strong_abs_rho: float,
    quantiles: list[float],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """
    Scan every continuous factor as a candidate anchor, and collect all discoveries where
    another factor reaches |rho| >= strong_abs_rho in a binary threshold slice.
    """
    out: dict[str, dict[str, Any]] = {}
    discoveries: list[dict[str, Any]] = []
    for f_scan in sorted(set(factors)):
        if f_scan not in col_list:
            continue
        sidx = col_list.index(f_scan)
        g_rho = _global_raw_shap_self_spearman(
            X[f_scan].to_numpy(dtype=np.float64, copy=False),
            shap_values[:, sidx],
            min_slice_size,
        )
        scanned = scan_discretization_thresholds_for_feature(
            X,
            shap_values,
            col_list=col_list,
            factors=factors,
            f_scan=f_scan,
            min_slice_size=min_slice_size,
            strong_abs_rho=strong_abs_rho,
            quantiles=quantiles,
        )
        scanned["global_raw_shap_spearman"] = (
            round(float(g_rho), 6) if g_rho is not None else None
        )
        out[f_scan] = scanned
        if scanned.get("skipped"):
            continue
        pct_label = _ordinal_percentile_label(float(scanned["quantile"]))
        t_val = float(scanned["threshold_value"])
        for row in scanned.get("per_slice_details", []):
            other = str(row["other_factor"])
            rho = float(row["spearman_correlation"])
            slice_label = ">" if row.get("slice") == "greater_than_threshold" else "<="
            discoveries.append(
                {
                    "discovered_anchor": f_scan,
                    "threshold_quantile": round(float(scanned["quantile"]), 6),
                    "threshold_value": round(t_val, 8),
                    "threshold_condition": (
                        f"{slice_label} {pct_label} percentile (value = {t_val:.6g})"
                    ),
                    "activated_factor": other,
                    "spearman_rho": round(rho, 6),
                    "n_samples": int(row.get("n_samples", 0)),
                    "insight": (
                        f"When {f_scan} is {slice_label} {t_val:.6g}, "
                        f"{other} becomes highly monotonic (Spearman = {rho:.4f})."
                    ),
                }
            )

    discoveries.sort(key=lambda d: -abs(float(d["spearman_rho"])))
    return out, discoveries


def compute_conditional_monotonic_report(
    X: pd.DataFrame,
    shap_values: np.ndarray,
    *,
    correlation_threshold: float,
    min_slice_size: int,
    manual_filters: list[str],
    manual_factors: list[str],
    max_unique_for_filter: int,
    role_primary_abs_rho: float = 0.8,
    role_noise_abs_rho: float = 0.5,
    role_anchor_min_abs_rho_spread: float = 0.35,
    role_high_shap_quantile: float = 0.5,
    discretization_scan_strong_abs_rho: float = 0.8,
    discretization_scan_quantiles: list[float] | None = None,
) -> dict[str, Any]:
    """
    For each filter column and each observed state, slice rows and compute Spearman
    between raw factor values and SHAP values for each factor column (excluding the filter itself).

    Emits ``findings`` for slices with |rho| >= ``correlation_threshold`` (legacy / quick view)
    and ``pair_analyses`` with ``recommended_role`` per (filter, factor) pair.
    """
    if shap_values.ndim != 2:
        raise ValueError("shap_values must be 2D (n_samples, n_features)")
    if len(X) != len(shap_values):
        raise ValueError("X and shap_values row count mismatch")
    n_feat = shap_values.shape[1]
    if n_feat != X.shape[1]:
        raise ValueError("shap_values n_features must match X.columns")

    col_list = list(X.columns)
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    global_shap_map = {col_list[i]: float(mean_abs_shap[i]) for i in range(len(col_list))}
    q = float(np.clip(role_high_shap_quantile, 0.0, 1.0))
    high_shap_cutoff = float(np.quantile(mean_abs_shap, q))

    filters, factors, split_meta = split_filter_vs_factor_columns(
        X,
        manual_filters=manual_filters,
        manual_factors=manual_factors,
        max_unique_for_filter=max_unique_for_filter,
    )

    findings: list[dict[str, Any]] = []
    pair_analyses: list[dict[str, Any]] = []

    for fcol in filters:
        states = X[fcol].dropna().unique()
        states = sorted(states, key=_state_key)

        for factor_col in factors:
            if factor_col == fcol:
                continue
            fac_idx = col_list.index(factor_col)
            state_rows: list[dict[str, Any]] = []

            for state in states:
                mask = X[fcol].eq(state).to_numpy()
                n_mask = int(mask.sum())
                state_json = _json_safe_state(state)
                state_disp = str(state)

                row_base: dict[str, Any] = {
                    "filter_state": state_json,
                    "filter_state_display": state_disp,
                    "n_rows_in_state": n_mask,
                }

                if n_mask < min_slice_size:
                    row_base["spearman_correlation"] = None
                    row_base["n_samples"] = n_mask
                    row_base["skip_reason"] = "insufficient_n"
                    state_rows.append(row_base)
                    continue

                x_raw = X.loc[mask, factor_col].to_numpy(dtype=np.float64, copy=False)
                shap_col = shap_values[mask, fac_idx]
                rho, nv, skip = _spearman_raw_vs_shap_slice(x_raw, shap_col, min_slice_size)
                row_base["n_samples"] = nv
                if rho is None:
                    row_base["spearman_correlation"] = None
                    row_base["skip_reason"] = skip
                    state_rows.append(row_base)
                    continue

                row_base["spearman_correlation"] = round(float(rho), 6)
                row_base["skip_reason"] = None
                abs_r = abs(float(rho))
                row_base["direction"] = "positive" if rho > 0 else "negative"
                state_rows.append(row_base)

                if abs_r >= correlation_threshold:
                    direction = row_base["direction"]
                    interp = (
                        f"When [{fcol}] == {state_disp}, [{factor_col}] has a strong {direction} monotonic "
                        f"relationship (Spearman = {float(rho):.4f}). As {factor_col} increases, the model's "
                        f"SHAP contribution for {factor_col} tends to {'increase' if rho > 0 else 'decrease'} "
                        f"within this slice (n={nv})."
                    )
                    findings.append(
                        {
                            "filter_feature": fcol,
                            "filter_state": state_json,
                            "filter_state_display": state_disp,
                            "factor_feature": factor_col,
                            "spearman_correlation": round(float(rho), 6),
                            "n_samples": nv,
                            "direction": direction,
                            "interpretation": interp,
                        }
                    )

            mas_f = global_shap_map.get(factor_col, 0.0)
            role, rationale = _recommend_role_for_pair(
                state_rows,
                mean_abs_shap_factor=mas_f,
                high_shap_cutoff=high_shap_cutoff,
                primary_abs_rho=role_primary_abs_rho,
                noise_abs_rho=role_noise_abs_rho,
                anchor_min_spread=role_anchor_min_abs_rho_spread,
            )

            pair_analyses.append(
                {
                    "filter_feature": fcol,
                    "factor_feature": factor_col,
                    "recommended_role": role,
                    "role_rationale": rationale,
                    "global_mean_abs_shap_factor": round(mas_f, 8),
                    "high_shap_cutoff": round(high_shap_cutoff, 8),
                    "states": state_rows,
                }
            )

    # Deterministic sort: strongest |rho| first
    findings.sort(key=lambda d: -abs(d["spearman_correlation"]))
    pair_analyses.sort(key=lambda d: (d["filter_feature"], d["factor_feature"]))

    dq = discretization_scan_quantiles
    if not dq:
        dq = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    scan_map, discoveries = _build_discretization_scan_results(
        X,
        shap_values,
        factors=factors,
        col_list=col_list,
        min_slice_size=min_slice_size,
        strong_abs_rho=discretization_scan_strong_abs_rho,
        quantiles=list(dq),
    )
    for p in pair_analyses:
        fn = str(p["factor_feature"])
        p["scanner_summary"] = scan_map.get(fn)
    for p in pair_analyses:
        if p.get("recommended_role") == ROLE_NOISE_COMPLEX:
            fn = str(p["factor_feature"])
            p["recommended_filter_threshold"] = scan_map.get(fn)

    return {
        "meta": {
            "correlation_threshold": correlation_threshold,
            "min_slice_size": min_slice_size,
            "max_unique_for_filter": max_unique_for_filter,
            "role_primary_abs_rho": role_primary_abs_rho,
            "role_noise_abs_rho": role_noise_abs_rho,
            "role_anchor_min_abs_rho_spread": role_anchor_min_abs_rho_spread,
            "role_high_shap_quantile": role_high_shap_quantile,
            "discretization_scan_strong_abs_rho": discretization_scan_strong_abs_rho,
            "discretization_scan_quantiles": list(dq),
            "global_mean_abs_shap_by_feature": {
                k: round(v, 8) for k, v in sorted(global_shap_map.items())
            },
            **split_meta,
        },
        "findings": findings,
        "pair_analyses": pair_analyses,
        "discovered_anchors": discoveries,
        "discretization_scan_by_feature": scan_map,
    }


def save_conditional_monotonic_json(report: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def plot_discovered_monotonic_thresholds(
    report: dict[str, Any],
    out_path: str | Path,
    figsize: tuple[float, float] = (10, 6),
) -> bool:
    """
    Plot threshold value on X and activated monotonic feature on Y.
    Returns True if a plot file is written.
    """
    disc = list(report.get("discovered_anchors") or [])
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    if not disc:
        plt.figure(figsize=figsize)
        plt.text(
            0.5,
            0.5,
            "No monotonic discoveries found",
            ha="center",
            va="center",
            fontsize=12,
        )
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(str(path), dpi=150, bbox_inches="tight")
        plt.close()
        return True

    df = pd.DataFrame(disc)
    if "threshold_value" not in df.columns or "activated_factor" not in df.columns:
        return False

    df = df.copy()
    df = df[np.isfinite(df["threshold_value"].astype(float))]
    if df.empty:
        return False

    feats = sorted(df["activated_factor"].astype(str).unique())
    y_map = {f: i for i, f in enumerate(feats)}
    y = df["activated_factor"].astype(str).map(y_map).to_numpy()
    x = df["threshold_value"].astype(float).to_numpy()
    rho = df.get("spearman_rho", pd.Series([0.0] * len(df))).astype(float).to_numpy()
    size = 60 + 180 * np.clip(np.abs(rho), 0.0, 1.0)

    plt.figure(figsize=figsize)
    sc = plt.scatter(
        x,
        y,
        c=rho,
        s=size,
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        alpha=0.85,
        edgecolors="black",
        linewidths=0.3,
    )
    plt.yticks(np.arange(len(feats)), feats)
    plt.xlabel("Threshold value")
    plt.ylabel("Monotonic feature (activated_factor)")
    plt.title("Discovered monotonic feature activations by threshold")
    cbar = plt.colorbar(sc)
    cbar.set_label("Spearman rho")
    plt.tight_layout()
    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close()
    return True


def print_conditional_summary(report: dict[str, Any]) -> None:
    meta = report.get("meta") or {}
    if meta.get("error"):
        return
    filt = meta.get("filters") or []
    if not filt:
        print(
            "Conditional monotonic: no filter columns (every feature treated as continuous). "
            "Add shap.conditional_filter_columns in config.yaml for binary/categorical splits."
        )
    fs = report["findings"]
    print(f"Conditional monotonic analysis: {len(fs)} slice(s) with |Spearman| >= threshold.")
    for d in fs[:20]:
        print(f"  • {d['interpretation']}")
    if len(fs) > 20:
        print(f"  ... and {len(fs) - 20} more (see conditional_monotonic_factors.json)")

    pairs = report.get("pair_analyses") or []
    if pairs:
        ctr = Counter(p.get("recommended_role", ROLE_UNCLASSIFIED) for p in pairs)
        print("Role recommendation (per filter–factor pair):")
        for role, n in sorted(ctr.items(), key=lambda x: (-x[1], x[0])):
            print(f"  • {role}: {n}")
        for p in pairs:
            if p.get("recommended_role") != ROLE_NOISE_COMPLEX:
                continue
            th = p.get("recommended_filter_threshold") or {}
            summ = th.get("summary")
            if summ:
                print(f"  • Threshold scan [{p.get('filter_feature')} → {p.get('factor_feature')}]: {summ}")
    disc = report.get("discovered_anchors") or []
    if disc:
        print(f"Discretization scanner discoveries: {len(disc)}")
        for d in disc[:20]:
            print(f"  • {d['insight']}")


def plot_mean_signed_shap_bar(
    shap_values: np.ndarray,
    feature_names: list[str],
    out_path: str | Path,
    figsize: tuple[float, float] = (8, 5),
) -> None:
    """Horizontal bar chart of mean SHAP (signed), not mean |SHAP|."""
    mean_signed = np.mean(shap_values, axis=0)
    order = np.argsort(-np.abs(mean_signed))
    names_ord = [feature_names[i] for i in order]
    vals_ord = mean_signed[order]
    colors = np.where(vals_ord >= 0, "#4169E1", "#DC143C")

    import matplotlib.pyplot as plt

    plt.figure(figsize=figsize)
    y_pos = np.arange(len(names_ord))
    plt.barh(y_pos, vals_ord, color=colors, height=0.7)
    plt.yticks(y_pos, names_ord)
    plt.axvline(0, color="gray", linewidth=0.8)
    plt.xlabel("Mean SHAP value (signed)")
    plt.title("Mean SHAP by feature (positive = blue, negative = red)")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
