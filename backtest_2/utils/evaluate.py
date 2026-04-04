import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from IPython.display import display
import quantstats as qs

# quantstats>=0.0.81 no longer calls extend_pandas() on import; Series lacks .cagr() until this runs.
qs.extend_pandas()


# factor 權重標準化
def Factor_to_Weight(factor:pd.DataFrame, only_long:bool=False):
    row_mean = factor.mean(axis=1)
    demeaned = factor.sub(row_mean, axis=0)
    denom = demeaned.abs().sum(axis=1).replace(0, np.nan)
    weights = demeaned.div(denom, axis=0)
    weights = weights.replace([np.inf, -np.inf], 0).fillna(0)
    if only_long:
        weights[weights<0] = 0
        weights *= 2
    return weights


# 計算單利最大回撤
def max_drawdown(prices):
    cumulative_max = prices.cummax()
    drawdown = (prices - cumulative_max) / cumulative_max
    mdd = drawdown.min()
    return mdd


def _build_group_masks(factor_for_group, group_mode='quantile', rank_range_n=10, custom_groups=None):
    """
    建立分組遮罩與分組標籤。

    Args:
        factor_for_group: 用來分組的因子 DataFrame
        group_mode: 'quantile' 或 'custom'
        rank_range_n: quantile 分組數量
        custom_groups: 自定義分組規則，支援以下格式
            1) tuple: (label, lower, upper)
            2) dict : {
                'label': str,
                'lower': float or None,
                'upper': float or None,
                'include_lower': bool (default True),
                'include_upper': bool (default False),
            }

    Returns:
        masks: list[pd.DataFrame[bool]]
        labels: list[str]
        range_labels: list[str]（顯示用）
    """
    if group_mode not in ('quantile', 'custom'):
        raise ValueError("group_mode 需為 'quantile' 或 'custom'")

    def _fmt(v):
        if v == 0:
            return '0'
        elif abs(v) < 1e-3:
            return f'{v:.2e}'
        elif abs(v) >= 1000:
            return f'{v:.0f}'
        elif abs(v) >= 1:
            return f'{v:.3f}'
        else:
            return f'{v:.4f}'

    if group_mode == 'quantile':
        factor_rank = factor_for_group.rank(axis=1, pct=True)
        masks = []
        labels = []
        for i in range(rank_range_n):
            labels.append(f'Q{i+1}')
            masks.append((factor_rank > i / rank_range_n) & (factor_rank <= (i + 1) / rank_range_n))

        factor_stacked = factor_for_group.stack().dropna()
        boundaries = np.nanpercentile(factor_stacked.values, np.linspace(0, 100, rank_range_n + 1))
        range_labels = [f'({_fmt(boundaries[i])}, {_fmt(boundaries[i+1])}]' for i in range(rank_range_n)]
        return masks, labels, range_labels

    if not custom_groups:
        raise ValueError("group_mode='custom' 時，custom_groups 不可為空")

    masks = []
    labels = []
    for group in custom_groups:
        if isinstance(group, dict):
            label = group['label']
            lower = group.get('lower', None)
            upper = group.get('upper', None)
            include_lower = group.get('include_lower', True)
            include_upper = group.get('include_upper', False)
        else:
            if len(group) != 3:
                raise ValueError("custom_groups tuple 格式需為 (label, lower, upper)")
            label, lower, upper = group
            include_lower = True
            include_upper = False

        mask = pd.DataFrame(True, index=factor_for_group.index, columns=factor_for_group.columns)
        if lower is not None:
            mask &= (factor_for_group >= lower) if include_lower else (factor_for_group > lower)
        if upper is not None:
            mask &= (factor_for_group <= upper) if include_upper else (factor_for_group < upper)

        labels.append(str(label))
        masks.append(mask)

    range_labels = labels.copy()
    return masks, labels, range_labels


def _save_plotly_figure(fig: go.Figure, out_dir: Path, basename: str) -> None:
    """Write interactive HTML; PNG if kaleido is available."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_dir / f"{basename}.html"))
    try:
        fig.write_image(str(out_dir / f"{basename}.png"), width=900, height=450, scale=2)
    except Exception:
        pass


# 因子分析
def factors_analyze(
    factor,
    exp_ret,
    one_side=False,
    rank_range_n=10,
    quantile_metric='cagr',
    name='Factor',
    group_mode='quantile',
    custom_groups=None,
    group_use_adjusted=None,
    output_dir=None,
    silent=False,
):
    """
    回測單一因子並輸出統計圖表
    
    Args:
        factor: 因子 DataFrame
        exp_ret: 預期收益率 DataFrame
        one_side: 是否單邊交易
        rank_range_n: Quantile 分組數量（group_mode='quantile' 時使用）
        quantile_metric: 分組比較圖使用的指標，可選 'cagr'、'mean'、'median'、'std'
        name: 因子名稱（顯示用）
        group_mode: 分組方式，'quantile'（預設，等樣本）或 'custom'
        custom_groups: 自定義分組規則，格式見 _build_group_masks docstring
        group_use_adjusted: 分組是否使用方向調整後因子。
            None 時：quantile 預設 True，custom 預設 False
        output_dir: 若設定，將分組表、統計表匯出 CSV，圖表匯出 HTML（及 kaleido 時 PNG）到此目錄
        silent: True 時不 print / 不在 notebook display（仍寫檔若 output_dir 有給）
    
    Returns:
        results: 包含回測結果的字典
        stats_df: 統計指標 DataFrame
    """
    _out: Path | None = Path(output_dir) if output_dir is not None else None

    # 計算因子權重
    factorWeight = Factor_to_Weight(factor, one_side)
    ret_positive = (factorWeight * exp_ret).sum(axis=1)
    ret_negative = (-factorWeight * exp_ret).sum(axis=1)
    
    # 判斷多空方向：哪個方向的累積報酬為正就選哪個
    if ret_positive.sum() > ret_negative.sum():
        ret = ret_positive
        sign = 1
    else:
        ret = ret_negative
        sign = -1
    
    # 自動過濾開頭的全 0 數據
    first_nonzero_idx = (ret != 0).idxmax() if (ret != 0).any() else ret.index[0]
    ret = ret[ret.index >= first_nonzero_idx]
    factor_filtered = factor[factor.index >= first_nonzero_idx]
    exp_ret_filtered = exp_ret[exp_ret.index >= first_nonzero_idx]
    
    # 計算分組收益
    factor_adjusted = factor_filtered * sign
    if group_use_adjusted is None:
        group_use_adjusted = (group_mode == 'quantile')
    factor_for_group = factor_adjusted if group_use_adjusted else factor_filtered

    group_masks, group_labels, range_labels = _build_group_masks(
        factor_for_group=factor_for_group,
        group_mode=group_mode,
        rank_range_n=rank_range_n,
        custom_groups=custom_groups,
    )

    benchmark_ret = exp_ret_filtered.mean(axis=1)
    quantile_ret_dict = {}
    for label, mask in zip(group_labels, group_masks):
        q_ret = exp_ret_filtered[mask].mean(axis=1) - benchmark_ret
        quantile_ret_dict[label] = q_ret

    quantile_ret = pd.concat(quantile_ret_dict, axis=1)

    # 計算各 quantile 的統計指標（平均、中位數、標準差）
    all_exp_ret_flat = exp_ret_filtered.stack().dropna()
    total_count = len(all_exp_ret_flat)
    quantile_stats_rows = []
    quantile_raw_stats = []
    for i, mask in enumerate(group_masks):
        q_returns_raw = exp_ret_filtered[mask].stack().dropna()
        mean_bps   = round(q_returns_raw.mean()   * 10000, 4)
        median_bps = round(q_returns_raw.median() * 10000, 4)
        std_bps    = round(q_returns_raw.std()    * 10000, 4)
        quantile_raw_stats.append({'mean_bps': mean_bps, 'median_bps': median_bps, 'std_bps': std_bps})
        quantile_stats_rows.append({
            f'X 範: {name}': range_labels[i],
            '計數': len(q_returns_raw),
            '比例(%)': round(len(q_returns_raw) / total_count * 100, 2),
            '平均(bps)': mean_bps,
            '中位數(bps)': median_bps,
            '標準差(bps)': std_bps
        })
    
    # nan 組：分組依據為 nan 的股票，只顯示計數與比例，不顯示報酬統計
    nan_mask = factor_for_group.isna()
    nan_ret_raw = exp_ret_filtered[nan_mask].stack().dropna()
    nan_count = len(nan_ret_raw)
    quantile_stats_rows.append({
        f'X 範: {name}': 'nan',
        '計數': nan_count,
        '比例(%)': round(nan_count / total_count * 100, 2) if total_count > 0 else 0,
        '平均(bps)': np.nan,
        '中位數(bps)': np.nan,
        '標準差(bps)': np.nan
    })
    
    # All Cases：只計算有分組依據值（非 nan）的股票對應的 exp_ret
    valid_mask = factor_for_group.notna()
    valid_exp_ret_flat = exp_ret_filtered[valid_mask].stack().dropna()
    valid_count = len(valid_exp_ret_flat)
    quantile_stats_rows.append({
        f'X 範: {name}': 'All Cases',
        '計數': valid_count,
        '比例(%)': round(valid_count / total_count * 100, 2) if total_count > 0 else 0,
        '平均(bps)': round(valid_exp_ret_flat.mean() * 10000, 4),
        '中位數(bps)': round(valid_exp_ret_flat.median() * 10000, 4),
        '標準差(bps)': round(valid_exp_ret_flat.std() * 10000, 4)
    })
    quantile_stats_df = pd.DataFrame(quantile_stats_rows).set_index(f'X 範: {name}')
    
    # 計算 IC（使用方向調整後的因子）；跨截面常數因子列會觸發 scipy ConstantInputWarning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        IC_se = factor_adjusted.corrwith(
            exp_ret_filtered, method="spearman", axis=1
        ).sort_index()
    
    # 計算統計指標
    stats = {}
    stats['CAGR(%)'] = round(ret.cagr() * 100, 2)
    stats['Sharpe'] = round((ret.mean() / ret.std() * (252 ** 0.5)), 2)
    stats['MDD(%)'] = round(ret.max_drawdown() * 100, 2)
    stats['單利MDD(%)'] = round(max_drawdown(ret.cumsum().add(1)) * 100, 2)
    stats['IC'] = round(IC_se.mean(), 4)
    stats['ICIR'] = round(IC_se.mean() / IC_se.std(), 4) if IC_se.std() != 0 else 0
    stats['樣本勝率(%)'] = round((lambda X:((X.dropna()>0).sum() / X.dropna().shape[0])*100)(ret), 2)
    stats['周勝率(%)'] = round((lambda X:((X.dropna().add(1).resample('W').prod().sub(1)>0).sum() / X.dropna().add(1).resample('W').prod().sub(1).shape[0])*100)(ret), 2)
    stats['月勝率(%)'] = round((lambda X:((X.dropna().add(1).resample('ME').prod().sub(1)>0).sum() / X.dropna().add(1).resample('ME').prod().sub(1).shape[0])*100)(ret), 2)
    stats['年勝率(%)'] = round((lambda X:((X.dropna().add(1).resample('YE').prod().sub(1)>0).sum() / X.dropna().add(1).resample('YE').prod().sub(1).shape[0])*100)(ret), 2)
    stats['盈虧比'] = round((lambda X:(X[X>0].mean() / abs(X[X<0].mean())))(ret), 2)
    stats['總賺賠比'] = round(ret.profit_factor(), 2)
    stats['預期報酬(bps)'] = round(ret.expected_return() * 10000, 2)
    stats['樣本數'] = round((lambda X:X.dropna().count())(ret), 2)
    
    results = {
        'return': ret,
        'cum_return': ret.cumsum(),
        'quantile_ret': quantile_ret,
        'quantile_stats': quantile_stats_df,
        'IC': IC_se,
        'sign': sign,
        'stats': stats
    }
    
    # ========== 繪製圖表 ==========
    
    # 1. Quantile 統計表
    if not silent:
        print(f"1. 分組統計指標表（bps） — {name} (方向: {'+' if sign == 1 else '-'})...")
    if _out is None:
        display(quantile_stats_df)
    else:
        _out.mkdir(parents=True, exist_ok=True)
        quantile_stats_df.to_csv(_out / "factors_analyze_quantile_stats.csv", encoding="utf-8-sig")
    
    # 2. Quantile 比較圖
    _metric_map = {
        'cagr':   ('年化收益率 (%)',   lambda: [quantile_ret[q].mean() * 252 * 100 for q in quantile_ret.columns]),
        'mean':   ('平均報酬 (bps)',   lambda: [s['mean_bps']   for s in quantile_raw_stats]),
        'median': ('中位數報酬 (bps)', lambda: [s['median_bps'] for s in quantile_raw_stats]),
        'std':    ('標準差 (bps)',     lambda: [s['std_bps']    for s in quantile_raw_stats]),
    }
    assert quantile_metric in _metric_map, f"quantile_metric 需為 {list(_metric_map.keys())} 之一"
    _ylabel, _metric_fn = _metric_map[quantile_metric]
    if not silent:
        print(f"\n2. 繪製分組 {quantile_metric} 比較圖...")
    fig_bar = go.Figure(go.Bar(x=range_labels, y=_metric_fn(), name=f"{name} ({sign})"))
    fig_bar.update_layout(
        title=f'{name} 分組 {quantile_metric} 比較',
        xaxis_title='分組',
        yaxis_title=_ylabel,
        template='seaborn',
        height=450
    )
    if _out is None:
        fig_bar.show()
    else:
        _save_plotly_figure(fig_bar, _out, "factors_analyze_quantile_bar")
    
    # 3. 累積收益曲線圖
    if not silent:
        print("3. 繪製累積收益曲線圖...")
    fig_cum = go.Figure(go.Scatter(
        x=results['cum_return'].index,
        y=results['cum_return'].values,
        mode='lines',
        name=f"{name} ({sign})"
    ))
    fig_cum.update_layout(
        title=f'{name} 累積收益曲線',
        xaxis_title='日期',
        yaxis_title='累積收益',
        template='seaborn',
        height=450
    )
    if _out is None:
        fig_cum.show()
    else:
        _save_plotly_figure(fig_cum, _out, "factors_analyze_cumulative")
    
    # 4. 統計指標表
    stats_df = pd.DataFrame([stats], index=[name]).round(4)
    if not silent:
        print("統計指標:")
    if _out is None:
        display(stats_df)
    else:
        stats_df.to_csv(_out / "factors_analyze_stats_summary.csv", encoding="utf-8-sig")
    
    return results, stats_df

 
# 多因子比較回測函式
def factors_compare(
    factors_dict,
    exp_ret,
    quantile_metric='cagr',
    rank_range_n=10,
    group_mode='quantile',
    custom_groups=None,
    group_use_adjusted=None,
):
    """
    同時回測多個因子並將結果放在同一張圖中比較。
    
    Args:
        factors_dict: 字典，key 為因子名稱，value 為因子 DataFrame
        exp_ret     : 預期收益率 DataFrame
        quantile_metric : 分組比較圖指標，可選 'cagr' / 'mean' / 'median' / 'std'
        rank_range_n: Quantile 分組數（group_mode='quantile' 時使用）
        group_mode: 分組方式，'quantile'（預設，等樣本）或 'custom'
        custom_groups: 自定義分組規則，格式見 _build_group_masks docstring
        group_use_adjusted: 分組是否使用方向調整後因子。
            None 時：quantile 預設 True，custom 預設 False
    
    Returns:
        results  : 包含所有因子回測結果的字典
        stats_df : 統計指標 DataFrame
    """
    _metric_map = {
        'cagr':   '年化收益率 (%)',
        'mean':   '平均報酬 (bps)',
        'median': '中位數報酬 (bps)',
        'std':    '標準差 (bps)',
    }
    assert quantile_metric in _metric_map, f"quantile_metric 需為 {list(_metric_map.keys())} 之一"

    if group_use_adjusted is None:
        group_use_adjusted = (group_mode == 'quantile')

    if group_mode == 'quantile':
        q_labels = [f'Q{i+1}' for i in range(rank_range_n)]
    else:
        if not custom_groups:
            raise ValueError("group_mode='custom' 時，custom_groups 不可為空")
        q_labels = [str(g['label']) if isinstance(g, dict) else str(g[0]) for g in custom_groups]

    results = {}
    all_meta = []   # 儲存每個因子的 quantile 原始統計，供繪圖用

    # ── 第一步：計算每個因子的收益和統計指標 ────────────────────────────────
    for name, factor in factors_dict.items():

        # 計算因子權重
        factorWeight = Factor_to_Weight(factor)
        ret_positive = (factorWeight * exp_ret).sum(axis=1)
        ret_negative = (-factorWeight * exp_ret).sum(axis=1)

        # 判斷多空方向：哪個方向的累積報酬為正就選哪個
        if ret_positive.sum() > ret_negative.sum():
            ret = ret_positive
            sign = 1
        else:
            ret = ret_negative
            sign = -1

        # 自動過濾開頭的全 0 數據
        first_nonzero_idx = (ret != 0).idxmax() if (ret != 0).any() else ret.index[0]
        ret = ret[ret.index >= first_nonzero_idx]
        factor_filtered  = factor[factor.index >= first_nonzero_idx]
        exp_ret_filtered = exp_ret[exp_ret.index >= first_nonzero_idx]

        # 計算分組收益
        factor_adjusted  = factor_filtered * sign
        factor_for_group = factor_adjusted if group_use_adjusted else factor_filtered
        group_masks, group_labels, _ = _build_group_masks(
            factor_for_group=factor_for_group,
            group_mode=group_mode,
            rank_range_n=rank_range_n,
            custom_groups=custom_groups,
        )
        benchmark_ret    = exp_ret_filtered.mean(axis=1)

        quantile_ret_dict  = {}
        quantile_raw_stats = []
        for label, mask in zip(group_labels, group_masks):
            q_ret_daily  = exp_ret_filtered[mask].mean(axis=1) - benchmark_ret
            q_ret_raw    = exp_ret_filtered[mask].stack().dropna()
            quantile_ret_dict[label] = q_ret_daily
            quantile_raw_stats.append({
                'mean_bps':   round(q_ret_raw.mean()   * 10000, 4),
                'median_bps': round(q_ret_raw.median() * 10000, 4),
                'std_bps':    round(q_ret_raw.std()    * 10000, 4),
            })

        quantile_ret = pd.concat(quantile_ret_dict, axis=1)

        # 計算 IC（跨截面常數列會觸發 spearman 常數輸入警告）
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            IC_se = factor_adjusted.corrwith(
                exp_ret_filtered, method="spearman", axis=1
            ).sort_index()

        # 計算統計指標
        stats = {}
        stats['CAGR(%)']        = round(ret.cagr() * 100, 2)
        stats['Sharpe']         = round(ret.mean() / ret.std() * (252 ** 0.5), 2)
        stats['MDD(%)']         = round(ret.max_drawdown() * 100, 2)
        stats['IC']             = round(IC_se.mean(), 4)
        stats['ICIR']           = round(IC_se.mean() / IC_se.std(), 4) if IC_se.std() != 0 else 0
        stats['樣本勝率(%)']    = round(((ret.dropna() > 0).sum() / ret.dropna().shape[0]) * 100, 2)
        stats['周勝率(%)']      = round(((ret.dropna().add(1).resample('W').prod().sub(1) > 0).sum() /
                                          ret.dropna().add(1).resample('W').prod().sub(1).shape[0]) * 100, 2)
        stats['月勝率(%)']      = round(((ret.dropna().add(1).resample('ME').prod().sub(1) > 0).sum() /
                                          ret.dropna().add(1).resample('ME').prod().sub(1).shape[0]) * 100, 2)
        stats['年勝率(%)']      = round(((ret.dropna().add(1).resample('YE').prod().sub(1) > 0).sum() /
                                          ret.dropna().add(1).resample('YE').prod().sub(1).shape[0]) * 100, 2)
        stats['盈虧比']         = round(ret[ret > 0].mean() / abs(ret[ret < 0].mean()), 2)
        stats['總賺賠比']       = round(ret.profit_factor(), 2)
        stats['預期報酬(bps)']  = round(ret.expected_return() * 10000, 2)
        stats['樣本數']         = int(ret.dropna().count())

        results[name] = {
            'return':       ret,
            'cum_return':   ret.cumsum(),
            'quantile_ret': quantile_ret,
            'IC':           IC_se,
            'cum_IC':       IC_se.cumsum(),
            'sign':         sign,
            'stats':        stats,
        }
        all_meta.append({
            'name':               name,
            'sign':               sign,
            'quantile_ret':       quantile_ret,
            'quantile_raw_stats': quantile_raw_stats,
        })

    # ── 繪製圖表 ─────────────────────────────────────────────────────────────

    # 1. Quantile 比較圖（grouped bar）
    ylabel = _metric_map[quantile_metric]

    def _get_bar_values(meta):
        qrs = meta['quantile_raw_stats']
        qr  = meta['quantile_ret']
        if quantile_metric == 'cagr':
            return [qr[col].mean() * 252 * 100 for col in qr.columns]
        elif quantile_metric == 'mean':
            return [s['mean_bps']   for s in qrs]
        elif quantile_metric == 'median':
            return [s['median_bps'] for s in qrs]
        else:  # std
            return [s['std_bps']    for s in qrs]

    fig_bar = go.Figure()
    for meta in all_meta:
        label = f"{meta['name']} ({'+' if meta['sign']==1 else '-'})"
        fig_bar.add_trace(go.Bar(
            name=label,
            x=q_labels,
            y=_get_bar_values(meta),
        ))
    fig_bar.update_layout(
        title=f'所有因子分組 {quantile_metric} 比較',
        xaxis_title='分組',
        yaxis_title=ylabel,
        barmode='group',
        template='seaborn',
        height=500,
    )
    fig_bar.show()

    # 2. 累積收益曲線比較圖
    fig_cum = go.Figure()
    for name, result in results.items():
        label = f"{name} ({'+' if result['sign']==1 else '-'})"
        fig_cum.add_trace(go.Scatter(
            x=result['cum_return'].index,
            y=result['cum_return'].values,
            mode='lines',
            name=label,
        ))
    fig_cum.update_layout(
        title='所有因子累積收益曲線比較',
        xaxis_title='日期',
        yaxis_title='累積收益',
        template='seaborn',
        height=500,
    )
    fig_cum.show()

    # 3. 統計指標比較表
    stats_df = pd.DataFrame({name: result['stats'] for name, result in results.items()}).T
    print("所有因子統計指標比較:")
    display(stats_df)

    return results, stats_df
