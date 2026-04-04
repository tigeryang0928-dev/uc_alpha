from functools import reduce
from utils.evaluate import *
from utils.data_bridge import *
from scipy import stats
from sklearn.preprocessing import QuantileTransformer

# 定義一個用來收集函數的 dict
group_funcs_methods = {}
funcs_methods = {}
def register_function(group:str, name=None):
    def decorator(func):
        if group not in group_funcs_methods:
            group_funcs_methods[group] = {}
        group_funcs_methods[group][name or func.__name__] = func
        funcs_methods[name or func.__name__] = func
        return func
    return decorator

# 中性化處理
def Neutralization(Y:pd.DataFrame, *X_list:pd.DataFrame):
    def get_rsid(X:np.ndarray, Y:np.ndarray):
        def get_beta(X:np.ndarray, Y:np.ndarray):
            beta = np.linalg.pinv(X.T@X) @ X.T@Y
            return beta
        beta = get_beta(X, Y)
        y_hat = X @ beta
        rsid = Y - y_hat
        return rsid
    Y_stack = Y.stack()  # test   (feture_stack=True)
    Y_stack.index.names = ['date', 'order_book_id']   
    X = pd.concat({_+1:X_list[_] for _ in range(len(X_list))},axis = 1).stack()
    X.index.names = ['date', 'order_book_id']
    Netrualization_df = pd.concat([Y_stack, X], axis=1).dropna().groupby('date').apply(lambda data:pd.DataFrame(get_rsid(data.iloc[:, 1:].values, data.iloc[:, 0].values), columns=['rsid'], index=data.index.get_level_values('order_book_id')))['rsid']
    Netrualization_df.index.names = [None, 'order_book_id']
    return Netrualization_df.unstack().reindex_like(Y)

# 行業分類數據轉換為數值
def convert_industry_to_numeric(industry_data):
    all_industries = pd.unique(industry_data.values.ravel('K'))
    all_industries = all_industries[pd.notna(all_industries)]
    industry_to_code = {industry: idx for idx, industry in enumerate(sorted(all_industries))}
    processed_data = industry_data.replace(industry_to_code).astype(float)
    return processed_data

# ------------------------------
# math_unary_methods 單元數學方法
# ------------------------------
@register_function('math_unary_methods')
def abs(d):
    """回傳輸入的絕對值。"""
    return d.abs()

@register_function('math_unary_methods')
def sign(d):
    """回傳輸入的符號：正數為1，負數為-1，零為0。"""
    return np.sign(d)

# @register_function('math_unary_methods')
# def log(d):
#     return np.log(d)

# @register_function('cs_math_methods')
# def inverse(d):
#     return 1 / (d + 1e-8)

# @register_function('cs_math_methods')
# def sqrt(d):
#     return np.sqrt(d.abs().clip(lower=0)).mul(np.sign(d))

# @register_function('cs_math_methods')
# def log(d):
#     return np.log1p(d.abs()).mul(np.sign(d))

# @register_function('cs_math_methods')
# def exp(d):
#     return np.exp(d)

# @register_function('cs_math_methods')
# def sigmoid(d):
#     return 1 / (1 + np.exp(-d))

# @register_function('cs_math_methods')
# def tanh(d):
#     return np.tanh(d.abs()).mul(np.sign(d))

# @register_function('cs_math_methods')
# def square(d):
#     return d ** 2


# ------------------------------
# math_binary_methods 二元數學方法
# ------------------------------
@register_function('math_binary_methods')
def pow(d1, n):
    """輸入d1的n次方。"""
    return d1 ** n

# @register_function('math_binary_methods')
# def log(d, base=np.e):
#     """計算d的對數，可指定底數（預設為自然對數）。"""
#     if base == np.e:
#         return np.log(d)
#     return np.log(d) / np.log(base)

@register_function('math_binary_methods')
def signed_power(d, n):
    """回傳保留正負號的n次方。"""
    return np.sign(d) * (d.abs() ** n)


# ------------------------------
# conditional_methods 條件方法
# ------------------------------
# @register_function('conditional_methods')
# def logical_and(condition1, condition2):
#     """元素對元素AND運算，兩者皆True時回傳1，否則回傳0。"""
#     return (condition1 & condition2).astype(int)

# @register_function('conditional_methods')
# def logical_or(condition1, condition2):
#     """元素對元素OR運算，任一為True時回傳1，否則回傳0。"""
#     return (condition1 | condition2).astype(int)

# ------------------------------
# cs_methods_compression 橫截面壓縮方法
# ------------------------------

# @register_function('cs_methods_compression')
# def cs_std(d):
#     """回傳每一橫截面的標準差。"""
#     row_std = d.std(axis=1).replace(0, np.nan)
#     return pd.DataFrame({col: row_std for col in d.columns})

# @register_function('cs_methods_compression')
# def cs_mean(d):
#     """回傳每一橫截面的均值。"""
#     row_mean = d.mean(axis=1)
#     return pd.DataFrame({col: row_mean for col in d.columns})

# @register_function('cs_methods_compression')
# def cs_sum(d):
#     """回傳每一橫截面的總和。"""
#     row_sum = d.sum(axis=1)
#     return pd.DataFrame({col: row_sum for col in d.columns})

# @register_function('cs_methods_compression')
# def cs_max(d):
#     """回傳每一橫截面的最大值。"""
#     row_max = d.max(axis=1)
#     return pd.DataFrame({col: row_max for col in d.columns})

# @register_function('cs_methods_compression')
# def cs_min(d):
#     """回傳每一橫截面的最小值。"""
#     row_min = d.min(axis=1)
#     return pd.DataFrame({col: row_min for col in d.columns})


# ------------------------------
# cs_methods 橫截面方法
# ------------------------------
# 排名相關方法
@register_function('cs_methods')
def cs_non(d):
    return d

@register_function('cs_methods')
def cs_rank(d):
    return d.rank(axis=1)

@register_function('cs_methods')
def cs_pct_rank(d):
    return d.rank(axis=1, pct=True)

# 中心化與標準化方法
@register_function('cs_methods')
def cs_center(d):
    return d.sub(d.mean(axis=1), axis=0)

@register_function('cs_methods')
def cs_zscore(d):
    std_val = d.std(axis=1).replace(0, np.nan)
    return (d.sub(d.mean(axis=1), axis=0)).div(std_val, axis=0)

@register_function('cs_methods')
def cs_mad_mean(d):
    median = d.median(axis=1)
    mad = (d.sub(median, axis=0)).abs().mean(axis=1).replace(0, np.nan)
    return (d.sub(median, axis=0)).div(mad, axis=0)

@register_function('cs_methods')
def cs_mad_median(d):
    median = d.median(axis=1)
    mad = (d.sub(median, axis=0)).abs().median(axis=1).replace(0, np.nan)  # median!
    return (d.sub(median, axis=0)).div(mad, axis=0)

# 規模調整與歸一化方法
@register_function('cs_methods')
def cs_normalization(d):
    range_val = (d.max(axis=1) - d.min(axis=1)).replace(0, np.nan)
    return (d.sub(d.min(axis=1), axis=0)).div(range_val, axis=0)

@register_function('cs_methods')
def cs_scale(d):
    abs_sum = d.abs().sum(axis=1).replace(0, np.nan)
    return d.mul(1).div(abs_sum, axis=0)

@register_function('cs_methods')
def cs_div_max(d):
    row_max = d.max(axis=1).replace(0, np.nan)
    return d.div(row_max, axis=0)

@register_function('cs_methods')
def cs_div_std(d):
    row_std = d.std(axis=1).replace(0, np.nan)
    return d.div(row_std, axis=0)

@register_function('cs_methods')
def cs_softmax(d):
    exp_d = np.exp(d)
    exp_sum = exp_d.sum(axis=1).replace(0, np.nan)
    return exp_d.div(exp_sum, axis=0)

@register_function('cs_methods')
def cs_gaussian(d):
    result = d.copy()
    valid_counts = d.notna().sum(axis=1)
    valid_rows = valid_counts > 3
    if valid_rows.any():
        d_valid = d.loc[valid_rows]
        ranked = d_valid.rank(axis=1, method='average', pct=False)
        counts = d_valid.notna().sum(axis=1).values[:, None]
        uniform = (ranked - 1) / np.where(counts > 1, counts - 1, np.nan)
        uniform_clipped = np.clip(uniform, 0.001, 0.999)
        gaussian_values = stats.norm.ppf(uniform_clipped)
        gaussian_df = pd.DataFrame(gaussian_values, index=d_valid.index, columns=d_valid.columns)
        gaussian_df = gaussian_df.where(d_valid.notna())
        result.loc[valid_rows] = gaussian_df
    return result

@register_function('cs_methods')
def cs_quantile(d):
    result = d.copy().values
    valid_mask = d.notna().values
    for i in range(d.shape[0]):
        row_mask = valid_mask[i]
        if row_mask.sum() > 3:
            row_data = d.values[i, row_mask]
            transformer = QuantileTransformer(output_distribution='normal', random_state=42)
            transformed = transformer.fit_transform(row_data.reshape(-1, 1)).flatten()
            result[i, row_mask] = transformed
    return pd.DataFrame(result, index=d.index, columns=d.columns) 


# ------------------------------
# ts_methods 時間序列方法
# ------------------------------
@register_function('ts_methods')
def ts_non(d, w):
    return d

@register_function('ts_methods')
def ts_rank(d, w):
    return d.rolling(w, min_periods=1).rank()

@register_function('ts_methods')
def ts_mean(d, w):
    return d.rolling(w, min_periods=1).mean()

@register_function('ts_methods')
def ts_std(d, w):
    return d.rolling(w, min_periods=1).std()

@register_function('ts_methods')
def ts_median(d, w):
    return d.rolling(w, min_periods=1).median()

@register_function('ts_methods')
def ts_sum(d, w):
    return d.rolling(w, min_periods=1).sum()

@register_function('ts_methods')
def ts_max(d, w):
    return d.rolling(w, min_periods=1).max()

@register_function('ts_methods')
def ts_min(d, w):
    return d.rolling(w, min_periods=1).min()

# @register_function('ts_methods')
# def ts_argmax(d, w):
#     return d.rolling(w, min_periods=1).apply(np.argmax) + 1

# @register_function('ts_methods')
# def ts_argmin(d, w):
#     return d.rolling(w, min_periods=1).apply(np.argmin) + 1

# 變異係數
@register_function('ts_methods')
def ts_cv(d, w):
    mean_val = d.rolling(w, min_periods=1).mean()
    return d.rolling(w, min_periods=1).std() / mean_val.replace(0, np.nan)

@register_function('ts_methods')
def ts_snr(d, w):
    std_val = d.rolling(w, min_periods=1).std()
    return d.rolling(w, min_periods=1).mean() / std_val.replace(0, np.nan)

# 偏度與峰度
@register_function('ts_methods')
def ts_skew(d, w):
    return d.rolling(w, min_periods=1).skew()

@register_function('ts_methods')
def ts_kurt(d, w):
    return d.rolling(w, min_periods=1).kurt()

# 指數加權移動平均與標準差
@register_function('ts_methods')
def ts_ewma_span(d, w):
    return d.ewm(span=w, min_periods=1, adjust=False).mean()

@register_function('ts_methods')
def ts_ewma_halflife(d, w):
    return d.ewm(halflife=w, min_periods=1, adjust=False).mean()

@register_function('ts_methods')
def ts_ewma_std(d, w):
    return d.ewm(span=w, min_periods=1, adjust=False).std()

# 變化率與Z-score
@register_function('ts_methods')
def ts_skew_rate(d, w):     
    return d.rolling(w, min_periods=1).skew().diff()

@register_function('ts_methods')
def ts_kurt_rate(d, w):     
    return d.rolling(w, min_periods=1).kurt().diff()

@register_function('ts_methods')
def ts_cv_rate(d, w):
    mean_val = d.rolling(w, min_periods=1).mean()
    return (d.rolling(w, min_periods=1).std() / mean_val.replace(0, np.nan)).diff()

@register_function('ts_methods')
def ts_pct_change(d, w):
    return d.pct_change(periods=w)

@register_function('ts_methods')
def ts_zscore(d, w):
    rolling_mean = d.rolling(w, min_periods=1).mean()
    rolling_std = d.rolling(w, min_periods=1).std()
    return (d - rolling_mean) / rolling_std.replace(0, np.nan)

@register_function('ts_methods')
def ts_dema(d, w):
    rolling_mean = d.rolling(w, min_periods=1).mean()
    return d - rolling_mean

@register_function('ts_methods')
def ts_delta(d, w):
    return d.diff(w)

@register_function('ts_methods')
def ts_delay(d, w):
    return d.shift(w)

@register_function('ts_methods')
def ts_mtm(d, w):
    shift = d.shift(w)
    return d / shift.replace(0, np.nan)

@register_function('ts_methods')
def ts_bias(d, w):
    rolling_mean = d.rolling(w, min_periods=1).mean()
    return d / rolling_mean.replace(0, np.nan)

@register_function('ts_methods')
def ts_ewm_bias(d, w):
    rolling_emean = d.ewm(span=w, min_periods=1, adjust=False).mean()
    return d / rolling_emean.replace(0, np.nan)

# 滾動排名方法
@register_function('ts_methods')
def ts_rank(d, w):
    return d.rolling(w, min_periods=1).rank()

@register_function('ts_methods')
def ts_pct_rank(d, w):
    return d.rolling(w, min_periods=1).rank(pct=True)

# 滾動極值相關方法
@register_function('ts_methods')
def ts_max_distance(d, w):
    return d.sub(d.rolling(w, min_periods=1).max(), axis=0)

@register_function('ts_methods')
def ts_min_distance(d, w):
    return d.sub(d.rolling(w, min_periods=1).min(), axis=0)

@register_function('ts_methods')
def ts_max_ratio(d, w):
    return d.div(d.rolling(w, min_periods=1).max(), axis=0) - 1

@register_function('ts_methods')
def ts_min_ratio(d, w):
    return d.div(d.rolling(w, min_periods=1).min(), axis=0) - 1

@register_function('ts_methods')
def ts_range(d, w):
    return d.rolling(w, min_periods=1).max().sub(d.rolling(w, min_periods=1).min(), axis=0)

# ------------------------------
# group_methods 分組橫截面方法
# ------------------------------
@register_function('group_methods')
def group_min(d, group):
    # 同一分組 y (如產業別) 中，變數 x 的最小值
    def _group_apply(row):
        g = group.loc[row.name]
        return row.groupby(g).transform('min')
    return d.apply(_group_apply, axis=1)

@register_function('group_methods')
def group_max(d, group):
    # 同一分組 y 中，變數 x 的最大值
    def _group_apply(row):
        g = group.loc[row.name]
        return row.groupby(g).transform('max')
    return d.apply(_group_apply, axis=1)

@register_function('group_methods')
def group_avg(d, group):
    # 同一分組 y 中，變數 x 的平均值
    def _group_apply(row):
        g = group.loc[row.name]
        return row.groupby(g).transform('mean')
    return d.apply(_group_apply, axis=1)


# ------------------------------
# interaction_methods 交互作用方法
# ------------------------------
@register_function('special_interaction_methods')
def interaction_sub(d1, d2): 
    return d1 - d2

# @register_function('interaction_methods')
# def interaction_add_r(d1, d2): 
#     return d1.rank(axis=1) + d2.rank(axis=1)

# @register_function('interaction_methods')
# def interaction_add_z(d1, d2): 
#     d1z = (d1.sub(d1.mean(axis=1), axis=0)).div(d1.std(axis=1).replace(0, np.nan), axis=0)
#     d2z = (d2.sub(d2.mean(axis=1), axis=0)).div(d2.std(axis=1).replace(0, np.nan), axis=0)
#     return d1z + d2z

# @register_function('interaction_methods')
# def interaction_sub_r(d1, d2): 
#     return d1.rank(axis=1) - d2.rank(axis=1)

# @register_function('interaction_methods')
# def interaction_sub_z(d1, d2): 
#     d1z = (d1.sub(d1.mean(axis=1), axis=0)).div(d1.std(axis=1).replace(0, np.nan), axis=0)
#     d2z = (d2.sub(d2.mean(axis=1), axis=0)).div(d2.std(axis=1).replace(0, np.nan), axis=0)
#     return d1z - d2z

# @register_function('interaction_methods')
# def interaction_mul_r(d1, d2): 
#     return d1.rank(axis=1, pct=True) * d2.rank(axis=1, pct=True)

# @register_function('interaction_methods')
# def interaction_mul_z(d1, d2): 
#     d1z = (d1.sub(d1.mean(axis=1), axis=0)).div(d1.std(axis=1).replace(0, np.nan), axis=0)
#     d2z = (d2.sub(d2.mean(axis=1), axis=0)).div(d2.std(axis=1).replace(0, np.nan), axis=0)
#     return d1z * d2z

@register_function('interaction_methods')
def interaction_div(d1, d2): 
    return d1 / d2.replace(0, np.nan)

# # 比較操作
# @register_function('interaction_methods')
# def interaction_max_r(d1, d2): 
#     return (d1.rank(axis=1)).combine(d2.rank(axis=1), np.maximum)

# @register_function('interaction_methods')
# def interaction_min_r(d1, d2): 
#     return (d1.rank(axis=1)).combine(d2.rank(axis=1), np.minimum)

# # 平均操作
# @register_function('interaction_methods')
# def interaction_mean_r(d1, d2):   
#     return (d1.rank(axis=1) + d2.rank(axis=1)) / 2

# @register_function('interaction_methods')
# def interaction_harmonic_mean_r(d1, d2):
#     denominator = (1 / (d1.rank(axis=1, pct=True)) + 1 / (d2.rank(axis=1, pct=True))).replace(0, np.nan)
#     return 2 / denominator

# @register_function('interaction_methods')
# def interaction_sqrt_mul_r(d1, d2):   
#     return (d1.rank(axis=1, pct=True) * d2.rank(axis=1, pct=True)).pow(0.5)

# # 比率操作
# @register_function('interaction_methods')
# def interaction_log_ratio(d1, d2):
#     # log 的自然延伸到負數域
#     return np.arcsinh(d1) - np.arcsinh(d2)

# 回歸操作
# @register_function('interaction_methods')
# def interaction_neut(d1, d2):  
#     neut = Neutralization(d1, d2)
#     return neut

@register_function('interaction_methods')
def interaction_neut_r(d1, d2):  
    neut = Neutralization(d1.rank(axis=1), d2.rank(axis=1))
    return neut

@register_function('interaction_methods')
def interaction_neut_g(d1, d2):  
    d1g = cs_gaussian(d1)
    d2g = cs_gaussian(d2)
    neut = Neutralization(d1g, d2g)
    return neut


# ------------------------------
# 6. interaction_ts_methods 時序交互方法
# ------------------------------
@register_function('interaction_ts_methods')
def ts_corr(d1, d2, w):
    """計算過去w天內的Pearson相關係數。"""
    return d1.rolling(w, min_periods=1).corr(d2)

@register_function('interaction_ts_methods')
def ts_cov(d1, d2, w):
    """計算過去w天內的共變異數。"""
    return d1.rolling(w, min_periods=1).cov(d2)


# ------------------------------
# neut_methods 中性化操作
# ------------------------------
# 使用前需先載入相關數據，如市值、產業別等
# handler = Handler(os.path.join(os.path.expanduser('~/Documents'), 'TW_TEJ_DATA'))
# cap = handler['Market_Cap_Dollars']
# industry_numeric = convert_industry_to_numeric(handler['Industry'])


# @register_function('neut_methods')
# def neut_cap(d):  
#     d = Neutralization(d, cap)
#     return d

# @register_function('neut_methods')
# def neut_cap_rank(d):  
#     d = Neutralization(d, cap.rank(axis=1))
#     return d

# @register_function('neut_methods')
# def neut_cap_gaussian(d):  
#     d = Neutralization(d, cs_gaussian(cap))
#     return d

# @register_function('neut_methods')
# def neut_industry(d):  
#     d = Neutralization(d, industry_numeric)
#     return d