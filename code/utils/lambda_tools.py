import numpy as np
import pandas as pd
import os
from scipy import stats  # <--- 新增：用于计算概率

# ============================================================================
# PART 1 — SIMPLE OLS (UPDATED with Standard Error)
# ============================================================================

def ols_regression(X, y):
    """
    Perform a simple OLS regression of y on X (with intercept).
    Returns beta, residuals, r_squared, AND standard_errors.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    n = len(y)

    # Add constant column (intercept)
    X_with_const = np.column_stack([np.ones(n), X])
    p = X_with_const.shape[1] # Number of parameters (2: alpha, beta)

    # Normal equation components
    XtX = X_with_const.T @ X_with_const
    Xty = X_with_const.T @ y

    # Solve β = (X^T X)^(-1) X^T y
    try:
        # Compute inverse for Standard Error calculation
        XtX_inv = np.linalg.inv(XtX)
        beta = XtX_inv @ Xty
    except np.linalg.LinAlgError:
        return None, None, 0, None

    # Predictions
    y_pred = X_with_const @ beta

    # Residuals
    residuals = y - y_pred

    # R² calculation
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - y.mean())**2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # --- NEW: Standard Error & T-Stat Calculation ---
    # Variance of residuals (MSE)
    if n > p:
        mse = ss_res / (n - p)
        # Variance-Covariance Matrix of Beta: MSE * (X'X)^-1
        var_beta = mse * XtX_inv
        # Standard Errors are sqrt of diagonal elements
        se_beta = np.sqrt(np.diag(var_beta))
    else:
        se_beta = np.zeros_like(beta)

    return beta, residuals, r_squared, se_beta


# ============================================================================
# PART 2 — HELPER: T-STAT TO PROBABILITY
# ============================================================================

def t_stat_to_confidence(t_stat):
    """
    将 T-Stat 转换为均值回归的置信度百分比。
    Null Hypothesis: Random Walk (Beta >= 0)
    Alternative: Mean Reverting (Beta < 0)
    我们使用的是单尾检验 (Left-tailed test)。
    """
    if t_stat is None or np.isnan(t_stat):
        return 0.0

    # 使用累积分布函数 (CDF)
    # 假设大样本，自由度很高，可以用 norm 或 t(df=100) 近似
    # P(T < t_stat) 就是它是均值回归的概率 (Rejection region is on the left)
    # 例如 t = -1.65 -> p_value = 0.05 -> Confidence = 95%
    p_value = stats.norm.cdf(t_stat)

    # 置信度 = (1 - p_value) * 100
    # 这里的逻辑是：p_value 是"它实际上是随机游走但看起来像回归"的概率
    # 所以 1 - p_value 就是"它是均值回归"的置信度
    confidence = (1 - p_value) * 100
    return confidence


# ============================================================================
# PART 3 — OU PARAMETER CALCULATION
# ============================================================================

def calculate_ou_params(series, dt=1):
    """
    Estimate OU parameters and confidence metrics on a specific series slice.
    """
    series = series.dropna()
    if len(series) < 10: return None

    x_t  = series[:-1].values
    x_t1 = series[1:].values

    # Linear regression: x_{t+1} = alpha + beta * x_t + ε
    results = ols_regression(x_t, x_t1)
    if results[0] is None: return None

    params, residuals, r2, se_params = results
    alpha, beta = params
    alpha_se, beta_se = se_params

    resid_std = residuals.std()

    if beta_se > 0:
        t_stat_adf = (beta - 1) / beta_se
    else:
        t_stat_adf = 0.0

    # 计算当前切片的置信度
    conf_adf = t_stat_to_confidence(t_stat_adf)

    if beta >= 1:
        return {
            'lambda': 1e-4,
            'theta': series.mean(),
            'sigma': resid_std,
            'half_life': np.inf,
            'r_squared': r2,
            't_stat': t_stat_adf,
            'confidence': conf_adf,
            'status': "NON_STATIONARY"
        }

    lam = -np.log(beta) / dt
    theta = alpha / (1 - beta)

    if 0 < beta < 1:
        sigma = resid_std * np.sqrt(-2 * np.log(beta) / (1 - beta**2) / dt)
    else:
        sigma = resid_std / np.sqrt(dt)

    half_life = np.log(2) / lam if lam > 0 else np.inf

    return {
        'lambda': lam,
        'theta': theta,
        'sigma': sigma,
        'half_life': half_life,
        'half_life_years': half_life / 252,
        'r_squared': r2,
        't_stat': t_stat_adf,
        'confidence': conf_adf, # <--- 新增
        'beta_se': beta_se,
        'status': "MEAN_REVERTING"
    }


# ============================================================================
# PART 4 — ROBUST HISTORICAL VERIFICATION
# ============================================================================

def calculate_historical_ma_reversion(full_series, window, dt=1):
    """
    计算【历史均值回归显著性】：
    假设我们总是以 `window` 日均线作为锚点，历史数据是否支持“偏离即回归”的假设？
    """
    if len(full_series) < window + 10:
        return None

    rolling_ma = full_series.rolling(window=window).mean()
    deviation = full_series - rolling_ma
    next_step_change = full_series.shift(-1) - full_series

    df_reg = pd.DataFrame({
        'y': next_step_change,
        'x': deviation
    }).dropna()

    if len(df_reg) < 10:
        return None

    results = ols_regression(df_reg['x'], df_reg['y'])
    if results[0] is None: return None

    params, residuals, r2, se_params = results
    alpha, beta = params
    _, beta_se = se_params

    t_stat_structural = beta / beta_se if beta_se > 0 else 0.0

    # 计算结构性置信度
    conf_structural = t_stat_to_confidence(t_stat_structural)

    return {
        'structural_beta': beta,
        'structural_t_stat': t_stat_structural,
        'structural_confidence': conf_structural, # <--- 新增
        'structural_r2': r2,
        'valid_samples': len(df_reg)
    }


# ============================================================================
# PART 5 — CSV LOADING & MAIN ACCESSOR
# ============================================================================

def load_pe_csv(csv_dir):
    extracted = {}
    if not os.path.exists(csv_dir):
        return extracted
    for fname in os.listdir(csv_dir):
        if fname.endswith("_pe.csv"):
            ticker = fname.split("_pe.csv")[0]
            path = os.path.join(csv_dir, fname)
            df = pd.read_csv(path, parse_dates=['date'])
            extracted[ticker] = df
    return extracted

def get_ou_for_ticker(ticker, dt=1, window=None):
    """
    Main entry point.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))

    csv_path = os.path.join(project_root, "pe_csv", f"{ticker}_pe.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"PE CSV not found for {ticker}: {csv_path}")

    df = pd.read_csv(csv_path, parse_dates=['date'])

    if 'value' not in df.columns:
        raise ValueError(f"CSV must contain 'value' column: {csv_path}")

    full_series = df.set_index('date')['value'].sort_index()

    if window is not None:
        if len(full_series) < window:
            raise ValueError(f"Not enough data. Have {len(full_series)}, need {window}.")
        calc_series = full_series.iloc[-window:]
    else:
        calc_series = full_series

    ou_params = calculate_ou_params(calc_series, dt=dt)
    if ou_params is None:
        raise ValueError(f"Not enough data to calculate OU parameters for {ticker}")

    if window is not None:
        historical_stats = calculate_historical_ma_reversion(full_series, window, dt)
        if historical_stats:
            ou_params['robust_t_stat'] = historical_stats['structural_t_stat']
            ou_params['robust_confidence'] = historical_stats['structural_confidence'] # <--- 新增
            ou_params['robust_beta'] = historical_stats['structural_beta']
        else:
            ou_params['robust_t_stat'] = np.nan
            ou_params['robust_confidence'] = 0.0
    else:
        ou_params['robust_t_stat'] = ou_params['t_stat']
        ou_params['robust_confidence'] = ou_params['confidence']

    return ou_params


# ============================================================================
# PART 6 — EXAMPLE
# ============================================================================

if __name__ == "__main__":
    ticker = "NVDA"

    try:
        window_size = 90

        print(f"--- Analyzing {ticker} with {window_size}-day Mean Reversion Logic ---")
        ou = get_ou_for_ticker(ticker, window=window_size)

        print(f"Current Lambda (Speed): {ou['lambda']:.4f}")
        print(f"Current Half-Life:      {ou['half_life']:.2f} days")
        print(f"Current T-Stat (Lagged): {ou['t_stat']:.2f}  (Conf: {ou['confidence']:.1f}%)")
        print("-" * 30)

        t_hist = ou['robust_t_stat']
        conf_hist = ou['robust_confidence']

        print(f"Robust T-Stat (All History): {t_hist:.2f}")
        print(f"Mean Reversion Confidence:   {conf_hist:.1f}%") # <--- 这里会打印出 89.2%

        # 调整了评价标准，不再死板
        if conf_hist > 95:
            print("CONCLUSION: [STRONG] Strategy is statistically significant (>95%).")
        elif conf_hist > 85:
            print("CONCLUSION: [MODERATE] Strategy has valid edge (>85%), but assumes risk.")
        else:
            print("CONCLUSION: [WEAK] Edge is statistically indistinguishable from noise.")

    except Exception as e:
        print(f"Error: {e}")