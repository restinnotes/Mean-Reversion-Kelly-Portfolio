# code/data/lambda_tools.py

import numpy as np
import pandas as pd
import os
from scipy import stats

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
    p = X_with_const.shape[1]

    # Normal equation components
    XtX = X_with_const.T @ X_with_const
    Xty = X_with_const.T @ y

    try:
        XtX_inv = np.linalg.inv(XtX)
        beta = XtX_inv @ Xty
    except np.linalg.LinAlgError:
        return None, None, 0, None

    y_pred = X_with_const @ beta
    residuals = y - y_pred

    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - y.mean())**2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    if n > p:
        mse = ss_res / (n - p)
        var_beta = mse * XtX_inv
        se_beta = np.sqrt(np.diag(var_beta))
    else:
        se_beta = np.zeros_like(beta)

    return beta, residuals, r_squared, se_beta


# ============================================================================
# PART 2 — HELPER: T-STAT TO PROBABILITY
# ============================================================================

def t_stat_to_confidence(t_stat):
    """
    将 T-Stat 转换为均值回归的置信度百分比。使用单尾检验 (Left-tailed test)。
    """
    if t_stat is None or np.isnan(t_stat):
        return 0.0

    p_value = stats.norm.cdf(t_stat)
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

    results = ols_regression(x_t, x_t1)
    if results[0] is None: return None

    params, residuals, r2, se_params = results
    # alpha, beta = params # unused
    beta = params[1]
    beta_se = se_params[1]

    resid_std = residuals.std()

    if beta_se > 0:
        t_stat_adf = (beta - 1) / beta_se
    else:
        t_stat_adf = 0.0

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
        'confidence': conf_adf,
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
    beta = params[1]
    beta_se = se_params[1]

    t_stat_structural = beta / beta_se if beta_se > 0 else 0.0
    conf_structural = t_stat_to_confidence(t_stat_structural)

    return {
        'structural_beta': beta,
        'structural_t_stat': t_stat_structural,
        'structural_confidence': conf_structural,
        'structural_r2': r2,
        'valid_samples': len(df_reg)
    }


# ============================================================================
# PART 5 — CSV LOADING & MAIN ACCESSOR
# ============================================================================

def get_ou_for_ticker(ticker, project_root, dt=1, window=None):
    """
    Main entry point for OU calculation.
    (Modified to accept project_root for file path resolution)
    """
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
            ou_params['robust_confidence'] = historical_stats['structural_confidence']
            ou_params['robust_beta'] = historical_stats['structural_beta']
        else:
            ou_params['robust_t_stat'] = np.nan
            ou_params['robust_confidence'] = 0.0
    else:
        ou_params['robust_t_stat'] = ou_params['t_stat']
        ou_params['robust_confidence'] = ou_params['confidence']

    return ou_params