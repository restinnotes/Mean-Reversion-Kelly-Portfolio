import numpy as np
import pandas as pd
import os

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
# PART 2 — OU PARAMETER CALCULATION (UPDATED with Confidence)
# ============================================================================

def calculate_ou_params(series, dt=1):
    """
    Estimate OU parameters and confidence metrics.
    """
    series = series.dropna()
    if len(series) < 10: return None

    x_t  = series[:-1].values
    x_t1 = series[1:].values

    # Linear regression: x_{t+1} = alpha + beta * x_t + ε
    results = ols_regression(x_t, x_t1)
    if results[0] is None: return None

    # Unpack including SE
    params, residuals, r2, se_params = results
    alpha, beta = params
    alpha_se, beta_se = se_params

    resid_std = residuals.std()

    # --- Confidence Calculation ---
    # Null Hypothesis: Beta = 1 (Random Walk)
    # T-statistic = (Beta - 1) / SE(Beta)
    # Interpretation: More negative = Stronger Evidence of Reversion
    if beta_se > 0:
        t_stat_adf = (beta - 1) / beta_se
    else:
        t_stat_adf = 0.0

    # ------------------------------

    if beta >= 1:
        return {
            'lambda': 1e-4,
            'theta': series.mean(),
            'sigma': resid_std,
            'half_life': np.inf,
            'r_squared': r2,
            't_stat': t_stat_adf, # return t-stat even if non-stationary
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
        't_stat': t_stat_adf,     # <--- 核心指标：ADF T统计量
        'beta_se': beta_se,       # 斜率标准误
        'status': "MEAN_REVERTING"
    }




# ============================================================================
# PART 3 — CSV LOADING
# ============================================================================

def load_pe_csv(csv_dir):
    """
    Load all *_pe.csv files in folder.
    """
    extracted = {}

    # Directory must exist
    if not os.path.exists(csv_dir):
        return extracted

    for fname in os.listdir(csv_dir):
        if fname.endswith("_pe.csv"):
            ticker = fname.split("_pe.csv")[0]
            path = os.path.join(csv_dir, fname)

            df = pd.read_csv(path, parse_dates=['date'])
            extracted[ticker] = df

    return extracted


# ============================================================================
# PART 4 — MAIN ACCESSOR
# ============================================================================

def get_ou_for_ticker(ticker, dt=1, window=None):
    """
    Read the PE CSV for a specific ticker and compute OU parameters.

    Parameters:
    -----------
    ticker: str
        Symbol (e.g. "NVDA")
    dt: float
        Time step (default 1 day)
    window: int or None
        If None, calculate using ALL historical data (Static).
        If int (e.g., 90), calculate using only the LAST `window` days (Rolling Snapshot).
    """
    # Project root = two levels above this file
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    csv_path = os.path.join(project_root, "pe_csv", f"{ticker}_pe.csv")

    # Check file exists
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"PE CSV not found for {ticker}: {csv_path}")

    df = pd.read_csv(csv_path, parse_dates=['date'])

    # Require "value"
    if 'value' not in df.columns:
        raise ValueError(f"CSV must contain 'value' column: {csv_path}")

    series = df.set_index('date')['value'].sort_index()

    # Apply Window Slicing if requested
    if window is not None:
        if len(series) < window:
            raise ValueError(f"Not enough data for rolling window. Have {len(series)}, need {window}.")
        series = series.iloc[-window:]

    ou_params = calculate_ou_params(series, dt=dt)
    if ou_params is None:
        raise ValueError(f"Not enough data to calculate OU parameters for {ticker}")

    return ou_params


# ============================================================================
# PART 5 — EXAMPLE
# ============================================================================

if __name__ == "__main__":
    ticker = "NVDA"

    try:
        # 1. Static (All History)
        ou_static = get_ou_for_ticker(ticker)
        print(f"[{ticker} STATIC] λ = {ou_static['lambda']:.4f}, Half-Life = {ou_static['half_life']:.2f} days")

        # 2. Rolling Window (Last 90 days)
        window_size = 90
        ou_rolling = get_ou_for_ticker(ticker, window=window_size)
        print(f"[{ticker} ROLLING {window_size}d] λ = {ou_rolling['lambda']:.4f}, Half-Life = {ou_rolling['half_life']:.2f} days")

    except Exception as e:
        print(f"Error: {e}")