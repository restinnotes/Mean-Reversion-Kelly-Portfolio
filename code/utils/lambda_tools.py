import numpy as np
import pandas as pd
import os

# ============================================================================
# PART 1 — SIMPLE OLS (NO statsmodels)
# ============================================================================

def ols_regression(X, y):
    """
    Perform a simple OLS regression of y on X (with intercept).
    """
    X = np.asarray(X)
    y = np.asarray(y)

    # Add constant column (intercept)
    X_with_const = np.column_stack([np.ones(len(X)), X])

    # Normal equation components
    XtX = X_with_const.T @ X_with_const
    Xty = X_with_const.T @ y

    # Solve β = (X^T X)^(-1) X^T y
    try:
        beta = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        # Handle singular matrix
        return None, None, 0

    # Predictions
    y_pred = X_with_const @ beta

    # Residuals
    residuals = y - y_pred

    # R² calculation
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - y.mean())**2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    return beta, residuals, r_squared


# ============================================================================
# PART 2 — OU PARAMETER CALCULATION
# ============================================================================

def calculate_ou_params(series, dt=1):
    """
    Estimate OU (Ornstein-Uhlenbeck) parameters from a time series.
    """
    series = series.dropna()

    # Require sufficient data points
    if len(series) < 10:
        return None

    # x_t and x_{t+1}
    x_t  = series[:-1].values
    x_t1 = series[1:].values

    # Linear regression: x_{t+1} = alpha + beta * x_t + ε
    results = ols_regression(x_t, x_t1)
    if results[0] is None:
        return None

    params, residuals, r2 = results
    alpha, beta = params
    resid_std = residuals.std()

    # If beta >= 1 ➜ non-stationary, no mean reversion
    if beta >= 1:
        return {
            'lambda': 0.01,              # fallback small λ
            'theta': series.mean(),      # mean
            'sigma': resid_std,
            'half_life': np.inf,
            'beta': beta,
            'alpha': alpha,
            'r_squared': r2,
            'status': "NON_STATIONARY"
        }

    # Convert OU discrete beta into continuous-time lambda
    lam = -np.log(beta) / dt

    # OU long-term mean theta
    theta = alpha / (1 - beta)

    # OU volatility conversion (standard formula)
    if 0 < beta < 1:
        sigma = resid_std * np.sqrt(-2 * np.log(beta) / (1 - beta**2) / dt)
    else:
        sigma = resid_std / np.sqrt(dt)

    # Half-life = ln(2) / λ
    half_life = np.log(2) / lam if lam > 0 else np.inf

    return {
        'lambda': lam,
        'theta': theta,
        'sigma': sigma,
        'half_life': half_life,
        'half_life_years': half_life / 252,
        'beta': beta,
        'alpha': alpha,
        'r_squared': r2,
        'status': "MEAN_REVERTING" if beta < 0.99 else "WEAK_REVERSION"
    }

def calculate_rolling_ou_params(series, window=90, dt=1):
    """
    Calculate OU parameters on a rolling window.
    Returns a DataFrame with index matching the series (dates at end of window).
    """
    results = []
    indices = []

    # Iterate through the series with a sliding window
    for i in range(window, len(series) + 1):
        window_series = series.iloc[i-window:i]
        date = series.index[i-1]

        params = calculate_ou_params(window_series, dt=dt)
        if params:
            # Flatten dict for DataFrame
            row = params.copy()
            row['date'] = date
            results.append(row)

    if not results:
        return pd.DataFrame()

    df_results = pd.DataFrame(results).set_index('date')
    return df_results


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