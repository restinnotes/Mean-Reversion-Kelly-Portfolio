import numpy as np
import pandas as pd
import os

# ============================================================================
# PART 1 — SIMPLE OLS (NO statsmodels)
# PART 1 — 简单 OLS 实现（不使用 statsmodels）
# ============================================================================

def ols_regression(X, y):
    """
    Perform a simple OLS regression of y on X (with intercept).
    执行简单的线性回归：用 X 回归 y，自动加入截距项。
    """
    X = np.asarray(X)
    y = np.asarray(y)

    # Add constant column (intercept)
    # 添加常数项（截距）
    X_with_const = np.column_stack([np.ones(len(X)), X])

    # Normal equation components
    # 正规方程计算项
    XtX = X_with_const.T @ X_with_const
    Xty = X_with_const.T @ y

    # Solve β = (X^T X)^(-1) X^T y
    # 求解 β
    beta = np.linalg.solve(XtX, Xty)

    # Predictions
    # 回归预测值
    y_pred = X_with_const @ beta

    # Residuals
    # 残差
    residuals = y - y_pred

    # R² calculation
    # 计算 R²
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - y.mean())**2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    return beta, residuals, r_squared


# ============================================================================
# PART 2 — OU PARAMETER CALCULATION
# PART 2 — OU 参数估计
# ============================================================================

def calculate_ou_params(series, dt=1):
    """
    Estimate OU (Ornstein-Uhlenbeck) parameters from a time series.
    根据时间序列估计 OU（均值回复过程）参数。
    """
    series = series.dropna()

    # Require sufficient data points
    # 数据太少就不计算
    if len(series) < 10:
        return None

    # x_t  and x_(t+1)
    # 构造 x_t 与 x_(t+1)
    x_t  = series[:-1].values
    x_t1 = series[1:].values

    # Linear regression: x_{t+1} = alpha + beta * x_t + ε
    # OLS 回归：x_{t+1} = α + β x_t + ε
    params, residuals, r2 = ols_regression(x_t, x_t1)
    alpha, beta = params
    resid_std = residuals.std()

    # If beta >= 1 ➜ non-stationary, no mean reversion
    # 若 β≥1 ➜ 非平稳，不存在均值回复
    if beta >= 1:
        return {
            'lambda': 0.01,               # fallback small λ
            'theta': series.mean(),       # mean
            'sigma': resid_std,
            'half_life': np.inf,
            'beta': beta,
            'alpha': alpha,
            'r_squared': r2,
            'status': "NON_STATIONARY"
        }

    # Convert OU discrete beta into continuous-time lambda
    # 将离散 OU 系数 β 转换为连续时间参数 λ
    lam = -np.log(beta) / dt

    # OU long-term mean theta
    # 长期均值 θ
    theta = alpha / (1 - beta)

    # OU volatility conversion (standard formula)
    # OU 波动率转换（标准公式）
    if 0 < beta < 1:
        sigma = resid_std * np.sqrt(-2 * np.log(beta) / (1 - beta**2) / dt)
    else:
        sigma = resid_std / np.sqrt(dt)

    # Half-life = ln(2) / λ
    # 半衰期公式
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


# ============================================================================
# PART 3 — CSV LOADING
# PART 3 — 从 CSV 读取 PE 数据
# ============================================================================

def load_pe_csv(csv_dir):
    """
    Load all *_pe.csv files in folder.
    读取文件夹中所有 *_pe.csv 文件。
    """
    extracted = {}

    # Directory must exist
    # 文件夹不存在则返回空
    if not os.path.exists(csv_dir):
        return extracted

    for fname in os.listdir(csv_dir):
        if fname.endswith("_pe.csv"):
            ticker = fname.split("_pe.csv")[0]
            path = os.path.join(csv_dir, fname)

            df = pd.read_csv(path, parse_dates=['date'])
            extracted[ticker] = df

    return extracted


def calculate_ou_for_csv(csv_dir, dt=1):
    """
    Calculate OU parameters for all tickers in a folder.
    为文件夹内全部股票计算 OU 参数。
    """
    extracted_dict = load_pe_csv(csv_dir)
    results = {}

    for ticker, df in extracted_dict.items():

        # Require 'value' column
        # 必须有 value 列
        if 'value' not in df.columns:
            continue

        series = df.set_index('date')['value']

        params = calculate_ou_params(series, dt=dt)
        if params:
            results[ticker] = params

    return results


# ============================================================================
# PART 4 — MAIN ACCESSOR
# PART 4 — 单标的接口
# ============================================================================

def get_ou_for_ticker(ticker, dt=1):
    """
    Read the PE CSV for a specific ticker and compute OU parameters.
    自动读取 <项目根目录>/pe_csv/<TICKER>_pe.csv，并计算 OU 参数。
    """
    # Project root = two levels above this file
    # 项目根目录（向上两级）
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    csv_path = os.path.join(project_root, "pe_csv", f"{ticker}_pe.csv")

    # Check file exists
    # 检查 CSV 是否存在
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"PE CSV not found for {ticker}: {csv_path}")

    df = pd.read_csv(csv_path, parse_dates=['date'])

    # Require "value"
    # 必须包含 value 列
    if 'value' not in df.columns:
        raise ValueError(f"CSV must contain 'value' column: {csv_path}")

    series = df.set_index('date')['value']

    ou_params = calculate_ou_params(series, dt=dt)
    if ou_params is None:
        raise ValueError(f"Not enough data to calculate OU parameters for {ticker}")

    return ou_params


# ============================================================================
# PART 5 — EXAMPLE
# PART 5 — 示例调用
# ============================================================================

if __name__ == "__main__":
    ticker = "NVDA"

    try:
        ou = get_ou_for_ticker(ticker)
        print(f"[{ticker}] λ = {ou['lambda']:.4f}, σ = {ou['sigma']:.4f}, 半衰期 = {ou['half_life']:.2f} 天")
    except Exception as e:
        print(f"Error: {e}")
