
import numpy as np
import pandas as pd
import os
import scipy.stats as stats  # for t-distribution; 用于 t 分布计算


# ============================================================================
# PART 1 — OLS WITH STANDARD ERROR
# PART 1 — 带标准误差的 OLS 线性回归
# ============================================================================

def ols_regression_with_se(X, y):
    """
    Perform OLS regression of y on X with intercept, including standard error of beta.
    执行带截距项的 OLS 回归，并计算回归系数 beta 的标准误。
    """
    X = np.asarray(X)
    y = np.asarray(y)
    n = len(X)

    # Add constant column
    # 添加常数项
    X_with_const = np.column_stack([np.ones(n), X])

    XtX = X_with_const.T @ X_with_const
    Xty = X_with_const.T @ y

    # Solve β = (XᵀX)^(-1) Xᵀy
    # 求解 β
    try:
        beta_vec = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        return None, None, None, None  # Singular matrix; 奇异矩阵处理

    alpha, beta = beta_vec

    # Predictions and residuals
    # 预测值与残差
    y_pred = X_with_const @ beta_vec
    residuals = y - y_pred

    rss = np.sum(residuals**2)   # residual sum of squares 残差平方和
    df = n - 2                   # degrees of freedom 自由度

    if df <= 0:
        return None, None, None, None

    mse = rss / df               # mean squared error 均方误差

    # Standard error of beta
    # beta 系数的标准误
    x_mean = np.mean(X)
    s_xx = np.sum((X - x_mean)**2)
    se_beta = np.sqrt(mse / s_xx) if s_xx > 0 else np.inf

    # R² = 1 - RSS / TSS
    # 计算决定系数 R²
    r_squared = 1 - rss / np.sum((y - np.mean(y))**2)

    return (alpha, beta), se_beta, residuals, r_squared


# ============================================================================
# PART 2 — OU PARAMETER CALCULATION (WITH CI)
# PART 2 — 带置信区间的 OU 参数估计
# ============================================================================

def calculate_ou_params(series, dt=1, confidence=0.95):
    """
    Estimate OU parameters with beta confidence interval.
    基于 OLS 回归估计 OU 参数，并计算 beta 的置信区间。
    """

    series = series.dropna()

    if len(series) < 10:
        return None  # too little data; 数据太少

    # Construct x_t and x_(t+1)
    # 构造序列 x_t 与 x_{t+1}
    x_t  = series[:-1].values
    x_t1 = series[1:].values

    # Run OLS
    # 运行 OLS 回归
    params, se_beta, residuals, r2 = ols_regression_with_se(x_t, x_t1)
    if params is None:
        return None

    alpha, beta = params

    # ----------------------------------------------------------------------
    # Confidence Interval of beta
    # beta 的置信区间
    # ----------------------------------------------------------------------

    t_score = stats.t.ppf((1 + confidence) / 2, len(x_t) - 2)

    beta_lower = beta - t_score * se_beta
    beta_upper = beta + t_score * se_beta

    # Avoid beta >= 1 for log()
    # 避免 beta >= 1 造成 log 非法
    beta_upper = min(beta_upper, 0.999)
    beta_lower = min(beta_lower, 0.999)

    # ----------------------------------------------------------------------
    # Convert discrete AR(1) to continuous OU
    # 将离散 AR(1) 系数转换为连续 OU 参数
    # ----------------------------------------------------------------------

    lam_est = -np.log(beta) / dt
    lam_min = -np.log(beta_upper) / dt  # slowest reversion 最慢均值回复
    lam_max = -np.log(beta_lower) / dt  # fastest reversion 最快均值回复

    # Half-life
    # 半衰期
    hl_est = np.log(2) / lam_est
    hl_max = np.log(2) / lam_min
    hl_min = np.log(2) / lam_max

    return {
        'lambda': lam_est,
        'lambda_min': lam_min,     # pessimistic; 悲观估计
        'lambda_max': lam_max,     # optimistic; 乐观估计

        'half_life': hl_est,
        'half_life_max_days': hl_max,
        'half_life_min_days': hl_min,

        'sigma': residuals.std(),  # simplified sigma; 简化 sigma
        'beta_se': se_beta,
        'r_squared': r2,
        'alpha': alpha,
        'beta': beta,
        'beta_ci': (beta_lower, beta_upper)
    }


# ============================================================================
# PART 3 — CSV LOADING
# PART 3 — CSV 读取
# ============================================================================

def load_pe_csv(csv_dir):
    """
    Load all *_pe.csv files in directory.
    从目录读取所有 *_pe.csv 文件。
    """
    extracted = {}

    if not os.path.exists(csv_dir):
        return extracted

    for fname in os.listdir(csv_dir):
        if fname.endswith("_pe.csv"):
            ticker = fname.replace("_pe.csv", "")
            path = os.path.join(csv_dir, fname)

            df = pd.read_csv(path, parse_dates=['date'])
            extracted[ticker] = df

    return extracted


def calculate_ou_for_csv(csv_dir, dt=1):
    """
    Compute OU parameters for all tickers in a CSV folder.
    计算文件夹中所有标的的 OU 参数。
    """
    extracted = load_pe_csv(csv_dir)
    results = {}

    for ticker, df in extracted.items():
        if 'value' not in df.columns:
            continue

        series = df.set_index('date')['value']
        params = calculate_ou_params(series, dt=dt)

        if params:
            results[ticker] = params

    return results


# ============================================================================
# PART 4 — SINGLE TICKER INTERFACE
# PART 4 — 单标的接口
# ============================================================================

def get_ou_for_ticker(ticker, dt=1):
    """
    Load a single <ticker>_pe.csv and compute OU parameters.
    读取单个 <ticker>_pe.csv 并计算 OU 参数。
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    csv_path = os.path.join(project_root, "pe_csv", f"{ticker}_pe.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"PE CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, parse_dates=['date'])

    if 'value' not in df.columns:
        raise ValueError(f"CSV must contain 'value' column: {csv_path}")

    series = df.set_index('date')['value']
    ou = calculate_ou_params(series, dt=dt)

    if ou is None:
        raise ValueError(f"Not enough data for OU: {ticker}")

    return ou


# ============================================================================
# PART 5 — EXAMPLE
# PART 5 — 示例
# ============================================================================

if __name__ == "__main__":
    ticker = "NVDA"
    try:
        ou = get_ou_for_ticker(ticker)
        print(f"[{ticker}] λ = {ou['lambda']:.4f}, σ = {ou['sigma']:.4f}, 半衰期 = {ou['half_life']:.2f} 天")
    except Exception as e:
        print("Error:", e)

# [前文的 ols_regression_with_se, calculate_ou_params, load_pe_csv 保持不变...]

# ============================================================================
# PART 6 — STRATEGY OPTIMIZATION (NEW)
# PART 6 — 策略优化建议 (新增模块)
# ============================================================================

def suggest_strategy(ticker, ou_params, current_price=None, target_price=None):
    """
    Based on OU variance, suggest optimal expiry and structure.
    基于 OU 方差，建议最优行权日和结构。
    """
    print(f"\n>>> 策略优化建议报告: {ticker} <<<")

    # 1. 提取关键风控指标
    hl_est = ou_params['half_life']
    hl_max = ou_params['half_life_max_days']  # 95% 置信度下的最慢回归
    lambda_est = ou_params['lambda']

    # 2. 计算最优行权日 (Time vs Lambda Balance)
    # 逻辑：必须覆盖 3 倍的最慢半衰期，以确保 Lambda 有足够时间战胜 Theta
    # 如果 hl_max 极其长（说明回归不显著），则发出警告
    min_expiry_days = hl_max * 3

    print(f"1. 周期风控 (Time Decay Risk):")
    print(f"   - 平均回归半衰期: {hl_est:.1f} 天")
    print(f"   - 95%置信上限 (最慢): {hl_max:.1f} 天 <--- 关键风控点")
    print(f"   - 建议最小持仓时间: {min_expiry_days:.0f} 天 (约 {min_expiry_days/30:.1f} 个月)")

    # 3. 结构建议
    print(f"2. 结构选择 (Structure Selection):")
    if lambda_est > 3.0: # 回归极快
        print(f"   - 特征: 高 Lambda ({lambda_est:.2f}), 脉冲式回归。")
        print(f"   - 推荐: Vertical Spread (垂直价差)。")
        print(f"   - 理由: 速度够快，可以用 Spread 的高杠杆换取 Theta 损耗。")
    elif lambda_est < 1.0: # 回归慢
        print(f"   - 特征: 低 Lambda ({lambda_est:.2f}), 慢牛/磨蹭。")
        print(f"   - 推荐: LEAPS / PMCC。")
        print(f"   - 理由: 时间不确定性大，不要碰 Spread，用长久期抗过波动。")
    else:
        print(f"   - 特征: 中等回归速度。")
        print(f"   - 推荐: Diagonal Spread (对角价差) 或 宽幅垂直价差。")

    # 4. 具体行权日推荐
    import datetime
    today = datetime.date.today()
    target_date = today + datetime.timedelta(days=min_expiry_days)
    print(f"3. 建议行权日 (Expiry Selection):")
    print(f"   - 请寻找 [ {target_date.strftime('%Y-%m-%d')} ] 之后的期权合约")

    if current_price and target_price:
        print(f"4. 建议行权价 (Strike Selection):")
        print(f"   - Buy Leg (进攻): ${current_price:.0f} (ATM) 或 ${current_price*0.9:.0f} (ITM)")
        print(f"   - Sell Leg (防守): ${target_price:.0f} (Target V)")


# ============================================================================
# PART 7 — EXECUTION MAIN
# PART 7 — 执行入口
# ============================================================================

if __name__ == "__main__":
    ticker = "NVDA"

    # 这里可以手动输入当前价格和目标价，用于生成更具体的建议
    current_P = 178.88
    target_V = 225.00

    try:
        # 1. 计算统计参数
        print(f"正在计算 {ticker} 的 OU 参数 (含置信区间)...")
        ou = get_ou_for_ticker(ticker)

        # 2. 打印详细统计数据
        print("\n" + "="*50)
        print(f"  {ticker} Mean Reversion Statistics")
        print("="*50)
        print(f"  λ (Lambda)       : {ou['lambda']:.4f}  (95% CI: {ou['lambda_min']:.4f} - {ou['lambda_max']:.4f})")
        print(f"  σ (Sigma)        : {ou['sigma']:.4f}")
        print(f"  R² (拟合优度)    : {ou['r_squared']:.4f}")
        print("-" * 30)
        print(f"  半衰期 (Expected): {ou['half_life']:.2f} 天")
        print(f"  半衰期 (Pessimistic): {ou['half_life_max_days']:.2f} 天 (运气差时)")
        print(f"  半衰期 (Optimistic) : {ou['half_life_min_days']:.2f} 天 (运气好时)")
        print("="*50)

        # 3. 生成策略建议
        suggest_strategy(ticker, ou, current_price=current_P, target_price=target_V)
        print("="*50)

    except Exception as e:
        import traceback
        traceback.print_exc()
        print("\n[Error] 计算失败。请检查是否已生成 csv 文件 (运行 data extract.py) 且路径正确。")