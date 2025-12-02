import sys
import os
import numpy as np
import pandas as pd

# ===============================
# 1. Environment & Path Setup
# ===============================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root_for_import = os.path.abspath(os.path.join(current_dir, ".."))
if project_root_for_import not in sys.path:
    sys.path.append(project_root_for_import)

# Import Utilities (Assumes utils folder is correctly set up relative to project root)
from utils.lambda_tools import get_ou_for_ticker
from utils.sigma_tools import get_sigma

# ===============================
# 2. Global Strategy Settings
# ===============================
TOTAL_CAPITAL = 100000.0   # 总账户价值
RISK_FREE_RATE = 0.041     # 4.1% 无风险利率
MAX_LEVERAGE_CAP = 1.00    # C_max: 最大现金分配上限 (1.0 = 100%)
DEFAULT_KELLY_K = 0.5      # 全局默认 K 值 (如果单个标的未指定)

# ===============================
# 3. Portfolio Configuration (现在支持可选的 'k_factor')
# ===============================
PORTFOLIO = [
    {
        "ticker": "NVDA",
        "P": 182.14,            # 当前股价
        "V_target": 225.00,     # 目标价
        "V_hard": 130.00,       # 硬底
        "opt_price": 64.63,     # LEAPS 价格
        "delta": 0.846,         # Delta
        "theta": 0.0432,        # 每日绝对 Theta 损耗
        "beta": 0.2,            # 估值折扣系数
        "k_factor": 0.7         # <- 个性化 K 值 (例如，对深度低估的标的提高至 0.7)
    },
    {
        "ticker": "MSFT",
        "P": 415.00,
        "V_target": 480.00,
        "V_hard": 350.00,
        "opt_price": 85.50,
        "delta": 0.82,
        "theta": 0.0350,
        "beta": 0.2,
        # 未指定 k_factor，将使用 DEFAULT_KELLY_K = 0.5
    },
    {
        "ticker": "META",
        "P": 590.00,
        "V_target": 650.00,
        "V_hard": 450.00,
        "opt_price": 120.00,
        "delta": 0.85,
        "theta": 0.0550,
        "beta": 0.25,
        "k_factor": 0.4         # <- 个性化 K 值 (例如，对涨幅较大的标的降低至 0.4)
    }
]

# ===============================
# 4. Calculation Engine
# ===============================

def calculate_kelly_for_asset(item, lam, sig):
    """Calculates the raw Kelly fraction for a single asset."""

    P = item['P']
    V = item['V_target']
    V_h = item['V_hard']
    L_price = item['opt_price']
    delta = item['delta']
    theta_abs = item['theta']
    beta = item['beta']

    # 核心改进: 使用个性化 K 值，否则使用全局默认 K 值
    k = item.get('k_factor', DEFAULT_KELLY_K)

    # --- Core Metrics Calculation ---
    Lev = delta * (P / L_price)
    theta_rate = (theta_abs / L_price) * 252.0
    mu_stock = lam * np.log(V / P)
    mu_leaps = mu_stock * Lev
    ERP = mu_leaps - RISK_FREE_RATE - theta_rate
    sigma_leaps = sig * Lev
    variance = sigma_leaps ** 2

    # Confidence (Alpha)
    range_len = max(1e-9, V - V_h)
    dist_floor = max(0, P - V_h)
    risk_ratio = min(1.0, dist_floor / range_len)
    alpha = 1.0 - (beta * risk_ratio)

    # Raw Kelly Fraction: f_raw = k * (Alpha * ERP) / Variance
    if ERP > 0:
        f_raw = (k * alpha * ERP) / variance
    else:
        f_raw = 0.0

    f_raw = max(0.0, f_raw)

    return {
        "Ticker": item['ticker'],
        "ERP": ERP,
        "Alpha": alpha,
        "Lev": Lev,
        "K_Factor_Used": k,
        "Raw_Kelly_Pct": f_raw,
        "Raw_Cash": f_raw * TOTAL_CAPITAL
    }

def run_portfolio_normalization_v2():
    print(f"\n{'='*70}")
    print(f"🌌 多资产凯利优化 (V2: 基于个性化 K 值计算后归一化)")
    print(f"账户总资本: ${TOTAL_CAPITAL:,.0f} | 归一化上限: {MAX_LEVERAGE_CAP:.2%}")
    print(f"{'='*70}\n")

    # --- Step A: Batch Fetch Volatility ---
    tickers = [item['ticker'] for item in PORTFOLIO]
    print(f"[数据获取] 正在获取稳健波动率 (Sigma) for: {tickers}...")
    try:
        sigma_dict, _, _, _ = get_sigma(
            tickers, period="5y", window=252, percentile=0.85, annualize=True, safety_lock=True
        )
    except Exception as e:
        print(f"获取波动率失败: {e}")
        return

    results = []

    # --- Step B: Loop Calculation (Raw Kelly with Per-Asset K) ---
    print(f"\n[单标的计算] 使用个性化 K 值进行理论仓位计算...")
    for item in PORTFOLIO:
        t = item['ticker']

        # 1. Get Lambda (OU Process)
        try:
            ou = get_ou_for_ticker(t, window=90)
            lam = ou["lambda"] * 252.0
        except FileNotFoundError:
             print(f"警告: 找不到 {t} 的 PE CSV 文件，无法计算 Lambda。跳过。")
             continue
        except Exception as e:
            print(f"警告: {t} 计算 Lambda 失败: {e}。使用默认值 3.0。")
            lam = 3.0

        # 2. Get Sigma
        sig = sigma_dict.get(t)
        if sig is None:
            print(f"警告: 未找到 {t} 的 Sigma。跳过。")
            continue

        # 3. Calculate
        result = calculate_kelly_for_asset(item, lam, sig)
        results.append(result)

    if not results:
        print("未计算出有效仓位，程序结束。")
        return

    df = pd.DataFrame(results)

    # --- Step C: Normalization Logic (归一化) ---
    total_raw_exposure = df['Raw_Kelly_Pct'].sum()

    print(f"\n[归一化分析]")
    print(f"原始总 Kelly 理论仓位 (C_raw): {total_raw_exposure:.2%}")

    # Calculate Scaling Factor
    if total_raw_exposure > MAX_LEVERAGE_CAP:
        scale_factor = MAX_LEVERAGE_CAP / total_raw_exposure
        print(f"⚠️ 总仓位超过上限。缩放系数 (Factor): {scale_factor:.4f}")
    else:
        scale_factor = 1.0
        print(f"✅ 总仓位在限制内。不进行缩放 (Factor: 1.0000)。")

    # Apply Normalization
    df['Final_Pct'] = df['Raw_Kelly_Pct'] * scale_factor
    df['Final_Cash'] = df['Final_Pct'] * TOTAL_CAPITAL

    # Calculate Contracts
    price_map = {item['ticker']: item['opt_price'] for item in PORTFOLIO if item['ticker'] in df['Ticker'].values}
    opt_prices = df['Ticker'].apply(lambda x: price_map[x]).values
    df['Contracts'] = df['Final_Cash'] / (opt_prices * 100)

    # --- Step D: Display Results ---
    print("\n" + "="*80)
    print("🚀 最终投资组合分配 (基于个性化 K 值和简单归一化)")
    print("="*80)

    # Formatting for display
    display_df = df[['Ticker', 'K_Factor_Used', 'Lev', 'ERP', 'Raw_Kelly_Pct', 'Final_Pct', 'Final_Cash', 'Contracts']].copy()
    display_df.rename(columns={'K_Factor_Used': 'K_Used', 'Raw_Kelly_Pct': 'Raw %', 'Final_Pct': 'Final %', 'Final_Cash': 'Final Cash'}, inplace=True)

    display_df['K_Used'] = display_df['K_Used'].apply(lambda x: f"{x:.2f}")
    display_df['Lev'] = display_df['Lev'].apply(lambda x: f"{x:.2f}x")
    display_df['ERP'] = display_df['ERP'].apply(lambda x: f"{x:.2%}")
    display_df['Raw %'] = display_df['Raw %'].apply(lambda x: f"{x:.2%}")
    display_df['Final %'] = display_df['Final %'].apply(lambda x: f"{x:.2%}")
    display_df['Final Cash'] = display_df['Final Cash'].apply(lambda x: f"${x:,.0f}")
    display_df['Contracts'] = display_df['Contracts'].apply(lambda x: f"{x:.1f}")

    print(display_df.to_string(index=False))
    print("-" * 80)
    print(f"总现金占用: {df['Final_Pct'].sum():.2%} (${df['Final_Cash'].sum():,.0f})")
    print("="*80)

if __name__ == "__main__":
    # Ensure constants are defined when running directly
    if 'DEFAULT_KELLY_K' not in locals():
        DEFAULT_KELLY_K = 0.5
        RISK_FREE_RATE = 0.041

    run_portfolio_normalization_v2()