import sys
import os
import numpy as np
from scipy.stats import norm

# ===============================
# 1. Environment & Path Setup / 环境与路径设置
# ===============================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

# ===============================
# 2. Import Utility Modules / 导入工具模块
# ===============================
from utils.lambda_tools import get_ou_for_ticker
from utils.sigma_tools import get_sigma

# ===============================
# 3. Core Inputs / 核心参数输入
# ===============================
ticker = "NVDA"
total_capital = 100000.0  # Total capital in USD / 总资金

# --- Manual Option Chain Inputs / 手动录入期权链数据 ---
P = 182.14              # Stock Price / 正股价格
option_price = 64.63    # LEAPS Option Price / LEAPS期权价格
delta = 0.8460          # Delta
theta_daily_abs = 0.0432  # Theta (absolute daily decay) / Theta 日损耗

# --- Strategy Parameters / 策略参数 ---
V = 225.00              # Target Price / 目标价
V_hard = 130.00         # Hard Floor / 硬底
r_f = 0.041             # Risk-Free Rate / 无风险利率
beta = 0.2              # Valuation Discount Factor / 估值折扣系数
k = 1                   # Kelly Fraction / 凯利系数

# ===============================
# 4. Auto-fetch Statistical Data / 自动获取统计数据
# ===============================
print(f"Fetching data for {ticker}... / 正在获取 {ticker} 数据...")

# Lambda (OU Drift) / 回归速度
try:
    ou = get_ou_for_ticker(ticker)
    lambda_annual = ou["lambda"] * 252.0
except Exception:
    lambda_annual = 4.46  # Default for demo / 默认值演示

# Sigma (Historical Volatility) / 历史波动率
try:
    sigma_dict, _, _ = get_sigma([ticker], period="3y", annualize=True)
    sigma_stock_annual = sigma_dict[ticker]
except Exception:
    sigma_stock_annual = 0.5103  # Default 51.03% / 默认值

# ===============================
# 5. Core Calculations / 核心计算
# ===============================

# --- A. Leverage and Returns / 杠杆与收益 ---
L = delta * (P / option_price)                     # Effective leverage / 有效杠杆
theta_rate = (theta_daily_abs / option_price) * 252.0  # Annualized theta / 年化Theta

mu_stock = lambda_annual * np.log(V / P)  # Stock drift / 正股回归收益
mu_leaps = mu_stock * L
ERP_leaps = mu_leaps - r_f - theta_rate    # Excess return / 超额收益

# --- B. Volatility Calculation / 波动率计算 ---
sigma_leaps_annual = sigma_stock_annual * L  # LEAPS annualized volatility / 年化波动
variance_leaps = sigma_leaps_annual ** 2

sigma_leaps_daily = sigma_leaps_annual / np.sqrt(252)  # Daily volatility / 日波动率

# --- C. Kelly Position / 凯利仓位 ---
range_len = max(1e-9, V - V_hard)
risk_ratio = max(0.0, min(1.0, (P - V_hard) / range_len))
alpha_discount = 1.0 - (beta * risk_ratio)

f_cash = max(0.0, (k * alpha_discount * ERP_leaps) / variance_leaps) if ERP_leaps > 0 else 0.0
position_value = f_cash * total_capital
contracts = position_value / (option_price * 100)

# --- D. Account Volatility / 账户组合波动 ---
account_daily_vol = f_cash * sigma_leaps_daily          # Daily account volatility / 日账户波动率
account_daily_pnl = account_daily_vol * total_capital  # Daily PnL estimate / 日盈亏预期

# ===============================
# 6. English Output / 英文输出
# ===============================
print("\n" + "="*60)
print(f"📊 {ticker} LEAPS Risk Analysis / LEAPS 单资产风险分析")
print("="*60)

print(f"[1. Instrument Info / 资产属性]")
print(f"  - Option Price:        ${option_price:.2f}")
print(f"  - Effective Leverage:  {L:.2f}x")
print(f"  - Kelly Suggested:     {f_cash:.2%} (Cash ${position_value:,.0f})")

print("-" * 60)
print(f"[2. LEAPS Instrument Volatility / LEAPS 自身波动]")
print(f"  - Annualized Volatility: {sigma_leaps_annual:.2%}")
print(f"  - Daily Volatility:      {sigma_leaps_daily:.2%}")
print(f"  - Single Contract Daily Move: ${sigma_leaps_daily * option_price:.2f}")

print("-" * 60)
print(f"[3. Account Daily Risk / 账户单日风险]")
print(f"  - Account Daily Volatility: {account_daily_vol:.2%}")
print(f"  - Expected Daily PnL:      ±${account_daily_pnl:,.0f}")

print("-" * 60)
print(f"[4. Stress Scenarios / 极端场景推演]")

confidence_levels = [0.68, 0.95, 0.99]
labels = ["Normal Move (1σ)", "Monthly Drop (2σ)", "Extreme Crash (3σ)"]

print(f"\n  {'Scenario':<20} | {'LEAPS Drop':<15} | {'Account Loss':<15}")
print("  " + "-"*50)
for i, conf in enumerate(confidence_levels):
    z = norm.ppf(conf + (1-conf)/2)
    leaps_drop = min(z * sigma_leaps_daily, 1.0)
    account_loss = position_value * leaps_drop
    print(f"  {labels[i]:<20} | -{leaps_drop:<14.2%} | -${account_loss:,.0f}")

print("="*60)
if account_daily_vol > 0.05:
    print(f"⚠️ HIGH RISK: Account daily volatility ({account_daily_vol:.2%}) is very high.")
    print(f"   Daily loss could be ${account_daily_pnl:,.0f}.")
    print(f"   Suggestion: Lower k to reduce f_cash.")
else:
    print(f"✅ Risk is within normal high-risk asset range.")

# ===============================
# 7. Chinese Output / 中文输出
# ===============================
print("\n" + "="*60)
print(f"📊 {ticker} LEAPS 单资产风险分析 / LEAPS Risk Analysis")
print("="*60)

print(f"[1. 资产属性]")
print(f"  - 期权价格:         ${option_price:.2f}")
print(f"  - 有效杠杆:         {L:.2f} 倍")
print(f"  - 凯利建议仓位:     {f_cash:.2%} (金额 ${position_value:,.0f})")

print("-" * 60)
print(f"[2. LEAPS 自身的波动]")
print(f"  - 年化波动率: {sigma_leaps_annual:.2%}")
print(f"  - 日波动率:   {sigma_leaps_daily:.2%}")
print(f"  - 单张合约日波动: ${sigma_leaps_daily * option_price:.2f}")

print("-" * 60)
print(f"[3. 账户单日风险]")
print(f"  - 账户单日波动率: {account_daily_vol:.2%}")
print(f"  - 账户单日盈亏预期: ±${account_daily_pnl:,.0f}")

print("-" * 60)
print(f"[4. 极端场景推演]")
print(f"\n  {'场景':<20} | {'LEAPS 跌幅':<15} | {'账户回撤金额':<15}")
print("  " + "-"*50)
for i, conf in enumerate(confidence_levels):
    z = norm.ppf(conf + (1-conf)/2)
    leaps_drop = min(z * sigma_leaps_daily, 1.0)
    account_loss = position_value * leaps_drop
    print(f"  {labels[i]:<20} | -{leaps_drop:<14.2%} | -${account_loss:,.0f}")

print("="*60)
if account_daily_vol > 0.05:
    print(f"⚠️ 高风险提示: 您的账户单日波动 ({account_daily_vol:.2%}) 极高。")
    print(f"   一天内可能亏损 ${account_daily_pnl:,.0f}。")
    print(f"   建议：降低 k 值以控制仓位 f_cash。")
else:
    print(f"✅ 风险提示: 当前波动率处于高风险资产常规范围。")
print("="*60)
