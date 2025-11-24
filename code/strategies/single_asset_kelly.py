import sys
import os
import numpy as np

# ===============================
# 1. Environment & Path Setup
# 1. 环境与路径设置
# ===============================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

# ===============================
# 2. Import Utility Modules
# 2. 导入工具模块
# ===============================
# Used to get lambda (Regression Power) / 用于获取 lambda (回归动力)
from utils.lambda_tools import get_ou_for_ticker
# Used to get sigma (Real Volatility) / 用于获取 sigma (真实波动率)
from utils.sigma_tools import get_sigma

# ===============================
# 3. Inputs & Data Entry
# 3. 基础参数与数据录入
# ===============================
ticker = "NVDA"
total_capital = 100000.0  # Total Capital / 总资金

# -------------------------------------------------
# [Auto-Fetch] Core Statistical Parameters
# [自动获取] 核心统计参数
# -------------------------------------------------

# 1. Get OU Regression Parameters (For expected return mu)
# 1. 获取 OU 回归参数 (用于计算预期收益 mu)
try:
    ou = get_ou_for_ticker(ticker)
    # Convert daily lambda to annualized / 将日频 lambda 转为年化
    lambda_annual = ou["lambda"] * 252.0
    print(f"SUCCESS: OU Parameters Fetched. Annualized Lambda = {lambda_annual:.4f}")
    print(f"成功: 获取 OU 参数。年化 Lambda = {lambda_annual:.4f}")
except Exception as e:
    print(f"ERROR: Failed to fetch OU parameters: {e}")
    print(f"错误: 获取 OU 参数失败: {e}")
    sys.exit(1)

# 2. Get Historical Volatility (For risk Sigma)
# 2. 获取正股历史波动率 (用于计算风险 Sigma)
# Using sigma_tools from YFinance / 使用 sigma_tools 从 YFinance 获取
try:
    # get_sigma returns (sigma_dict, corr_matrix, cov_matrix)
    sigma_dict, _, _ = get_sigma([ticker], period="3y", annualize=True)
    # Get Annualized Volatility for NVDA (e.g., 0.45) / 获取 NVDA 的年化波动率
    sigma_iv = sigma_dict[ticker]
    print(f"SUCCESS: YF Volatility Fetched. Annualized Sigma = {sigma_iv:.2%}")
    print(f"成功: 获取 YF 波动率。年化 Sigma = {sigma_iv:.2%}")
except Exception as e:
    print(f"ERROR: Failed to fetch volatility: {e}")
    print(f"错误: 获取波动率失败: {e}")
    sys.exit(1)

# -------------------------------------------------
# [Manual Entry] Market Snapshot (Option Chain)
# [手动录入] 市场实时快照 (期权链数据)
# -------------------------------------------------
P = 182.14             # Current Stock Price / 正股现价
option_price = 64.63   # LEAPS Price / LEAPS 期权价格
delta = 0.8460         # Option Delta / 期权 Delta
theta_daily_abs = 0.0432 # Daily Theta (Absolute Value) / 期权日 Theta (绝对值)

# -------------------------------------------------
# [Strategy Parameters] Targets & Risk Control
# [策略参数] 目标与风控
# -------------------------------------------------
V = 225.00             # Target Price (Fair Value) / 目标价
V_hard = 130.00        # Hard Floor Price / 硬底
r_f = 0.041            # Risk-free Rate (Annualized 4.1%) / 无风险利率
beta = 0.2             # Valuation Discount Coeff / 估值折扣系数 (水位高时的减仓力度)
k = 1.0                # Kelly Fraction (1.0 = Full Kelly) / 凯利系数

# ===============================
# 4. Core Logic Calculation (V23.1)
# 4. 核心逻辑计算 (V23.1 修正版)
# ===============================

# --- A. Leverage & Cost / 杠杆与成本 ---

# Effective Leverage / 有效杠杆
L = delta * (P / option_price)

# Annualized Theta Decay Rate / 年化 Theta 损耗率
theta_rate = (theta_daily_abs / option_price) * 252.0

# --- B. Expected Return & Net Edge (ERP) / 预期收益与净优势 ---

# Stock Expected Annual Return (Based on OU) / 正股预期年化收益
mu_stock = lambda_annual * np.log(V / P)

# LEAPS Expected Annual Return (Leveraged) / LEAPS 预期年化收益
mu_leaps = mu_stock * L

# LEAPS Net Edge (ERP) = Return - Capital Cost - Time Rent
# LEAPS 净优势 (ERP) = 收益 - 资金成本 - 时间租金
# Logic: All annualized, direct subtraction / 逻辑：全部为年化比率，直接相减
ERP_leaps = mu_leaps - r_f - theta_rate

# --- C. Risk Calculation (Variance) / 风险计算 ---

# LEAPS Volatility = Stock Vol * Leverage / LEAPS 波动率
sigma_leaps = sigma_iv * L

# Kelly Denominator: Variance
# 凯利公式分母：方差
# Core Correction: Risk scales with Leverage Squared / 核心修正：风险随杠杆平方级放大
variance_leaps = sigma_leaps ** 2

# --- D. Confidence Level (Alpha) / 信心水位 ---

# Logic: Closer to floor -> Alpha near 1.0; Closer to Target -> Alpha decreases
# 逻辑：股价离地板越近，Alpha 越接近 1.0；离目标越近，Alpha 越小
range_len = V - V_hard
dist_from_floor = P - V_hard

if range_len <= 1e-9: range_len = 1e-9
risk_ratio = dist_from_floor / range_len
risk_ratio = max(0.0, min(1.0, risk_ratio)) # Limit to 0~1

# Calculate Discount Coefficient / 计算折扣系数
alpha_discount = 1.0 - (beta * risk_ratio)

# ===============================
# 5. Kelly Cash Allocation
# 5. 凯利现金仓位计算
# ===============================

# Formula: Cash% = k * (Alpha * ERP) / Variance
# Only open position if ERP is positive / 只有当 ERP 为正时才开仓
if ERP_leaps > 0:
    f_cash = (k * alpha_discount * ERP_leaps) / variance_leaps
else:
    f_cash = 0.0

f_cash = max(0.0, f_cash)

# Amount & Contracts / 金额与张数
cash_amt = f_cash * total_capital
contract_cost = option_price * 100.0
contracts = cash_amt / contract_cost

# ===============================
# 6. Bilingual Output Report
# 6. 双语结果输出
# ===============================

# -----------------------------
# English Report
# -----------------------------
print("\n" + "="*60)
print(f"🚀 {ticker} LEAPS Strategy Calculator (V23.1 Auto-Vol)")
print("="*60)

print(f"[1. Market Snapshot]")
print(f"  - Price P:          ${P}")
print(f"  - Target V:         ${V} (Hard Floor ${V_hard})")
print(f"  - Option Price:     ${option_price} (Delta={delta}, Theta=${theta_daily_abs})")

print(f"\n[2. Statistical Params (Auto-Fetched)]")
print(f"  - Regression Lambda: {lambda_annual:.2f} (OU Fit)")
print(f"  - Real Sigma:        {sigma_iv:.2%} (YFinance 3Y)")

print(f"\n[3. LEAPS Core Attributes]")
print(f"  - Eff. Leverage L:   {L:.2f}x")
print(f"  - Cost of Capital:   {r_f:.1%}")
print(f"  - Time Decay Theta:  {theta_rate:.2%} (Annualized)")
print(f"  - Total Risk SigmaL: {sigma_leaps:.2%} (Variance {variance_leaps:.2f})")

print(f"\n[4. Strategy Verdict]")
print(f"  - Exp. Drift:        {mu_leaps:.2%} (Stock {mu_stock:.2%})")
print(f"  - Net Edge (ERP):    {ERP_leaps:.2%} (After Rf & Theta)")
print(f"  - Risk Level:        {risk_ratio:.1%} (Alpha Discount = {alpha_discount:.3f})")

print("-" * 60)
print(f"[5. Kelly Suggestion]")
print(f"  > Allocation %:      {f_cash:.2%}")
print(f"  > Cash Amount:       ${cash_amt:,.0f}")
print(f"  > Contracts:         {contracts:.2f}")
print("=" * 60)

# -----------------------------
# Chinese Report / 中文报告
# -----------------------------
print("\n" + "="*60)
print(f"🚀 {ticker} LEAPS 策略计算器 (V23.1 自动波动率版)")
print("="*60)

print(f"[1. 市场快照]")
print(f"  - 股价 P:           ${P}")
print(f"  - 目标 V:           ${V} (硬底 ${V_hard})")
print(f"  - 期权 Price:       ${option_price} (Delta={delta}, Theta=${theta_daily_abs})")

print(f"\n[2. 统计参数 (自动获取)]")
print(f"  - 回归速度 Lambda:  {lambda_annual:.2f} (来自 OU 拟合)")
print(f"  - 真实波动 Sigma:   {sigma_iv:.2%} (来自 YFinance 3年数据)")

print(f"\n[3. LEAPS 核心属性]")
print(f"  - 有效杠杆 L:       {L:.2f}x")
print(f"  - 资金成本 Rf:      {r_f:.1%}")
print(f"  - 时间损耗 Theta:   {theta_rate:.2%} (年化)")
print(f"  - 综合风险 SigmaL:  {sigma_leaps:.2%} (方差 {variance_leaps:.2f})")

print(f"\n[4. 策略判定]")
print(f"  - 预期收益 Drift:   {mu_leaps:.2%} (正股 {mu_stock:.2%})")
print(f"  - 净优势 ERP:       {ERP_leaps:.2%} (扣除 Rf & Theta)")
print(f"  - 当前水位 Risk:    {risk_ratio:.1%} (Alpha折扣系数 = {alpha_discount:.3f})")

print("-" * 60)
print(f"[5. 凯利建议 (Kelly Criterion)]")
print(f"  > 建议仓位比例:     {f_cash:.2%}")
print(f"  > 建议现金投入:     ${cash_amt:,.0f}")
print(f"  > 建议购买张数:     {contracts:.2f} 张")
print("=" * 60)