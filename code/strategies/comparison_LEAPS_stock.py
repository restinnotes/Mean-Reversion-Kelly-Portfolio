import sys
import os
import numpy as np

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
# 3. Inputs & Data Entry / 输入与数据录入
# ===============================
ticker = "NVDA"
total_capital = 100000.0

# -------------------------------------------------
# [Auto-Fetch] Core Statistical Parameters / 自动获取核心统计参数
# -------------------------------------------------
try:
    ou = get_ou_for_ticker(ticker)
    lambda_annual = ou["lambda"] * 252.0
except Exception as e:
    print(f"ERROR: Failed to fetch OU parameters: {e}")
    sys.exit(1)

try:
    sigma_dict, _, _ = get_sigma([ticker], period="3y", annualize=True)
    sigma_iv = sigma_dict[ticker]
except Exception as e:
    print(f"ERROR: Failed to fetch volatility: {e}")
    sys.exit(1)

# -------------------------------------------------
# [Manual Entry] Market Snapshot / 手动录入市场快照
# -------------------------------------------------
P = 182.14
option_price = 64.63
delta = 0.8460
theta_daily_abs = 0.0

# -------------------------------------------------
# [Strategy Parameters] / 策略参数
# -------------------------------------------------
V = 225.00
V_hard = 130.00
r_f = 0.041
beta = 0.2
k = 1  # 凯利比例暂时保留，用于资金分配，但主要逻辑用 Sharpe

# ===============================
# 4. Core Logic Calculation / 核心计算逻辑
# ===============================
mu_stock = lambda_annual * np.log(V / P)

# Alpha / Confidence / 信心水位
range_len = max(V - V_hard, 1e-9)
dist_from_floor = P - V_hard
risk_ratio = max(0.0, min(1.0, dist_from_floor / range_len))
alpha_discount = 1.0 - (beta * risk_ratio)

# --- Stock Path / 正股 ---
L_stock = 1.0
variance_stock = sigma_iv ** 2
ERP_stock = mu_stock - r_f
f_stock = max(0.0, (k * alpha_discount * ERP_stock) / variance_stock)
stock_cash = f_stock * total_capital
stock_shares = stock_cash / P

# --- LEAPS Path / LEAPS ---
L_leaps = delta * (P / option_price)
theta_leaps_annual = (theta_daily_abs / option_price) * 252.0
variance_leaps = (sigma_iv * L_leaps) ** 2
mu_leaps = mu_stock * L_leaps
ERP_leaps = mu_leaps - r_f - theta_leaps_annual
f_leaps = max(0.0, (k * alpha_discount * ERP_leaps) / variance_leaps)
leaps_cash = f_leaps * total_capital
leaps_contracts = leaps_cash / (option_price * 100.0)

# ===============================
# 5. Calculate Risk-Adjusted Efficiency / 计算风险调整后效率
# ===============================
# Volatility (Sigma)
sigma_stock = sigma_iv
sigma_leaps_approx = sigma_iv * L_leaps  # LEAPS近似波动率

# Sharpe Ratio (Efficiency) / 夏普比率 = ERP / Sigma
sharpe_stock = ERP_stock / sigma_stock if sigma_stock > 0 else 0
sharpe_leaps = ERP_leaps / sigma_leaps_approx if sigma_leaps_approx > 0 else 0

# ===============================
# 6. English Output / 英文输出
# ===============================
print("\n" + "="*60)
print(f"⚔️ Strategy Comparison: Stock vs. LEAPS ({ticker})")
print("="*60)
print(f"[Market Snapshot]")
print(f"  - Stock Price: ${P} -> Target: ${V}")
print(f"  - OU Drift (Mu): {mu_stock:.2%} (Lambda={lambda_annual:.2f})")
print(f"  - Confidence (Alpha): {alpha_discount:.3f}")
print("-" * 60)
print(f"{'Metric':<20} | {'Stock':<20} | {'LEAPS':<20}")
print("-" * 60)
print(f"{'Leverage (L)':<20} | {L_stock:<20.2f} | {L_leaps:<20.2f}")
print(f"{'Cost (Rf+Theta)':<20} | {r_f:<20.1%} | {(r_f + theta_leaps_annual):<20.1%}")
print(f"{'Expected Return':<20} | {mu_stock:<20.1%} | {mu_leaps:<20.1%}")
print(f"{'Net Advantage (ERP)':<20} | {ERP_stock:<20.1%} | {ERP_leaps:<20.1%}")
print(f"{'Risk (Sigma)':<20} | {sigma_stock:<20.2%} | {sigma_leaps_approx:<20.2%}")
print(f"{'Sharpe Ratio':<20} | {sharpe_stock:<20.4f} | {sharpe_leaps:<20.4f}")
print("-" * 60)

# Decision based on Sharpe / 基于夏普比率判断胜负
if sharpe_leaps > sharpe_stock:
    advantage = (sharpe_leaps / sharpe_stock - 1) * 100 if sharpe_stock > 0 else 999
    print(f"✅ 🏆 Winner: LEAPS")
    print(f"   Reason: Higher risk-adjusted return (Sharpe).")
    print(f"   LEAPS is {advantage:.1f}% more efficient than Stock per unit of risk.")
    print(f"   Action: Allocate {f_leaps:.1%} of capital to LEAPS (limited by volatility risk).")
elif sharpe_stock > sharpe_leaps:
    print(f"✅ 🏆 Winner: Stock")
    print(f"   Reason: LEAPS Theta cost drags down efficiency.")
    print(f"   Even with leverage, the risk-adjusted return is lower.")
    print(f"   Action: Buy Stock ({f_stock:.1%} allocation).")
else:
    print(f"⏸️ Draw / Hold. No clear statistical advantage.")
print("="*60)

# ===============================
# 7. Chinese Output / 中文输出
# ===============================
print("\n" + "="*60)
print(f"⚔️ 策略对比: 正股 vs. LEAPS ({ticker})")
print("="*60)
print(f"[市场快照]")
print(f"  - 股价: ${P} -> 目标: ${V}")
print(f"  - 回归动力 (Mu): {mu_stock:.2%} (Lambda={lambda_annual:.2f})")
print(f"  - 信心水位 (Alpha): {alpha_discount:.3f}")
print("-" * 60)
print(f"{'指标':<20} | {'正股':<20} | {'LEAPS':<20}")
print("-" * 60)
print(f"{'有效杠杆 (L)':<20} | {L_stock:<20.2f} | {L_leaps:<20.2f}")
print(f"{'持有成本 (Rf+Theta)':<20} | {r_f:<20.1%} | {(r_f + theta_leaps_annual):<20.1%}")
print(f"{'预期收益':<20} | {mu_stock:<20.1%} | {mu_leaps:<20.1%}")
print(f"{'净优势 (ERP)':<20} | {ERP_stock:<20.1%} | {ERP_leaps:<20.1%}")
print(f"{'风险 (Sigma)':<20} | {sigma_stock:<20.2%} | {sigma_leaps_approx:<20.2%}")
print(f"{'夏普比率':<20} | {sharpe_stock:<20.4f} | {sharpe_leaps:<20.4f}")
print("-" * 60)

if sharpe_leaps > sharpe_stock:
    print(f"✅ 🏆 胜者: LEAPS")
    print(f"   原因: 风险调整后的收益更高（夏普比率）。")
    print(f"   每单位风险下，LEAPS 的效率比正股高 {advantage:.1f}%。")
    print(f"   操作建议: 配置 {f_leaps:.1%} 资金于 LEAPS（受波动风险限制）。")
elif sharpe_stock > sharpe_leaps:
    print(f"✅ 🏆 胜者: 正股")
    print(f"   原因: LEAPS 的 Theta 成本降低了效率。")
    print(f"   即使杠杆放大，风险调整后收益仍低。")
    print(f"   操作建议: 买入正股 ({f_stock:.1%} 配置)。")
else:
    print(f"⏸️ 平手 / 观望。无明显统计优势。")
print("="*60)
