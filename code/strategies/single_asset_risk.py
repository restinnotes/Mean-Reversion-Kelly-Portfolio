import sys
import os
import numpy as np
from scipy.stats import norm

# ===============================
# 1. Environment & Path Setup
# ===============================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

# ===============================
# 2. Import Utility Modules
# ===============================
from utils.lambda_tools import get_ou_for_ticker
from utils.sigma_tools import get_sigma

# ===============================
# 3. Core Inputs
# ===============================
ticker = "NVDA"
total_capital = 100000.0  # Total capital in USD

# --- Manual Option Chain Inputs ---
P = 182.14              # Stock Price
option_price = 64.63    # LEAPS Option Price
delta = 0.8460          # Delta
theta_daily_abs = 0.0432  # Theta (absolute daily decay)

# --- Strategy Parameters ---
V = 225.00              # Target Price
V_hard = 130.00         # Hard Floor
r_f = 0.041             # Risk-Free Rate (Annualized)
beta = 0.2              # Valuation Discount Factor
k = 1                   # Kelly Fraction

# ===============================
# 4. Auto-fetch Statistical Data
# ===============================
print(f"Fetching data for {ticker}...")

# Lambda (OU Drift) - Rolling 90-Day Logic
try:
    # Assuming lambda_tools supports 'window' parameter for PE regression
    ou = get_ou_for_ticker(ticker, window=90)
    lambda_annual = ou["lambda"] * 252.0
except Exception:
    lambda_annual = 4.46  # Default for demo

# Sigma (Historical Volatility) - Robust Percentile Logic
try:
    # Use robust parameters: percentile=0.85, safety_lock=True
    sigma_dict, _, _, _ = get_sigma(
        [ticker],
        period="5y",
        window=252,
        percentile=0.85,
        annualize=True,
        safety_lock=True
    )
    sigma_stock_annual = sigma_dict[ticker]
except Exception:
    sigma_stock_annual = 0.5103  # Default 51.03%

# ===============================
# 5. Core Calculations
# ===============================

# --- A. Leverage and Returns ---
L = delta * (P / option_price)                     # Effective leverage
theta_rate = (theta_daily_abs / option_price) * 252.0  # Annualized theta

mu_stock = lambda_annual * np.log(V / P)  # Stock drift
mu_leaps = mu_stock * L
ERP_leaps = mu_leaps - r_f - theta_rate    # Excess return

# --- B. Volatility Calculation ---
sigma_leaps_annual = sigma_stock_annual * L  # LEAPS annualized volatility
variance_leaps = sigma_leaps_annual ** 2

sigma_leaps_daily = sigma_leaps_annual / np.sqrt(252)  # Daily volatility

# --- C. Kelly Position ---
range_len = max(1e-9, V - V_hard)
risk_ratio = max(0.0, min(1.0, (P - V_hard) / range_len))
alpha_discount = 1.0 - (beta * risk_ratio)

f_cash = max(0.0, (k * alpha_discount * ERP_leaps) / variance_leaps) if ERP_leaps > 0 else 0.0
position_value = f_cash * total_capital
contracts = position_value / (option_price * 100)

# --- D. Account Volatility ---
account_daily_vol = f_cash * sigma_leaps_daily          # Daily account volatility
account_daily_pnl = account_daily_vol * total_capital  # Daily PnL estimate

# ===============================
# 6. Final Output (English Only)
# ===============================
print("\n" + "="*60)
print(f"📊 {ticker} LEAPS Risk Analysis")
print("="*60)

print(f"[1. Instrument Info]")
print(f"  - Option Price:        ${option_price:.2f}")
print(f"  - Effective Leverage:  {L:.2f}x")
print(f"  - Kelly Suggested:     {f_cash:.2%} (Cash ${position_value:,.0f})")

print("-" * 60)
print(f"[2. LEAPS Instrument Volatility]")
print(f"  - Annualized Volatility: {sigma_leaps_annual:.2%}")
print(f"  - Daily Volatility:      {sigma_leaps_daily:.2%}")
print(f"  - Single Contract Daily Move: ${sigma_leaps_daily * option_price:.2f}")

print("-" * 60)
print(f"[3. Account Daily Risk]")
print(f"  - Account Daily Volatility: {account_daily_vol:.2%}")
print(f"  - Expected Daily PnL:      ±${account_daily_pnl:,.0f}")

print("-" * 60)
print(f"[4. Stress Scenarios]")

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
print("="*60)