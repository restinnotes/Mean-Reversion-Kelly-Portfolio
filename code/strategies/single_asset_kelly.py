import sys
import os
import numpy as np

# ===============================
# 1. Environment & Path Setup
# ===============================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

# ===============================
# 2. Import Utility Modules
# ===============================
# Used to get lambda (Regression Power)
from utils.lambda_tools import get_ou_for_ticker
# Used to get sigma (Real Volatility)
from utils.sigma_tools import get_sigma

# ===============================
# 3. Inputs & Data Entry
# ===============================
ticker = "NVDA"
total_capital = 100000.0  # Total Capital

# -------------------------------------------------
# [Auto-Fetch] Core Statistical Parameters
# -------------------------------------------------

# 1. Get OU Regression Parameters (Using 90-Day Rolling Window for fresh market state)
rolling_window = 90
try:
    ou = get_ou_for_ticker(ticker, window=rolling_window)
    # Convert daily lambda to annualized
    lambda_annual = ou["lambda"] * 252.0
    print(f"SUCCESS: OU Parameters Fetched (Rolling {rolling_window}d). Annualized Lambda = {lambda_annual:.4f}")
except Exception as e:
    print(f"ERROR: Failed to fetch OU parameters: {e}")
    sys.exit(1)

# 2. Get Historical Volatility (For risk Sigma)
try:
    # Use robust parameters: percentile=0.85 (85th percentile safety), safety_lock=True
    sigma_dict, _, _, _ = get_sigma(
        [ticker],
        period="5y",
        window=252,
        percentile=0.85,
        annualize=True,
        safety_lock=True
    )
    # Get Annualized Volatility for NVDA
    sigma_iv = sigma_dict[ticker]
    print(f"SUCCESS: YF Volatility Fetched. Annualized Sigma = {sigma_iv:.2%}")
except Exception as e:
    print(f"ERROR: Failed to fetch volatility: {e}")
    sys.exit(1)

# -------------------------------------------------
# [Manual Entry] Market Snapshot (Option Chain)
# -------------------------------------------------
P = 182.14             # Current Stock Price
option_price = 64.63   # LEAPS Price
delta = 0.8460         # Option Delta
theta_daily_abs = 0.0432 # Daily Theta (Absolute Value)

# -------------------------------------------------
# [Strategy Parameters] Targets & Risk Control
# -------------------------------------------------
V = 225.00             # Target Price (Fair Value)
V_hard = 130.00        # Hard Floor Price
r_f = 0.041            # Risk-free Rate (Annualized 4.1%)
beta = 0.2             # Valuation Discount Coeff
k = 1.0                # Kelly Fraction (1.0 = Full Kelly)

# ===============================
# 4. Core Logic Calculation (V23.2)
# ===============================

# --- A. Leverage & Cost ---

# Effective Leverage
L = delta * (P / option_price)

# Annualized Theta Decay Rate
theta_rate = (theta_daily_abs / option_price) * 252.0

# --- B. Expected Return & Net Edge (ERP) ---

# Stock Expected Annual Return (Based on OU)
mu_stock = lambda_annual * np.log(V / P)

# LEAPS Expected Annual Return (Leveraged)
mu_leaps = mu_stock * L

# LEAPS Net Edge (ERP) = Return - Capital Cost - Time Rent
# Logic: All annualized, direct subtraction
ERP_leaps = mu_leaps - r_f - theta_rate

# --- C. Risk Calculation (Variance) ---

# LEAPS Volatility = Stock Vol * Leverage
sigma_leaps = sigma_iv * L

# Kelly Denominator: Variance
variance_leaps = sigma_leaps ** 2
# Core Correction: Risk scales with Leverage Squared

# --- D. Confidence Level (Alpha) ---

# Logic: Closer to floor -> Alpha near 1.0; Closer to Target -> Alpha decreases
range_len = V - V_hard
dist_from_floor = P - V_hard

if range_len <= 1e-9: range_len = 1e-9
risk_ratio = dist_from_floor / range_len
risk_ratio = max(0.0, min(1.0, risk_ratio)) # Limit to 0~1

# Calculate Discount Coefficient
alpha_discount = 1.0 - (beta * risk_ratio)

# ===============================
# 5. Kelly Cash Allocation
# ===============================

# Formula: Cash% = k * (Alpha * ERP) / Variance
# Only open position if ERP is positive
if ERP_leaps > 0:
    f_cash = (k * alpha_discount * ERP_leaps) / variance_leaps
else:
    f_cash = 0.0

f_cash = max(0.0, f_cash)

# Amount & Contracts
cash_amt = f_cash * total_capital
contract_cost = option_price * 100.0
contracts = cash_amt / contract_cost

# ===============================
# 6. Final Output (English Only)
# ===============================
print("\n" + "="*60)
print(f"🚀 {ticker} LEAPS Strategy Calculator (V23.2 Rolling-Lambda)")
print("="*60)

print(f"[1. Market Snapshot]")
print(f"  - Price P:          ${P}")
print(f"  - Target V:         ${V} (Hard Floor ${V_hard})")
print(f"  - Option Price:     ${option_price} (Delta={delta}, Theta=${theta_daily_abs})")

print(f"\n[2. Statistical Params (Auto-Fetched)]")
print(f"  - Regression Lambda: {lambda_annual:.2f} (90-Day Rolling Window)")
print(f"  - Real Sigma:        {sigma_iv:.2%} (Robust 85%ile)")

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
