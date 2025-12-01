import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

# ==============================================================================
# Module 1: Black-Scholes Pricing Engine
# ==============================================================================
def bs_greek_calculator(S, K, T, r, sigma):
    """Calculate European call option Price, Delta, and Annualized Theta Cost"""
    if T <= 0.001:
        val = max(0.0, S - K)
        return val, 1.0 if S > K else 0.0, 0.0

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    delta = norm.cdf(d1)

    # Annualized theta cost (returned as positive value)
    term1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    term2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
    theta_annual_cost = -(term1 + term2)

    return price, delta, theta_annual_cost

# ==============================================================================
# Module 2: Single Asset Kelly Calculation
# ==============================================================================
def calculate_single_asset_kelly_ratio(
    P, option_price, delta, theta_annual,
    V, V_hard, lambda_annual, sigma_iv, r_f, beta=0.2
):
    if option_price <= 0.01: return 0.0

    # Leverage
    L = delta * (P / option_price)
    theta_rate = theta_annual / option_price

    # Equity Risk Premium (net advantage)
    mu_stock = lambda_annual * np.log(V / P)
    mu_leaps = mu_stock * L
    ERP_leaps = mu_leaps - r_f - theta_rate

    # Variance (risk)
    sigma_leaps = sigma_iv * L
    variance_leaps = sigma_leaps ** 2

    # Alpha (confidence discount based on distance from floor)
    range_len = max(1e-9, V - V_hard)
    dist_from_floor = P - V_hard
    risk_ratio = max(0.0, min(1.0, dist_from_floor / range_len))
    alpha_discount = 1.0 - (beta * risk_ratio)

    # Kelly Formula
    if ERP_leaps > 0 and variance_leaps > 0:
        f_cash = (alpha_discount * ERP_leaps) / variance_leaps
    else:
        f_cash = 0.0

    return max(0.0, f_cash)

# ==============================================================================
# Module 3: Dynamic Pilot Position Cap
# ==============================================================================
def calculate_pilot_cap(P, V_fill, Strike, T, r, sigma):
    """
    Calculate maximum initial position to enable 1:1 refill at V_fill
    """
    c0, _, _ = bs_greek_calculator(P, Strike, T, r, sigma)
    # Assume T unchanged after crash (stress test)
    c_fill, _, _ = bs_greek_calculator(V_fill, Strike, T, r, sigma)

    if c0 + c_fill <= 0: return 0.0
    return c0 / (c0 + c_fill)

# ==============================================================================
# Module 4: Main Solver & Data Generation
# ==============================================================================
def find_perfect_expiry(
    ticker, P, V_target, V_floor, V_fill,
    lambda_annual, sigma_asset, iv_pricing, r_f
):
    print(f"\n🔍 Finding perfect expiry for {ticker}...")
    print(f"   (Params: P=${P}, Target=${V_target}, Floor=${V_floor}, Fill=${V_fill})")

    results = []

    # Scan from 30 to 1100 days (~3 years)
    for days in range(30, 1100, 7):
        T = days / 365.0

        c_price, c_delta, c_theta = bs_greek_calculator(P, V_floor, T, r_f, iv_pricing)

        # Full Kelly (100%)
        kelly_full = calculate_single_asset_kelly_ratio(
            P, c_price, c_delta, c_theta,
            V_target, V_floor,
            lambda_annual, sigma_asset, r_f
        )

        # Target position (0.5 * Kelly)
        kelly_target = kelly_full * 0.5

        # Capital hard cap
        cap_limit = calculate_pilot_cap(P, V_fill, V_floor, T, r_f, iv_pricing)

        results.append({
            "Days": days,
            "Option_Price": c_price,
            "Kelly_Half": kelly_target,
            "Pilot_Cap": cap_limit,
            "Diff": kelly_target - cap_limit
        })

    df = pd.DataFrame(results)

    # Find intersection point (Diff closest to 0)
    best_idx = df['Diff'].abs().idxmin()
    best_row = df.loc[best_idx]

    print("\n✅ [FOUND] Perfect balance point:")
    print(f"   > Optimal expiry: {int(best_row['Days'])} days (~{best_row['Days']/30.4:.1f} months)")
    print(f"   > Suggested position: {best_row['Pilot_Cap']:.1%}")
    print(f"   > Option price: ${best_row['Option_Price']:.2f}")

    return best_row, df

# ==============================================================================
# 5. Run & Plot
# ==============================================================================
if __name__ == "__main__":
    # --- User Input ---
    TICKER = "NVDA"
    P_CURRENT = 182.14
    V_TARGET = 225.00
    V_HARD_FLOOR = 130.00   # Strike
    V_FILL_PLAN = 145.00    # Refill price

    LAMBDA = 4.46
    SIGMA_ASSET = 0.51
    IV_PRICING = 0.45
    R_RISKFREE = 0.041

    # Calculate data
    best, data = find_perfect_expiry(
        TICKER, P_CURRENT, V_TARGET, V_HARD_FLOOR, V_FILL_PLAN,
        LAMBDA, SIGMA_ASSET, IV_PRICING, R_RISKFREE
    )

    # Generate chart
    print("\n📊 Generating chart...")

    plt.figure(figsize=(12, 7))

    # Kelly curve (offense)
    plt.plot(data['Days'], data['Kelly_Half'], label='Offense: 0.5 * Kelly Ratio',
             color='blue', linewidth=2, linestyle='--')

    # Cap curve (defense)
    plt.plot(data['Days'], data['Pilot_Cap'], label='Defense: Pilot Cash Cap (for 1:1 refill)',
             color='red', linewidth=2)

    # Mark optimal point
    plt.scatter(best['Days'], best['Pilot_Cap'], color='green', s=150, zorder=5, label='Optimal Expiry')

    plt.annotate(
        f"Sweet Spot\n{int(best['Days'])} Days\n{best['Pilot_Cap']:.1%} Alloc",
        xy=(best['Days'], best['Pilot_Cap']),
        xytext=(best['Days']+100, best['Pilot_Cap']+0.1),
        arrowprops=dict(facecolor='black', shrink=0.05),
        fontsize=10, fontweight='bold'
    )

    plt.title(f"Optimal Expiry Solver: {TICKER}\n(Strike=${V_HARD_FLOOR}, Fill @ ${V_FILL_PLAN})", fontsize=14)
    plt.xlabel("Days to Expiration", fontsize=12)
    plt.ylabel("Position Allocation %", fontsize=12)
    plt.axhline(best['Pilot_Cap'], color='gray', linestyle=':', alpha=0.5)
    plt.axvline(best['Days'], color='gray', linestyle=':', alpha=0.5)

    plt.xticks(np.arange(0, 1100, 90))

    plt.legend(fontsize=12, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.show()