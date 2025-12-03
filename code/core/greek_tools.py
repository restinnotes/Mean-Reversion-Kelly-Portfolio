# code/core/greek_tools.py

import numpy as np
from scipy.stats import norm

def bs_greek_calculator(S, K, T, r, sigma):
    """
    Calculate European call option Price, Delta, and Absolute Daily Theta (loss).
    (Refactored from optimal_expiry_solver.py)
    """
    if T <= 0.001:
        val = max(0.0, S - K)
        return val, 1.0 if S > K else 0.0, 0.0

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    delta = norm.cdf(d1)

    # Absolute Daily Theta Decay
    # BS formula gives Annual Theta. We divide by 252 to get Trading Daily Theta.
    term1 = (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    term2 = r * K * np.exp(-r * T) * norm.cdf(d2)
    theta_annual_abs = term1 + term2

    # === 修复: 将年化 Theta 转换为日 Theta (除以 252) ===
    theta_daily_abs = theta_annual_abs / 252.0

    return price, delta, theta_daily_abs


def calculate_single_asset_kelly_ratio(
    P, option_price, delta, theta_daily_abs,
    V, V_hard, lambda_annual, sigma_asset, r_f, beta=0.2
):
    """
    Calculate the raw Kelly ratio (f_cash) before applying k_factor.
    (Refactored from optimal_expiry_solver.py and app_unified_zh.py)
    """
    if option_price <= 0.01: return 0.0

    # Leverage (L)
    L = delta * (P / option_price)

    # Annualized Theta Decay Rate (theta_rate)
    # Assumes input theta_daily_abs is per trading day (1/252 year)
    theta_rate = (theta_daily_abs / option_price) * 252.0

    # Equity Risk Premium (ERP)
    mu_stock = lambda_annual * np.log(V / P)
    mu_leaps = mu_stock * L
    ERP_leaps = mu_leaps - r_f - theta_rate

    # Variance (Risk)
    sigma_leaps = sigma_asset * L
    variance_leaps = sigma_leaps ** 2

    # Alpha (Confidence Discount)
    range_len = max(1e-9, V - V_hard)
    dist_from_floor = P - V_hard
    risk_ratio = max(0.0, min(1.0, dist_from_floor / range_len))
    alpha_discount = 1.0 - (beta * risk_ratio)

    # Kelly Formula: f_cash = (Alpha * ERP) / Variance (where k=1.0)
    if ERP_leaps > 0 and variance_leaps > 0:
        f_cash = (alpha_discount * ERP_leaps) / variance_leaps
    else:
        f_cash = 0.0

    return max(0.0, f_cash)