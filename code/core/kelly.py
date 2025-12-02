# code/core/kelly.py

import numpy as np
from .greek_tools import bs_greek_calculator, calculate_single_asset_kelly_ratio
import pandas as pd

def calculate_kelly_for_dashboard(
    P, V_target, V_hard, V_fill, opt_price, delta, theta, # Contract/Price Inputs
    lambda_val, sigma_val, r_f, beta, # Market/Strategy Params
    k_factor, k_fill, total_capital # Allocation Params
):
    """
    Calculates the final Kelly allocation (f_cash) and all intermediary metrics for Step 1 Dashboard.
    (Refactored from app_unified_zh.py:page_dashboard)
    """

    # 1. Core Metrics Calculation (using the input option values)
    L = delta * (P / opt_price) if opt_price > 0 else 0
    # Annualized Theta Decay Rate: theta (Daily Abs) / Price * 252
    theta_rate = (theta / opt_price) * 252.0 if opt_price > 0 else 0

    # 2. Returns & ERP
    mu_stock = lambda_val * np.log(V_target / P)
    mu_leaps = mu_stock * L
    ERP = mu_leaps - r_f - theta_rate

    # 3. Risk
    sigma_leaps = sigma_val * L
    variance_leaps = sigma_leaps ** 2

    # 4. Alpha
    range_len = max(1e-9, V_target - V_hard)
    dist_from_floor = P - V_hard
    risk_ratio = max(0.0, min(1.0, dist_from_floor / range_len))
    alpha = 1.0 - (beta * risk_ratio)

    # 5. Kelly Cash (Raw)
    # Kelly_Raw = (Alpha * ERP) / Variance (where k=1.0)
    kelly_ratio_raw = (alpha * ERP) / variance_leaps if (ERP > 0 and variance_leaps > 0) else 0.0

    # 6. Final Allocation (k-factor and cap logic)
    current_k = k_factor
    if P <= V_fill:
        # When price hits or breaches V_fill, use k_fill
        current_k = k_fill

    f_cash = kelly_ratio_raw * current_k
    f_cash = min(1.0, max(0.0, f_cash)) # Final cap at 100%

    # 7. Contracts Calculation
    contract_cost = opt_price * 100
    if contract_cost > 0:
        target_contracts_float = (f_cash * total_capital) / contract_cost
        target_contracts = int(target_contracts_float)
    else:
        target_contracts = 0
        target_contracts_float = 0.0

    return {
        'f_cash': f_cash,
        'target_contracts': target_contracts,
        'target_contracts_float': target_contracts_float,
        'contract_cost': contract_cost,
        'ERP': ERP,
        'L': L,
        'alpha': alpha,
        'sigma_leaps': sigma_leaps,
        'k_factor_used': current_k,
        'kelly_ratio_raw': kelly_ratio_raw
    }


def calculate_dynamic_kelly_path(
    P_CURRENT, V_TARGET, V_HARD, V_FILL,
    lambda_val, sigma_val, r_f, beta,
    k_factor, k_fill, total_capital,
    days_to_expiry, iv_pricing, num_points=100
):
    """
    Calculates the full dynamic Kelly allocation path (allocation % and K value)
    from V_hard to V_target.
    (Extracted from app_unified_zh.py:page_dashboard)
    """
    T_year = days_to_expiry / 365.0
    sim_prices = np.linspace(V_HARD, V_TARGET * 1.05, num_points)
    allocations = []
    k_values = []
    contracts_series = []

    for p_sim in sim_prices:
        # --- A. Dynamic K Factor Calculation ---
        k_dynamic = k_factor
        if p_sim <= P_CURRENT:
            # Downside: Linear interpolation for K from k_factor (Start) to k_fill (Max)
            progress = (P_CURRENT - p_sim) / max(1e-9, (P_CURRENT - V_FILL))
            progress_clamped = min(1.0, max(0.0, progress))

            k_dynamic = k_factor + (k_fill - k_factor) * progress_clamped

            if p_sim < V_FILL:
                k_dynamic = k_fill
        else:
            # Upside: Maintain initial K (k_factor)
            k_dynamic = k_factor

        k_values.append(k_dynamic)

        # --- B. Full Dynamic Kelly Calculation ---
        # NOTE: We use the full, proper Kelly calculation for each price point p_sim
        from .greek_tools import bs_greek_calculator # Local import to avoid circular dependency issues
        c_sim, d_sim, t_val_abs = bs_greek_calculator(p_sim, V_HARD, T_year, r_f, iv_pricing)

        kelly_ratio_raw = calculate_single_asset_kelly_ratio(
            p_sim, c_sim, d_sim, t_val_abs, # t_val_abs is theta_daily_abs
            V_TARGET, V_HARD, lambda_val, sigma_val, r_f, beta=beta
        )

        final_alloc = kelly_ratio_raw * k_dynamic

        # Cap logic
        if p_sim <= V_FILL:
            final_alloc = min(1.0, final_alloc)
        else:
            final_alloc = min(1.0, final_alloc)

        final_alloc = max(0.0, final_alloc)
        allocations.append(final_alloc)

        # Calculate contracts at this price point
        cost_sim = c_sim * 100
        if cost_sim > 0:
            num_c = (final_alloc * total_capital) / cost_sim
        else:
            num_c = 0
        contracts_series.append(num_c)

    return sim_prices, allocations, k_values, contracts_series


def calculate_grid_signals(sim_prices, contracts_series, current_contracts, P_CURRENT):
    """
    Calculates grid trading advice (buy/sell points).
    (Extracted from app_unified_zh.py:page_dashboard)

    Returns:
        tuple: (buy_points, sell_points, step_size)
    """

    # Determine step size
    max_c = max(contracts_series) if contracts_series else 0
    if max_c > 50:
         step_size = max(1, int(max_c / 20))
    else:
        step_size = 1

    idx_current = np.argmin(np.abs(sim_prices - P_CURRENT))

    # BUYING: Iterate from current price index down to V_HARD (index 0)
    buy_points = []
    for i in range(idx_current, -1, -1):
        p_val = sim_prices[i]
        c_val = contracts_series[i]

        next_threshold = current_contracts + (len(buy_points) + 1) * step_size

        if c_val >= next_threshold:
             target_hold = current_contracts + (len(buy_points) + 1) * step_size
             buy_points.append({'price': p_val, 'target_hold': target_hold, 'step': step_size})
             if len(buy_points) >= 3: break

    # SELLING: Iterate from current price index up to V_TARGET
    sell_points = []
    for i in range(idx_current, len(sim_prices)):
        p_val = sim_prices[i]
        c_val = contracts_series[i]

        next_threshold = current_contracts - (len(sell_points) + 1) * step_size

        if c_val <= next_threshold and next_threshold >= 0:
            target_hold = current_contracts - (len(sell_points) + 1) * step_size
            sell_points.append({'price': p_val, 'target_hold': target_hold, 'step': step_size})
            if len(sell_points) >= 3: break

    return buy_points, sell_points, step_size