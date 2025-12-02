# code/core/solver.py

import numpy as np
import pandas as pd
from .greek_tools import bs_greek_calculator, calculate_single_asset_kelly_ratio

def find_optimal_expiry(
    P_CURRENT, V_TARGET, V_HARD_FLOOR, V_FILL_PLAN,
    LAMBDA, SIGMA_ASSET, IV_PRICING, R_RISKFREE,
    K_FACTOR, k_fill_target, BETA
):
    """
    Solves for the optimal contract expiration date (Days) that satisfies
    kelly_alloc_at_fill == 100% when using k_fill_target.
    (Extracted and refactored from app_unified_zh.py:page_solver)

    Returns:
        tuple: (best_row_dict, df_results)
    """
    if V_FILL_PLAN >= P_CURRENT:
        return None, None

    results = []
    IV_PRICING_SOLVER = IV_PRICING # Use the IV from input

    # Scan from 90 days to avoid volatile short-term structures
    for days in range(90, 1100, 7):
        T = days / 365.0

        # A. 计算【当前】状态 (P_CURRENT, k=K_FACTOR)
        c_price, c_delta, c_theta_daily_abs = bs_greek_calculator(P_CURRENT, V_HARD_FLOOR, T, R_RISKFREE, IV_PRICING_SOLVER)

        kelly_full_now = calculate_single_asset_kelly_ratio(
            P_CURRENT, c_price, c_delta, c_theta_daily_abs,
            V_TARGET, V_HARD_FLOOR, LAMBDA, SIGMA_ASSET, R_RISKFREE, beta=BETA
        )
        kelly_alloc_now = kelly_full_now * K_FACTOR  # Apply Start K

        # B. 计算【补仓】状态 (V_FILL_PLAN, k=k_fill_target)
        c_fill_price, c_fill_delta, c_fill_theta_daily_abs = bs_greek_calculator(V_FILL_PLAN, V_HARD_FLOOR, T, R_RISKFREE, IV_PRICING_SOLVER)

        kelly_full_at_fill = calculate_single_asset_kelly_ratio(
            V_FILL_PLAN, c_fill_price, c_fill_delta, c_fill_theta_daily_abs,
            V_TARGET, V_HARD_FLOOR, LAMBDA, SIGMA_ASSET, R_RISKFREE, beta=BETA
        )
        kelly_alloc_at_fill = kelly_full_at_fill * k_fill_target # Apply Target K

        # C. 记录结果
        diff = abs(kelly_alloc_at_fill - 1.0)

        results.append({
            "Days": days,
            "Kelly_Now": kelly_alloc_now,
            "Kelly_At_Fill": kelly_alloc_at_fill,
            "Diff_From_100": diff,
            "Price_Now": c_price
        })

    df = pd.DataFrame(results)

    if df.empty:
        return None, None

    # --- 寻找最优解 ---
    best_idx = df['Diff_From_100'].idxmin()
    best_row = df.loc[best_idx].to_dict()

    return best_row, df


def calculate_dynamic_k_path(P_CURRENT, V_FILL_PLAN, K_FACTOR, k_fill_target, T_best, V_HARD_FLOOR, V_TARGET, LAMBDA, SIGMA_ASSET, R_RISKFREE, BETA, IV_PRICING, num_points=50):
    """
    Calculates the dynamic allocation path based on price drop and linearly interpolated K value.
    (Extracted from app_unified_zh.py:page_solver)
    """
    sim_prices = np.linspace(P_CURRENT, V_FILL_PLAN, num_points)
    sim_allocations = []
    sim_ks = []

    for p in sim_prices:
        # 1. 动态计算当前的 K 值 (线性插值)
        # progress: 0.0 (Top) -> 1.0 (Bottom)
        progress = (P_CURRENT - p) / max(1e-9, (P_CURRENT - V_FILL_PLAN))
        k_dynamic = K_FACTOR + (k_fill_target - K_FACTOR) * progress

        # 2. 计算期权和凯利
        c, d, t_val_abs = bs_greek_calculator(p, V_HARD_FLOOR, T_best, R_RISKFREE, IV_PRICING)

        kelly_ratio_raw = calculate_single_asset_kelly_ratio(
            p, c, d, t_val_abs, # t_val_abs is theta_daily_abs
            V_TARGET, V_HARD_FLOOR, LAMBDA, SIGMA_ASSET, R_RISKFREE, beta=BETA
        )

        final_alloc = kelly_ratio_raw * k_dynamic
        sim_allocations.append(final_alloc)
        sim_ks.append(k_dynamic)

    return sim_prices, sim_allocations, sim_ks