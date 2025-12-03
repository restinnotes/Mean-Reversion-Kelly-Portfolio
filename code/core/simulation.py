# code/core/simulation.py

import numpy as np
import pandas as pd

def run_simulation(current_pe, target_pe, lambda_annual, sigma_daily, days_to_simulate=252, num_paths=10000):
    """
    Runs Monte Carlo simulation for P/E ratio using the Ornstein-Uhlenbeck process.
    (Extracted from app_unified_zh.py)
    """
    dt = 1/252
    paths = np.zeros((days_to_simulate + 1, num_paths))
    paths[0] = current_pe
    theta = target_pe

    for t in range(1, days_to_simulate + 1):
        X_t = paths[t-1]
        drift = lambda_annual * (theta - X_t) * dt
        shock = sigma_daily * np.random.normal(0, 1, num_paths)
        paths[t] = X_t + drift + shock
    return paths

def analyze_risk_reward(paths, current_pe, days_map):
    """
    计算 Hold (持有到底) 和 Touch (触碰高点) 的风险收益分布
    (Extracted from app_unified_zh.py)
    """
    results = []
    max_sim_days = paths.shape[0] - 1

    for label, day in days_map.items():
        if day > max_sim_days: continue

        # --- A. HOLD 逻辑 (持有到底) ---
        final_values = paths[day]
        # 1. 亏损概率
        prob_loss = np.mean(final_values < current_pe)
        # 2. 10% 底线 (Worst Case)
        worst_10_val = np.percentile(final_values, 10)
        worst_10_pnl = (worst_10_val - current_pe) / current_pe
        # 3. 预期收益
        expected_val = np.mean(final_values)
        expected_pnl = (expected_val - current_pe) / current_pe

        # --- B. TOUCH 逻辑 (触碰高点) ---
        # 路径切片: [0..day]
        path_slice = paths[:day+1, :]
        # 每条路径在期间的最高点
        max_values = np.max(path_slice, axis=0)
        # 4. 10% 高点 (Best Case / Lucky Case)
        lucky_10_val = np.percentile(max_values, 90)
        lucky_10_pnl = (lucky_10_val - current_pe) / current_pe

        results.append({
            "时间窗口": label,
            "亏损概率 (Loss%)": prob_loss,
            "10%底线 (Hold)": worst_10_pnl,
            "预期收益 (Exp)": expected_pnl,
            "10%高点 (Touch)": lucky_10_pnl
        })

    return pd.DataFrame(results)