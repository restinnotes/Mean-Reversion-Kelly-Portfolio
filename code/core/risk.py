# code/core/risk.py

import numpy as np
import pandas as pd

def calculate_stress_test(f_cash, leverage, sigma_val, total_capital):
    """
    Calculates stress test scenarios based on Kelly allocation, leverage, and volatility.
    (Extracted from app_unified_zh.py:page_dashboard)
    """
    # 1. Get Daily Sigma for Stock
    sigma_daily_stock = sigma_val / np.sqrt(252)

    # 2. Define Scenarios (Drop in Stock Price)
    scenarios = [
        ("日常波动 (1σ)", -1.0 * sigma_daily_stock),
        ("周度回调 (2σ)", -2.0 * sigma_daily_stock),
        ("极端黑天鹅 (3σ)", -3.0 * sigma_daily_stock),
        ("熔断级崩盘 (-20%)", -0.20)
    ]

    risk_table = []

    # Use Delta Approximation: LEAPS Drop % ≈ Leverage * Stock Drop %
    NOMINAL_ACCOUNT_VALUE = total_capital
    L = leverage

    for name, stock_drop in scenarios:
        if L == 0:
            leaps_drop_pct = 0.0
        else:
            # Use effective leverage L for approximation
            leaps_drop_pct = stock_drop * L

        # Account Impact = Kelly_Pct * Leaps_Drop_Pct
        account_impact_pct = f_cash * leaps_drop_pct
        account_loss_usd = account_impact_pct * NOMINAL_ACCOUNT_VALUE

        risk_table.append({
            "情景": name,
            "标的跌幅": stock_drop,
            "LEAPS 预估跌幅": leaps_drop_pct,
            "账户总净值回撤": account_impact_pct,
            "预估亏损": account_loss_usd,
        })

    return pd.DataFrame(risk_table)