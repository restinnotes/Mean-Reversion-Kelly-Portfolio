# code/data/rolling.py

import pandas as pd
import numpy as np
import os
import sys

from .lambda_tools import calculate_ou_params, calculate_historical_ma_reversion

def run_rolling_analysis(ticker, project_root, window_days=90):
    """
    Performs rolling OU parameter estimation and robust historical verification.
    (Modified to remove plotting logic and return data structure for UI layer).

    Returns:
        dict: {
            'df': full PE DataFrame,
            'rolling_df': DataFrame of rolling metrics (PE, MA, Lambda, HL),
            'current_metrics': dict of final calculated values,
            'robust_stats': dict of historical structural stats
        }
    """
    csv_path = os.path.join(project_root, "pe_csv", f"{ticker}_pe.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data not found: {csv_path}")

    df = pd.read_csv(csv_path, parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)

    if len(df) < window_days:
        return None

    # Context Data
    df['rolling_mean'] = df['value'].rolling(window=window_days).mean()

    dates = []; pe_values = []; pe_means = []; lambdas_annual = []; half_lives = []; sigmas_daily = []

    start_index = window_days - 1

    for i in range(start_index, len(df)):
        window_data = df.iloc[i-window_days+1 : i+1]
        series = window_data.set_index('date')['value']

        ou = calculate_ou_params(series)

        if ou:
            dates.append(df.iloc[i]['date'])
            pe_values.append(df.iloc[i]['value'])
            pe_means.append(df.iloc[i]['rolling_mean'])
            lambdas_annual.append(ou['lambda'] * 252)
            half_lives.append(ou['half_life'])
            sigmas_daily.append(ou['sigma'])

    if not lambdas_annual: return None

    # 1. Rolling Metrics DataFrame
    rolling_df = pd.DataFrame({
        'date': dates,
        'value': pe_values,
        'rolling_mean': pe_means,
        'Lambda': lambdas_annual,
        'Half_Life': half_lives,
        'Sigma_Daily': sigmas_daily
    }).set_index('date')

    # 2. Current Metrics
    current_metrics = {
        'current_lambda': lambdas_annual[-1],
        'current_hl': half_lives[-1],
        'current_pe': pe_values[-1],
        'current_mean': pe_means[-1],
        'current_sigma_daily': sigmas_daily[-1]
    }

    # 3. Robust Historical Verification
    full_series_for_robust = df.set_index('date')['value'].sort_index()
    robust_stats = calculate_historical_ma_reversion(full_series_for_robust, window=window_days)

    return {
        'df': df, # Full original dataframe
        'rolling_df': rolling_df,
        'current_metrics': current_metrics,
        'robust_stats': robust_stats
    }