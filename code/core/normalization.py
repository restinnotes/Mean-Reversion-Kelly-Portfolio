# code/core/normalization.py

import numpy as np
import pandas as pd

def normalize_portfolio(portfolio_data, max_leverage_cap):
    """
    Performs simple linear normalization on a portfolio of assets based on their raw Kelly Pct.
    (Refactored from app_unified_zh.py:page_multi_asset_normalization)

    Args:
        portfolio_data (list[dict]): List of assets with 'Raw_Kelly_Pct', 'ERP', 'L', etc.
        max_leverage_cap (float): The maximum total allocation limit (e.g., 1.0 for 100%).

    Returns:
        tuple: (df_normalized, total_raw_exposure, scale_factor)
    """
    if not portfolio_data:
        return pd.DataFrame(), 0.0, 0.0

    df = pd.DataFrame(portfolio_data)

    # 1. Calculate Raw Exposure
    total_raw_exposure = df['Raw_Kelly_Pct'].sum()

    # 2. Normalize Logic
    if total_raw_exposure > max_leverage_cap:
        scale_factor = max_leverage_cap / total_raw_exposure
    else:
        scale_factor = 1.0

    # 3. Apply Normalization
    df['Final_Pct'] = df['Raw_Kelly_Pct'] * scale_factor

    return df, total_raw_exposure, scale_factor