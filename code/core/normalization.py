# code/core/normalization.py

import numpy as np
import pandas as pd

def normalize_portfolio(df_input, max_leverage_cap):
    """
    Performs Linear Weighted Normalization based on USER-DEFINED Confidence.

    Args:
        df_input (pd.DataFrame or list): Input data containing 'Raw_Kelly_Pct' and 'User_Confidence'.
        max_leverage_cap (float): The maximum total allocation limit.

    Returns:
        tuple: (df_result, total_raw_exposure, scale_factor)
    """
    # [修复]：兼容 List 类型输入，防止 AttributeError: 'list' object has no attribute 'empty'
    if isinstance(df_input, list):
        if not df_input:
            return pd.DataFrame(), 0.0, 0.0
        df = pd.DataFrame(df_input)
    elif isinstance(df_input, pd.DataFrame):
        if df_input.empty:
            return df_input, 0.0, 0.0
        df = df_input.copy()
    else:
        return pd.DataFrame(), 0.0, 0.0

    # 1. Calculate Raw Exposure
    if 'Raw_Kelly_Pct' not in df.columns:
        return df, 0.0, 0.0

    total_raw_exposure = df['Raw_Kelly_Pct'].sum()

    # 2. 计算加权分数 (Score)
    # 逻辑：原始想买的量 * 用户手动给的信心权重
    if 'User_Confidence' not in df.columns:
        df['User_Confidence'] = 1.0

    df['Score'] = df['Raw_Kelly_Pct'] * df['User_Confidence']
    total_score = df['Score'].sum()

    # 3. 分配逻辑：统一归一化
    # 如果 total_score > max_leverage_cap，则所有组之间按比例缩放
    # 如果 total_score <= max_leverage_cap，则不需要缩放（或者您可以选择是否要填满）

    if total_score == 0:
        df['Alloc_Calculated'] = 0.0
        scale_factor = 0.0
    else:
        # 核心公式：根据每个资产算出的“理想分值”占总分值的比例，来分配总上限
        # 这就实现了“组间归一化”
        if total_score > max_leverage_cap:
             scale_factor = max_leverage_cap / total_score
        else:
             scale_factor = 1.0 # 资金充足时不放大，保持原样

        df['Final_Pct'] = df['Score'] * scale_factor

    # 4. 边界处理：最终仓位通常不应超过“原始凯利建议”（除非您特意要把信心设得很高来突破限制）
    # 但按照标准逻辑，我们取 Min(缩放后的值, 原始值) 往往更安全，
    # 不过为了严格执行您的“归一化”逻辑，这里直接使用缩放后的值即可。
    # (如果需要保留 Raw 限制，取消下面这行的注释)
    # df['Final_Pct'] = df[['Final_Pct', 'Raw_Kelly_Pct']].min(axis=1)

    return df, total_raw_exposure, scale_factor