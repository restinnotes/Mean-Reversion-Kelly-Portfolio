# code/core/normalization.py

import numpy as np
import pandas as pd

def normalize_portfolio(df_input, max_leverage_cap):
    """
    执行简化的分层加权全局归一化。

    核心公式：
    Final_Pct = Raw_Kelly × Group_Confidence × User_Confidence /
                (Sum_User_Confidence_in_Group × Sum_All_Group_Confidence × Scale_Factor)

    Args:
        df_input (pd.DataFrame or list): 输入数据，需包含以下列：
            - Raw_Kelly_Pct: 原始凯利建议仓位
            - User_Confidence: 用户对单个资产的信心权重
            - Group_Confidence: 资产所属分组的信心权重
            - Group (可选): 分组标识
        max_leverage_cap (float): 最大总仓位上限

    Returns:
        tuple: (df_result, total_raw_exposure, scale_factor)
    """
    # 输入数据类型兼容处理
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

    # 检查必需列
    required_cols = ['Raw_Kelly_Pct']
    if not all(col in df.columns for col in required_cols):
        return df, 0.0, 0.0

    # 设置默认值
    if 'User_Confidence' not in df.columns:
        df['User_Confidence'] = 1.0
    if 'Group_Confidence' not in df.columns:
        df['Group_Confidence'] = 1.0
    if 'Group' not in df.columns:
        df['Group'] = 'default'

    # 计算原始总敞口
    total_raw_exposure = df['Raw_Kelly_Pct'].sum()

    # === 核心公式：一步到位计算 ===
    # Final_Pct = Raw_Kelly × Group_Confidence × User_Confidence /
    #             (Sum_User_Confidence_in_Group × Sum_All_Group_Confidence × Scale_Factor)

    # 1. 计算每个组内的用户信心权重总和
    df['Sum_User_Confidence_in_Group'] = df.groupby('Group')['User_Confidence'].transform('sum')

    # 2. 计算所有组的组信心权重总和（每个唯一组只计算一次）
    sum_all_group_confidence = df.groupby('Group')['Group_Confidence'].first().sum()
    sum_all_group_confidence = sum_all_group_confidence if sum_all_group_confidence > 0 else 1.0

    # 3. 先计算未缩放的分数
    df['Unscaled_Score'] = (df['Raw_Kelly_Pct'] * df['Group_Confidence'] * df['User_Confidence']) / \
                           (df['Sum_User_Confidence_in_Group'] * sum_all_group_confidence)

    # 处理除零情况
    df['Unscaled_Score'] = df['Unscaled_Score'].fillna(0)

    # 4. 计算总需求（用于确定缩放系数）
    total_unscaled = df['Unscaled_Score'].abs().sum()

    # 5. 计算缩放系数
    if total_unscaled == 0:
        scale_factor = 0.0
        df['Final_Pct'] = 0.0
    else:
        scale_factor = min(1.0, max_leverage_cap / total_unscaled)
        df['Final_Pct'] = df['Unscaled_Score'] * scale_factor

    # 保存中间结果用于调试
    df['Sum_All_Group_Confidence'] = sum_all_group_confidence
    df['Scale_Factor'] = scale_factor

    return df, total_raw_exposure, scale_factor


# === 示例使用 ===
if __name__ == "__main__":
    # 示例数据：3个组，每组2个资产
    sample_data = [
        {'Asset': 'A1', 'Group': 'Tech', 'Raw_Kelly_Pct': 0.15, 'User_Confidence': 0.8, 'Group_Confidence': 1.5},
        {'Asset': 'A2', 'Group': 'Tech', 'Raw_Kelly_Pct': 0.10, 'User_Confidence': 0.6, 'Group_Confidence': 1.5},
        {'Asset': 'B1', 'Group': 'Finance', 'Raw_Kelly_Pct': 0.20, 'User_Confidence': 0.9, 'Group_Confidence': 1.0},
        {'Asset': 'B2', 'Group': 'Finance', 'Raw_Kelly_Pct': 0.08, 'User_Confidence': 0.5, 'Group_Confidence': 1.0},
        {'Asset': 'C1', 'Group': 'Energy', 'Raw_Kelly_Pct': 0.12, 'User_Confidence': 0.7, 'Group_Confidence': 0.8},
        {'Asset': 'C2', 'Group': 'Energy', 'Raw_Kelly_Pct': 0.18, 'User_Confidence': 1.0, 'Group_Confidence': 0.8},
    ]

    result_df, raw_exp, scale = normalize_portfolio(sample_data, max_leverage_cap=0.5)

    print("=== 分层加权全局归一化结果 ===")
    print(f"\n原始总敞口: {raw_exp:.4f}")
    print(f"全局缩放系数: {scale:.4f}")
    print(f"\n详细计算过程:")
    print(result_df[['Asset', 'Group', 'Raw_Kelly_Pct', 'User_Confidence',
                     'Group_Confidence', 'Sum_User_Confidence_in_Group',
                     'Unscaled_Score', 'Final_Pct']].to_string(index=False))
    print(f"\n最终总仓位: {result_df['Final_Pct'].sum():.4f}")