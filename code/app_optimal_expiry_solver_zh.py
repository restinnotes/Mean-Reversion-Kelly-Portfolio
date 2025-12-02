import streamlit as st
import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm # 仅用于兼容性，实际英文模式下不再需要

# ===============================
# 1. SETUP: Path & Imports
# ===============================
current_dir = os.path.dirname(os.path.abspath(__file__))
# Add 'code/strategies' directory to Python search path
strategies_dir = os.path.join(current_dir, "strategies")
if strategies_dir not in sys.path:
    sys.path.append(strategies_dir)

# Import the core solver functions
try:
    from optimal_expiry_solver import find_perfect_expiry
except ImportError as e:
    st.error(f"Module import failed: {e}. Ensure optimal_expiry_solver.py is located in the code/strategies/ directory.")
    st.stop()

# ===============================
# 1.1 MATPLOTLIB FONT CONFIGURATION (REMOVED - Use English for plot text)
# ===============================
# 移除中文配置，防止 Matplotlib 警告。图表文字将使用默认英文。
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Verdana']
plt.rcParams['axes.unicode_minus'] = True
# ===============================
# End Font Configuration
# ===============================


# ===============================
# 2. Streamlit App Layout
# ===============================
st.set_page_config(page_title="选品辅助 - 最优期限求解器", layout="wide", page_icon="🎯")
st.title("🎯 Step 0.5: 选品辅助 - 最优期限求解器 (Optimal Expiry Solver)")

st.sidebar.header("参数输入")

# --- Default Values (NVDA example from optimal_expiry_solver.py) ---
DEFAULT_PARAMS = {
    "TICKER": "NVDA",
    "P_CURRENT": 182.14,
    "V_TARGET": 225.00,
    "V_HARD_FLOOR": 130.00,
    "V_FILL_PLAN": 145.00,
    "LAMBDA": 4.46,
    "SIGMA_ASSET": 0.51,
    "IV_PRICING": 0.45,
    "R_RISKFREE": 0.041,
}

# --- Sidebar Inputs: Asset ---
st.sidebar.subheader("资产与估值")
ticker = st.sidebar.text_input("股票代码 (Ticker)", value=DEFAULT_PARAMS['TICKER'])
P_CURRENT = st.sidebar.number_input("当前股价 P ($)", value=DEFAULT_PARAMS['P_CURRENT'], format="%.2f")
V_TARGET = st.sidebar.number_input("目标价 V_target ($)", value=DEFAULT_PARAMS['V_TARGET'], format="%.2f")
V_HARD_FLOOR = st.sidebar.number_input("硬底 (Strike) V_hard ($)", value=DEFAULT_PARAMS['V_HARD_FLOOR'], format="%.2f")
V_FILL_PLAN = st.sidebar.number_input("计划补仓价 V_fill ($)", value=DEFAULT_PARAMS['V_FILL_PLAN'], format="%.2f")

# --- Sidebar Inputs: Statistical ---
st.sidebar.subheader("统计与风险参数")
LAMBDA = st.sidebar.number_input("年化 Lambda (λ)", value=DEFAULT_PARAMS['LAMBDA'], format="%.4f", help="从 Step 0 诊断结果获取")
SIGMA_ASSET = st.sidebar.number_input("标的真实波动率 (σ)", value=DEFAULT_PARAMS['SIGMA_ASSET'], format="%.4f", help="从 Step 1 波动率计算器获取")
IV_PRICING = st.sidebar.number_input("期权定价波动率 (IV)", value=DEFAULT_PARAMS['IV_PRICING'], format="%.4f", help="用于 Black-Scholes 定价")
R_RISKFREE = st.sidebar.number_input("无风险利率 (r_f)", value=DEFAULT_PARAMS['R_RISKFREE'], format="%.4f")


if st.sidebar.button("运行最优期限求解", type="primary"):
    st.session_state['run_solver'] = True
else:
    st.session_state['run_solver'] = False

# --- Main Content Execution ---
if st.session_state.get('run_solver', False):

    st.subheader("⚠️ 注意: 当前求解器使用原代码内置的默认 K=0.5 和 Beta=0.2 进行计算。")

    try:
        # Call the core solver function
        best, data = find_perfect_expiry(
            ticker, P_CURRENT, V_TARGET, V_HARD_FLOOR, V_FILL_PLAN,
            LAMBDA, SIGMA_ASSET, IV_PRICING, R_RISKFREE
        )

        st.success("✅ 最优期限计算完成。")

        st.subheader("分析结果")
        col_r1, col_r2, col_r3 = st.columns(3)

        with col_r1:
            st.metric("最优期限", f"{int(best['Days'])} 天", f"~{best['Days']/30.4:.1f} 月")
        with col_r2:
            st.metric("建议分配比例 (Cap)", f"{best['Pilot_Cap']:.2%}")
        with col_r3:
            st.metric("期权价格 (BS 估值)", f"${best['Option_Price']:.2f}")

        st.markdown("---")
        st.markdown("##### 攻守平衡曲线图")
        st.caption(f"最优解为进攻曲线 (0.5 * Kelly) 与防守上限 (Pilot Cash Cap) 的交点。")

        # Plotting - ALL PLOT TEXT IN ENGLISH
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(data['Days'], data['Kelly_Half'], label='Offense: 0.5 * Kelly Ratio', # 英文标签
                 color='blue', linewidth=2, linestyle='--')

        ax.plot(data['Days'], data['Pilot_Cap'], label='Defense: Pilot Cash Cap (1:1 Refill)', # 英文标签
                 color='red', linewidth=2)

        # Mark optimal point
        ax.scatter(best['Days'], best['Pilot_Cap'], color='green', s=150, zorder=5, label='Optimal Expiry') # 英文标签

        ax.annotate(
            f"Sweet Spot\n{int(best['Days'])} Days\n{best['Pilot_Cap']:.1%} Alloc", # 英文注释
            xy=(best['Days'], best['Pilot_Cap']),
            xytext=(best['Days']+100, best['Pilot_Cap']+0.1),
            arrowprops=dict(facecolor='black', shrink=0.05),
            fontsize=10, fontweight='bold'
        )

        ax.set_title(f"Optimal Expiry Solver: {ticker} (Strike=${V_HARD_FLOOR}, Refill @ ${V_FILL_PLAN})", fontsize=14) # 英文标题
        ax.set_xlabel("Days to Expiration", fontsize=12) # 英文 X 轴标签
        ax.set_ylabel("Position Allocation %", fontsize=12) # 英文 Y 轴标签
        ax.axhline(best['Pilot_Cap'], color='gray', linestyle=':', alpha=0.5)
        ax.axvline(best['Days'], color='gray', linestyle=':', alpha=0.5)

        ax.set_xticks(np.arange(0, 1100, 180))

        ax.legend(fontsize=12, loc='best')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

        st.markdown("---")
        st.warning("⚠️ **下一步行动**：将最优期限对应的 **真实期权价格**、**Delta** 和 **Theta**，回填到主仓位计算器 (`app_dashboard_zh.py`) 中。")


    except Exception as e:
        st.error(f"运行求解器时发生错误: {e}")

st.info("请在左侧栏输入资产信息和统计参数（Lambda, Sigma, IV），然后点击运行。")
st.caption("注：此工具执行 '使用指南' 中 Step 0.5 的功能。")