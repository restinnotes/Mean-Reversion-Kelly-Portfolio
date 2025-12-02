# code/ui/zh/page_solver.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# Import Core modules - 修复后的导入
from core.solver import find_optimal_expiry, calculate_dynamic_k_path

def render_page_solver(P_CURRENT, V_TARGET, V_HARD_FLOOR, V_FILL_PLAN, LAMBDA, SIGMA_ASSET, IV_PRICING, R_RISKFREE, ticker, K_FACTOR, BETA):
    st.title("🎯 Step 0.5: 最优期限求解 (动态 K 值版)")
    st.subheader(f"资产: {ticker} | 目标: 在 {V_FILL_PLAN} 时打满子弹")
    st.markdown("---")

    # --- K Factor Inputs ---
    col_k1, col_k2 = st.columns(2)
    with col_k1:
        st.metric("起始 K 值 (Start)", f"{K_FACTOR:.2f}", help="当前左侧边栏设定的 K 值")
    with col_k2:
        k_fill_target = st.session_state.get('k_fill', 1.0)
        st.metric("满仓 K 值 (Target at Fill)", f"{k_fill_target:.2f}", help="当股价跌到 V_fill 时，你愿意使用多大的 K 值？")


    # --- 1. 策略配置区 (新增) ---
    with st.expander("❓ 什么是“动态 K 值”求解？", expanded=True):
        st.markdown(f"""
            **核心思想**：
            通常我们在建仓时比较谨慎（使用较小的 $k$，如 0.5），但随着股价下跌，安全边际变大，我们的信心会增强（使用较大的 $k$，如 1.0）。

            **本工具的目标**：
            寻找一张合约，使得：
            1.  **现在 ($P={P_CURRENT}$)**：应用 **起始 K={K_FACTOR}** 时，仓位适中。
            2.  **到底 ($P={V_FILL_PLAN}$)**：应用 **最终 K={k_fill_target}** 时，建议仓位 **恰好为 100%**。

            这样你就能设计出一个“越跌越买，到底正好满仓”的完美加仓路径。
        """)

    st.markdown("---")

    if V_FILL_PLAN >= P_CURRENT:
        st.error(f"错误：补仓价 V_fill ({V_FILL_PLAN}) 必须低于当前价格 ({P_CURRENT})。")
        return

    # --- 2. 求解循环 ---
    best_row, df_results = find_optimal_expiry(
        P_CURRENT, V_TARGET, V_HARD_FLOOR, V_FILL_PLAN,
        LAMBDA, SIGMA_ASSET, IV_PRICING, R_RISKFREE,
        K_FACTOR, k_fill_target, BETA
    )

    if best_row is None:
        st.warning("无法计算。请检查参数。")
        return

    # --- 3. 寻找最优解 ---
    if best_row['Diff_From_100'] > 0.1:
        st.warning(f"⚠️ 未找到完美匹配。最接近的合约在满仓时仓位为 {best_row['Kelly_At_Fill']:.2%}。")
    else:
        st.success(f"✅ 找到完美合约！期限 **{int(best_row['Days'])} 天**。")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("推荐合约期限", f"{int(best_row['Days'])} 天", f"~{best_row['Days']/30.4:.1f} 月")
    with col2:
        st.metric("当前建仓 (Start K)", f"{best_row['Kelly_Now']:.2%}", f"k={K_FACTOR}")
    with col3:
        st.metric("触底仓位 (Target K)", f"{best_row['Kelly_At_Fill']:.2%}", f"k={k_fill_target}")

    T_best = best_row['Days'] / 365.0

    # --- 4. 动态路径推演 (Dynamic K Simulation) ---
    st.markdown("---")
    st.subheader("📉 动态 K 值加仓路径推演")
    st.caption(f"模拟：股价下跌，K 值从 {K_FACTOR} 线性增加至 {k_fill_target}。")

    # The calculation is now in core.solver.calculate_dynamic_k_path
    sim_prices, sim_allocations, sim_ks = calculate_dynamic_k_path(
        P_CURRENT, V_FILL_PLAN, K_FACTOR, k_fill_target, T_best,
        V_HARD_FLOOR, V_TARGET, LAMBDA, SIGMA_ASSET, R_RISKFREE, BETA, IV_PRICING
    )

    # 绘图
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # 左轴：仓位
    ax1.plot(sim_prices, sim_allocations, color='#1f77b4', linewidth=3, label='建议仓位 %')
    ax1.set_xlabel("股价 (模拟下跌)", fontsize=12)
    ax1.set_ylabel("建议仓位", color='#1f77b4', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax1.axhline(1.0, color='red', linestyle='--', alpha=0.5, label='100% 满仓线')
    ax1.invert_xaxis() # 从高到低

    # 右轴：K值
    ax2 = ax1.twinx()
    ax2.plot(sim_prices, sim_ks, color='gray', linestyle=':', label='动态 K 值')
    ax2.set_ylabel("K Factor (信心)", color='gray', fontsize=12)
    ax2.set_ylim(0, 2.0)

    # 标记
    ax1.scatter(P_CURRENT, best_row['Kelly_Now'], color='green', s=100, zorder=5)
    ax1.scatter(V_FILL_PLAN, best_row['Kelly_At_Fill'], color='red', s=100, zorder=5)

    plt.title(f"加仓路径: 价格下跌 {P_CURRENT:.2f}->{V_FILL_PLAN:.2f} | 信心增强 k={K_FACTOR:.2f}->{k_fill_target:.2f}", fontsize=12)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.info(f"""
        **图表解读**：
        * **蓝色实线**：你应该持有的总仓位。它现在的斜率更陡峭了，因为不仅期权在变便宜，你的 K 值（虚线）也在变大。
        * **灰色虚线**：K 值的变化路径。这代表了你的心态——股价越低，下注越重。
        * **结果**：这张 {int(best_row['Days'])} 天的合约，完美配合了你的心态，在 $V_{{fill}}$ 处精准达到满仓。
    """)