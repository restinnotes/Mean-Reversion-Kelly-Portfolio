# code/ui/zh/page_diagnosis.py

import streamlit as st
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# Import Core/Data modules - 修复后的导入，不再手动操作 sys.path
from core.simulation import run_simulation, analyze_risk_reward
from data.rolling import run_rolling_analysis
from ui.plot_utils import get_resource_root

project_root = get_resource_root()


def render_page_diagnosis(ticker, window_days, lambda_val, sigma_val, P_anchor_global):
    st.title("📈 Step 0: 市场诊断 - 估值与分布")
    st.subheader(f"资产: {ticker} | 滚动窗口: {window_days} 交易日")
    st.markdown("---")

    # --- User Guide: 融合了原版的参数警示与新版的分布思维 ---
    with st.expander("❓ Step 0 核心逻辑：先验证参数，再推演未来", expanded=True):
        st.markdown("""
            **在使用任何模型前，必须完成以下两步逻辑闭环：**

            **第一步：参数验证 (Diagnosis)**
            * **Lambda (回归动力)**：衡量股价向目标回归的速度。关键原则是 **宁低勿高**：Lambda 越低，模型越保守，不会过度认为股价会快速回归，从而避免仓位过重。即便当前 Lambda 不在历史高位，也建议根据风险偏好适当调低，以保持充足安全边际。
            * **Sigma (波动率)**：确认我们使用的是稳健的波动率（通常是历史 85% 分位数），确保在计算风险时足够保守。

            **第二步：分布推演 (Simulation)**
            * **假设前提**：*“如果估值回归真的按照上述历史规律运行...”*
            * **盈亏分布**：看清楚 **10%底线 (Hold Risk)** 和 **10%高点 (Touch Gain)**。
            * **决策**：只有当 Lambda 真实可靠，且蒙特卡洛推演出的“底线风险”你能承受时，才能进入 Step 1 开仓。
        """)
    st.markdown("---")

    # --- Data Loading & Calculation ---
    try:
        analysis_data = run_rolling_analysis(ticker, project_root, window_days)
    except FileNotFoundError:
        st.warning(f"警告: 找不到 {ticker}_pe.csv 文件。")
        return
    except Exception as e:
        st.error(f"读取 PE 数据失败或计算指标失败: {e}")
        return

    if analysis_data is None:
        st.warning("数据不足。")
        return

    df = analysis_data['df']
    rolling_df = analysis_data['rolling_df']
    metrics = analysis_data['current_metrics']
    robust_stats = analysis_data['robust_stats']

    current_lambda = metrics['current_lambda']
    current_hl = metrics['current_hl']
    current_pe = metrics['current_pe']
    current_mean = metrics['current_mean']
    current_sigma_daily = metrics['current_sigma_daily']

    current_t_stat = robust_stats.get('structural_t_stat', 0.0)
    current_conf = robust_stats.get('structural_confidence', 0.0)

    # --- Calculate Target Price and Entry Price ---
    current_P_anchor = P_anchor_global
    if current_pe > 0 and current_P_anchor > 0:
        target_price_from_pe = current_P_anchor * (current_mean / current_pe)
    else:
        target_price_from_pe = None

    annual_sigma_for_ref = sigma_val # Use the global sigma from sidebar
    daily_sigma_for_ref = annual_sigma_for_ref / np.sqrt(252)

    if current_P_anchor > 0 and daily_sigma_for_ref > 0:
        price_drop_1sd = current_P_anchor * np.exp(-daily_sigma_for_ref)
    else:
        price_drop_1sd = None

    # 3. 基于“历史结构性置信度”判定强度 (文案相应调整)
    if current_conf >= 95.0:
        conf_label = "⭐⭐⭐ 极高 (Robust)"
        conf_color = "green"
        conf_help = f"历史结构性置信度 {current_conf:.1f}% (>95%)。\n数据证实：该资产在历史上长期遵循围绕 {window_days} 日均线的均值回归规律，策略有效性极高。"
    elif current_conf >= 85.0:
        conf_label = "⭐⭐ 较高 (Valid)"
        conf_color = "orange"
        conf_help = f"历史结构性置信度 {current_conf:.1f}% (>85%)。\n数据证实：该资产存在均值回归特征，策略长期有效，但噪音稍大。"
    else:
        conf_label = "⚠️ 存疑 (Weak)"
        conf_color = "red"
        conf_help = f"历史结构性置信度 {current_conf:.1f}% (<85%)。\n警惕：该资产历史上并没有表现出稳定的均值回归特征（可能是趋势型或随机游走），当前策略可能不适用。"

    # =========================================================
    # Part 1: 参数验证与历史回溯 (The Gatekeeper)
    # =========================================================
    st.subheader("1. 核心参数验证 (Diagnosis)")

    col_d1, col_d2, col_d3, col_d4 = st.columns(4)

    with col_d1:
        st.markdown("**估值偏离度**")
        st.code(f"当前 PE: {current_pe:.2f}")
        st.metric("均值偏离", f"{(current_pe - current_mean)/current_mean:.1%}")

    with col_d2:
        st.markdown("**回归动力 (Lambda)**")
        st.code(f"λ: {current_lambda:.4f}")
        st.caption(f"半衰期: {current_hl:.1f} 天")

    with col_d3:
        st.markdown("**均值回归置信度**")
        st.markdown(f":{conf_color}[**{current_conf:.1f}%**]")
        st.caption(f"{conf_label}", help=f"T-Stat: {current_t_stat:.2f}\n{conf_help}")

    with col_d4:
        st.markdown("**估值中枢目标价**")
        if target_price_from_pe is not None and current_P_anchor > 1.0:
             st.code(f"P_target: {target_price_from_pe:.2f}")
             st.caption(f"参考加仓点 (1σ): {price_drop_1sd:.2f}",
                        help=f"这是基于锚定股价 P (${current_P_anchor:.2f}) 预期日波动 (-1σ) 推算的参考加仓点。请在侧边栏更新锚定价格。")
        else:
             st.code("P_target: N/A")
             st.caption("⚠️ 请在侧边栏 **Step 0 参数** 中设置 **当前股价 P (Anchor)** 以计算目标价。")

    # --- 历史图表 (Visual Verification) ---
    # Plot 1: PE Context
    fig1, ax0 = plt.subplots(figsize=(10, 3))
    ax0.plot(rolling_df.index, rolling_df['value'], 'k', alpha=0.8, label='PE')
    ax0.plot(rolling_df.index, rolling_df['rolling_mean'], 'b--', label=f'{window_days}日均线')
    ax0.set_title(f'{ticker} 估值偏离度 (验证: 低估是否真实？)', fontsize=10)
    ax0.legend(loc='upper left'); ax0.grid(True, alpha=0.3)
    st.pyplot(fig1)
    plt.close(fig1)

    # Plot 2: Lambda History (Critical Check)
    lambda_80 = np.percentile(rolling_df['Lambda'], 80)

    fig2, ax1 = plt.subplots(figsize=(10, 3))
    ax1.plot(rolling_df.index, rolling_df['Lambda'], color='#1f77b4', label='Lambda')
    ax1.axhline(lambda_80, color='r', linestyle='--', label=f'80%分位 ({lambda_80:.1f})')
    ax1.set_title('Lambda 历史走势 (验证: 是否处于不可持续的极高位？)', fontsize=10)
    ax1.legend(loc='upper left'); ax1.grid(True, alpha=0.3)
    st.pyplot(fig2)
    plt.close(fig2)

    # Plot 3: Sigma (Volatility Check)
    st.markdown("**波动率验证 (Sigma Check)**")
    if st.session_state.get('sigma_rolling_data') and ticker in st.session_state.sigma_rolling_data:
        roll_vol = st.session_state.sigma_rolling_data[ticker]

        fig4, ax3 = plt.subplots(figsize=(10, 3))
        if isinstance(roll_vol.index, pd.DatetimeIndex): idx_plot = roll_vol.index
        else: idx_plot = roll_vol.index.values

        ax3.plot(idx_plot, roll_vol.values, color='gray', alpha=0.6, label='滚动波动率')
        ax3.axhline(sigma_val, color='green', linewidth=2, label=f'当前采用 Sigma ({sigma_val:.1%})')
        ax3.set_title(f'波动率验证 (当前采用值是否覆盖了历史大部分风险？)', fontsize=10)
        ax3.legend(loc='upper left'); ax3.grid(True, alpha=0.3)
        ax3.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        st.pyplot(fig4)
        plt.close(fig4)
    else:
        st.info("💡 正在自动加载数据，请稍候。")

    st.markdown("---")

    # =========================================================
    # Part 2: 未来推演 (The Crystal Ball)
    # =========================================================
    st.subheader("2. 盈亏分布推演 (Simulation)")

    lambda_display = f"{current_lambda:.2f}" if current_lambda is not None else 'N/A'
    st.caption(f"👉 **前提假设**：如果估值回归真的遵循上述 Lambda={lambda_display} 的历史规律，那么正态分布下的结局是：")

    # 定义关键时间窗口
    check_points_map = {
        "1个月 (21交易日)": 21,
        "3个月 (63交易日)": 63,
        "6个月 (126交易日)": 126,
        "9个月 (189交易日)": 189
    }

    # 运行模拟
    if current_sigma_daily is None or current_sigma_daily == 0:
        st.warning("日内 Sigma (波动率) 数据缺失或为零，无法运行蒙特卡洛模拟。")
        return

    paths = run_simulation(current_pe, current_mean, current_lambda, current_sigma_daily, days_to_simulate=200)

    # 分析分布
    df_risk = analyze_risk_reward(paths, current_pe, check_points_map)

    # 输出表格
    st.dataframe(
        df_risk.style.format({
            "亏损概率 (Loss%)": "{:.1%}",
            "10%底线 (Hold)": "{:+.2%}",
            "预期收益 (Exp)": "{:+.2%}",
            "10%高点 (Touch)": "{:+.2%}"
        }).applymap(lambda v: 'color: #ff4b4b' if v < 0 else 'color: #2dc937',
                    subset=["10%底线 (Hold)", "预期收益 (Exp)", "10%高点 (Touch)"]),
        hide_index=True,
        use_container_width=True
    )

    # 模拟路径分布图
    fig_mc, ax_mc = plt.subplots(figsize=(10, 4))
    percentiles = [10, 50, 90]
    colors = ['#ff4b4b', '#1f77b4', '#2dc937']
    labels = ['10% 底线 (Hold)', '50% 中位数', '90% 高点 (Touch)']
    days = np.arange(paths.shape[0])

    for p, c, l in zip(percentiles, colors, labels):
        line_data = np.percentile(paths, p, axis=1)
        ax_mc.plot(days, line_data, color=c, lw=2, label=l)

    ax_mc.axhline(current_pe, color='gray', linestyle=':', label='当前价')
    ax_mc.set_title(f"{ticker} 未来价格路径分布锥")
    ax_mc.set_xlabel("交易日")
    ax_mc.set_ylabel("PE Ratio")
    ax_mc.legend(loc='upper left', fontsize=8)
    ax_mc.grid(True, alpha=0.3)

    st.pyplot(fig_mc)
    plt.close(fig_mc)