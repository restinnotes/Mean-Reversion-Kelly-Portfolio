# code/ui/zh/page_dashboard.py

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# Import Core modules - 修复后的导入
from core.kelly import calculate_kelly_for_dashboard, calculate_dynamic_kelly_path, calculate_grid_signals
from core.risk import calculate_stress_test
from core.normalization import normalize_portfolio


def render_page_dashboard(ticker, lambda_val, sigma_val, r_f, k_factor, beta, P, V_target, V_hard, opt_price, delta, theta, V_fill, iv_pricing, days_to_expiry, k_fill, total_capital):
    st.title("🌌 Step 1: 凯利 LEAPS 仓位主计算器")
    st.markdown("---")

    # --- User Guide ---
    with st.expander("❓ Step 1：核心目标 (买多少？)"):
        st.markdown("""
            本计算器是系统的**核心步骤**。它将**均值回归动力** ($\lambda$) 与 **LEAPS 的杠杆风险** ($L^2\sigma^2$) 相结合，计算出在您设定的风险偏好 (k) 和信心 ($\\alpha$) 下，能够**最大化长期几何增长率**的现金投入比例。
            **核心判断：** 确保 **净优势 (ERP)** 为正值。如果 ERP < 0，即使是理论上最优的杠杆，也无法覆盖期权的租金成本 ($\\theta$) 和无风险利率 ($r_f$)，应避免开仓。
            *输入前，请确保您已从 Step 0 或券商处获取了**准确的合约数据**。*
        """)
    st.markdown("---")
    # ----------------------------

    # --- A. Core Kelly Calculation ---
    kelly_results = calculate_kelly_for_dashboard(
        P, V_target, V_hard, V_fill, opt_price, delta, theta,
        lambda_val, sigma_val, r_f, beta,
        k_factor, k_fill, total_capital
    )

    f_cash = kelly_results['f_cash']
    target_contracts = kelly_results['target_contracts']
    target_contracts_float = kelly_results['target_contracts_float']
    contract_cost = kelly_results['contract_cost']
    ERP = kelly_results['ERP']
    L = kelly_results['L']
    alpha = kelly_results['alpha']
    sigma_leaps = kelly_results['sigma_leaps']
    k_factor_used = kelly_results['k_factor_used']
    kelly_ratio_raw = kelly_results['kelly_ratio_raw']

    # --- Display Results ---
    col_d, col_m = st.columns([1, 2])
    with col_d:
        st.subheader("核心结果")
        if ERP > 0:
            st.metric(
                label=f"建议仓位 (本金 ${total_capital:,.0f})",
                value=f"{f_cash:.2%}",
                delta=f"建议持仓: {target_contracts} 张"
            )
            st.caption(f"精确计算: {target_contracts_float:.2f} 张 | 合约单价 ${contract_cost:.0f}")
        else:
            st.error("净优势为负 (ERP < 0).")

        st.divider()

        # --- ERP Explanation ---
        st.write(f"**净优势 (ERP):** {ERP:.2%}")
        with st.expander("❓ 净优势 (ERP) 解读"):
            st.markdown(r"""
                **ERP (Excess Return Premium)** 是指在扣除所有成本后的**预期年化超额收益率**。

                $$\text{ERP}_i = (\mu_{\text{stock}, i} \cdot L_i) - r_f - \theta_{\text{annual}, i}$$

                * **进攻端:** 均值回归预期收益 $\times$ 杠杆 $L$
                * **防守端:** 减去资金成本 $r_f$ 和时间损耗 $\theta_{\text{annual}}$

                **如果 ERP > 0，则表明这是一笔具有正期望值的交易。**
            """)

        # --- Alpha Explanation ---
        st.write(f"**信心系数 (Alpha):** {alpha:.3f}")
        with st.expander("❓ 信心系数 (Alpha) 解读"):
            st.markdown(r"""
                **Alpha (信心折扣系数)** 是一个动态调节因子，用于对 **Kelly 理论仓位进行限制和折扣**，其值始终 $\le 1.0$，确保您不会过度买入回归潜力减弱的资产。

                $$\alpha_i = 1 - \beta \cdot \left( \frac{P_i - P_{\text{floor}, i}}{V_i - P_{\text{floor}, i}} \right)$$

                * **关系强调：** $\alpha$ 与您设定的**估值折扣系数 ($\beta$) 成负相关关系**。$\beta$ 越大，接近目标价时的折扣越深。
                * **当股价接近硬底 ($V_{\text{hard}}$) 时:** $\alpha \to 1.0$，折扣取消，推荐分配全部 Kelly 仓位（信心最高）。
                * **当股价接近目标价 ($V_{\text{target}}$) 时:** $\alpha \to (1-\beta)$，折扣生效，Kelly 仓位被缩减。
            """)

        st.write(f"**LEAPS 年化波动率:** {sigma_leaps:.2%}")

    with col_m:
        # --- Dynamic Kelly Path Logic (NEW) ---
        st.subheader("🔮 动态 K 值仓位路径推演 (含网格买卖点)")
        st.caption(f"全景推演：下跌 K 值增强 ({k_factor:.2f}$\\to${k_fill:.2f})，上涨时自动止盈。")

        # 1. Generate path data
        sim_prices, allocations, k_values, contracts_series = calculate_dynamic_kelly_path(
            P, V_target, V_hard, V_fill,
            lambda_val, sigma_val, r_f, beta,
            k_factor, k_fill, total_capital,
            days_to_expiry, iv_pricing
        )

        # 2. Plotting (Dual Axis)
        fig, ax1 = plt.subplots(figsize=(10, 5))

        # 绘制区域填充
        ax1.axvspan(V_hard, P, color='#d4edda', alpha=0.5, label='加仓区')
        ax1.axvspan(P, V_target * 1.05, color='#f8d7da', alpha=0.5, label='减仓区')

        # 绘制仓位曲线
        ax1.plot(sim_prices, allocations, color='#1f77b4', linewidth=3, label='建议仓位 %')
        ax1.set_xlabel("股价模拟 ($)", fontsize=12)
        ax1.set_ylabel("仓位比例", color='#1f77b4', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='#1f77b4')
        ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

        # 绘制 K 值变化 (右轴)
        ax2 = ax1.twinx()
        ax2.plot(sim_prices, k_values, color='gray', linestyle=':', alpha=0.7, label='动态 K 值')
        ax2.set_ylabel("信心系数 K", color='gray', fontsize=12)
        ax2.set_ylim(0, 2.0)

        # 标记关键点
        ax1.scatter([P], [f_cash], color='black', s=100, zorder=5, label=f'当前点 P (${P:.2f})')
        v_fill_alloc_index = np.argmin(np.abs(sim_prices - V_fill))
        ax1.scatter([V_fill], allocations[v_fill_alloc_index], color='red', s=100, zorder=5, label=f'补仓点 V_fill (${V_fill:.2f})')
        v_hard_alloc_index = np.argmin(np.abs(sim_prices - V_hard))
        ax1.scatter([V_hard], allocations[v_hard_alloc_index], color='green', s=100, zorder=5, label=f'硬底 V_hard (${V_hard:.2f})')

        # 增加图例
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='upper right')

        plt.title(f"{ticker} 动态凯利仓位路径 ($V_{{hard}}$ 到 $V_{{target}}$)", fontsize=14)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # --- Grid Trading Advice ---
        buy_points, sell_points, step_size = calculate_grid_signals(sim_prices, contracts_series, target_contracts, P)

        st.info(f"💡 **网格操作提示** (检测到最大持仓约 {int(max(contracts_series) if contracts_series else 0)} 张，已自动将提示步长设为 **{step_size}** 张):")

        col_buy, col_sell = st.columns(2)
        with col_buy:
            st.markdown("##### 📉 下跌加仓参考")
            if not buy_points:
                st.write("无近期加仓点 (或已接近满仓)")
            else:
                for point in buy_points:
                    st.write(f"- 跌至 **${point['price']:.2f}** : 加至 **{int(point['target_hold'])}** 张 (+{point['step']}张)")

        with col_sell:
            st.markdown("##### 📈 上涨减仓参考")
            if not sell_points:
                st.write("无近期减仓点 (或已空仓)")
            else:
                 for point in sell_points:
                      st.write(f"- 涨至 **${point['price']:.2f}** : 减至 **{int(point['target_hold'])}** 张 (-{point['step']}张)")


        st.markdown("---")


        # --- G. Stress Test ---
        st.subheader("⚠️ 压力测试 (Stress Test) - 账户净值模拟")
        st.caption(f"基于当前建议仓位 ({f_cash:.2%}) 的次日盈亏模拟")

        with st.expander("📊 点击展开：如果明天发生暴跌，我的账户将承受？", expanded=True):
            # Calculate stress test data
            risk_df = calculate_stress_test(f_cash, L, sigma_val, total_capital)

            # Format for display
            risk_df_display = risk_df.copy()
            risk_df_display['标的跌幅'] = risk_df_display['标的跌幅'].apply(lambda x: f"{x:.2%}")
            risk_df_display['LEAPS 预估跌幅'] = risk_df_display['LEAPS 预估跌幅'].apply(lambda x: f"{x:.2%}")
            risk_df_display['账户总净值回撤'] = risk_df_display['账户总净值回撤'].apply(lambda x: f"{x:.2%}")
            risk_df_display['预估亏损'] = risk_df_display['预估亏损'].apply(lambda x: f"${x:,.0f}" if x >= 0 else f"-${abs(x):,.0f}")

            st.table(risk_df_display)
            st.caption("*注：此处使用有效杠杆 (L) 进行线性估算，实际期权在暴跌中的跌幅可能因 Gamma/Vega 效应有所不同。仅供风控参考。如果 $3\\sigma$ 亏损额让你感到恐慌，请在侧边栏调低 $k$ 值。")


        # --- Save to Portfolio Feature ---
        if opt_price > 0 and ERP > 0:
            st.markdown("---")
            st.subheader("💾 保存到组合")

            if st.button("➕ 保存当前配置到组合", type="primary"):
                asset_record = {
                    'Ticker': ticker,
                    'Raw_Kelly_Pct': kelly_ratio_raw, # Save the K=1 raw Kelly ratio for normalization
                    'ERP': ERP,
                    'L': L,
                    'k_factor': k_factor,
                    'Alpha': alpha,
                    'P': P,
                    'V_target': V_target,
                    'V_hard': V_hard,
                    'Sigma_Leaps': sigma_leaps
                }

                existing_tickers = [item['Ticker'] for item in st.session_state.get('portfolio_data', [])]

                if ticker in existing_tickers:
                    idx = existing_tickers.index(ticker)
                    st.session_state['portfolio_data'][idx] = asset_record
                    st.success(f"✅ 已更新 {ticker} 的组合数据")
                else:
                    if 'portfolio_data' not in st.session_state:
                                st.session_state['portfolio_data'] = []
                    st.session_state['portfolio_data'].append(asset_record)
                    st.success(f"✅ 已将 {ticker} 添加到组合")

                st.info(f"当前组合共有 {len(st.session_state.get('portfolio_data', []))} 个标的")


def render_page_multi_asset_normalization(max_leverage_cap):
    st.title("⚖️ Step 2: 多标的组合管理 - 简单归一化")
    st.markdown("---")

    # --- USER REQUESTED CORRELATION GUIDANCE ---
    with st.expander("❓ 组合相关性与仓位上限 (C_max) 设定指南"):
        st.markdown(r"""
            组合中资产的相关性（Correlation）是确定最终总仓位上限 $C_{max}$ 的关键因素。
            ... (omitted repetitive text for brevity)
            *本计算器采用简单的线性归一化方法 (Final Pct = Raw Kelly $\times$ Scale Factor)，请根据您的组合相关性设置合理的 $C_{max}$。*
        """)
    st.markdown("---")

    portfolio_data = st.session_state.get('portfolio_data')

    if not portfolio_data:
        st.warning("组合中没有资产。请回到 Step 1 计算并点击 '保存当前配置到组合'。")
        return

    df, total_raw_exposure, scale_factor = normalize_portfolio(portfolio_data, max_leverage_cap)

    if df.empty:
        st.warning("组合数据为空。")
        return

    st.markdown(f"**总资产数量:** `{len(df)}`")
    st.markdown(f"**原始 Kelly 理论总仓位 (C_raw):** `{total_raw_exposure:.2%}`")
    st.markdown(f"**设置的现金上限 (C_max):** `{max_leverage_cap:.2%}`")

    # 2. Normalize Logic Display
    if scale_factor < 1.0:
        st.error(f"🚨 总仓位超限，已进行归一化缩放。缩放因子: {scale_factor:.4f}")
    else:
        st.success("✅ 总仓位在限制内。无需缩放。")

    # 4. Format Output
    df_display = df[['Ticker', 'Raw_Kelly_Pct', 'Final_Pct', 'ERP', 'L', 'Sigma_Leaps', 'k_factor', 'Alpha']].copy()

    # Apply formatting
    df_display.rename(columns={
        'Raw_Kelly_Pct': '原始 Kelly %',
        'Final_Pct': '最终仓位 %',
        'ERP': '净优势 (ERP)',
        'L': '杠杆 (L)',
        'Sigma_Leaps': 'LEAPS波动率',
        'Alpha': '信心 (Alpha)',
        'k_factor': 'K 因子'
    }, inplace=True)

    df_display['原始 Kelly %'] = df_display['原始 Kelly %'].apply(lambda x: f"{x:.2%}")
    df_display['最终仓位 %'] = df_display['最终仓位 %'].apply(lambda x: '**{}**'.format(f'{x:.2%}'))
    df_display['净优势 (ERP)'] = df_display['净优势 (ERP)'].apply(lambda x: f"{x:.2%}")
    df_display['杠杆 (L)'] = df_display['杠杆 (L)'].apply(lambda x: f"{x:.2f}x")
    df_display['LEAPS波动率'] = df_display['LEAPS波动率'].apply(lambda x: f"{x:.2%}")
    df_display['信心 (Alpha)'] = df_display['信心 (Alpha)'].apply(lambda x: f"{x:.3f}")
    df_display['K 因子'] = df_display['K 因子'].apply(lambda x: f"{x:.2f}")


    st.subheader(f"\n最终组合分配结果 (总仓位: {df['Final_Pct'].sum():.2%})")
    st.dataframe(df_display, hide_index=True, use_container_width=True)

    if st.button("清空组合数据", help="这将删除所有已保存的资产记录"):
        st.session_state['portfolio_data'] = []
        st.rerun()