import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# 尝试导入 Core 模块
try:
    from core.kelly import calculate_kelly_for_dashboard, calculate_dynamic_kelly_path, calculate_grid_signals
    from core.risk import calculate_stress_test
except ImportError:
    st.error("无法导入 core 模块，请确保 core 文件夹及 __init__.py 存在，且包含 kelly.py 和 risk.py。")
    st.stop()

# ==========================================
# 本地辅助函数：高级分组归一化逻辑 (包含 Group Confidence 权重)
# ==========================================
def normalize_portfolio_with_grouping(df_input, max_leverage_cap, group_conf_map=None):
    """
    Performs grouping-based capital allocation:
    1. Inter-group: Allocation based on 'Group Confidence' weights.
    2. Intra-group: Allocation based on (Raw Kelly * User Confidence) weights.
    """
    if isinstance(df_input, list):
        df = pd.DataFrame(df_input)
    elif isinstance(df_input, pd.DataFrame):
        df = df_input.copy()
    else:
        return pd.DataFrame(), 0.0, 1.0, pd.DataFrame()

    if df.empty:
        return pd.DataFrame(), 0.0, 1.0, pd.DataFrame()

    # 1. Calculate Weighted Score
    if 'User_Confidence' not in df.columns:
        df['User_Confidence'] = 1.0

    df['Raw_Kelly_Pct'] = df['Raw_Kelly_Pct'].astype(float)
    df['Weighted_Kelly_Score'] = df['Raw_Kelly_Pct'] * df['User_Confidence']

    # 2. Identify Groups
    if 'Group' not in df.columns:
        df['Group'] = 'Default'
    df['Group'] = df['Group'].fillna('Default').replace('', 'Default')

    unique_groups = df['Group'].unique()

    # --- NEW LOGIC: Group Weight Calculation ---
    # Use provided map or default to 1.0 for all
    if group_conf_map is None:
        group_conf_map = {g: 1.0 for g in unique_groups}

    # Ensure all groups exist in the map
    for g in unique_groups:
        if g not in group_conf_map:
            group_conf_map[g] = 1.0

    # Calculate Total Group Weight (The denominator)
    total_group_weight = sum(group_conf_map[g] for g in unique_groups)
    if total_group_weight == 0:
        total_group_weight = 1.0
    # -------------------------------------------

    # 3. Calculate Intra-group Totals
    group_totals = df.groupby('Group')['Weighted_Kelly_Score'].sum().reset_index(name='Group_Weighted_Total')
    group_ask_mean = df.groupby('Group')['Raw_Kelly_Pct'].mean().reset_index(name='Group_Ask_Mean')

    # Merge group info back to main df
    df = pd.merge(df, group_totals, on='Group', how='left')

    # 4. Core Allocation Logic
    def calculate_final_pct(row):
        group_name = row['Group']
        group_total_score = row['Group_Weighted_Total']
        asset_score = row['Weighted_Kelly_Score']

        if group_total_score == 0:
            return 0.0

        # --- CHANGED: Apply Group Confidence Weight ---
        # Group Allocation = Total Cap * (Group Conf / Sum of All Group Confs)
        group_conf = group_conf_map.get(group_name, 1.0)
        group_target_cap = max_leverage_cap * (group_conf / total_group_weight)
        # ----------------------------------------------

        # Internal Ratio = Asset Score / Group Total Score
        internal_ratio = asset_score / group_total_score

        # Final = Group Cap * Internal Ratio
        return group_target_cap * internal_ratio

    df['Final_Pct'] = df.apply(calculate_final_pct, axis=1)

    # 5. Stats & Return
    total_raw_old = df['Raw_Kelly_Pct'].sum()
    total_final_alloc = df['Final_Pct'].sum()

    total_weighted_ask = df['Weighted_Kelly_Score'].sum()
    scale_factor = total_final_alloc / total_weighted_ask if total_weighted_ask > 0 else 1.0

    group_stats = group_totals.merge(group_ask_mean, on='Group')
    group_stats['Asset_Count'] = df.groupby('Group').size().values

    # Add Group Confidence to stats for display
    group_stats['Group_Confidence'] = group_stats['Group'].map(group_conf_map)

    group_allocated = df.groupby('Group')['Final_Pct'].sum().reset_index(name='Group_Allocated')
    group_stats = group_stats.merge(group_allocated, on='Group')
    group_stats.rename(columns={'Group_Ask_Mean': 'Group_Ask'}, inplace=True)

    return df, total_raw_old, scale_factor, group_stats

# ==========================================
# 页面渲染函数
# ==========================================

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
    # k_factor_used = kelly_results['k_factor_used']
    # kelly_ratio_raw = kelly_results['kelly_ratio_raw']

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
        # --- Dynamic Kelly Path Logic ---
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
        # 注意：使用 contracts_series 的最大值来判断
        max_contracts_sim = np.max(contracts_series) if len(contracts_series) > 0 else 0
        buy_points, sell_points, step_size = calculate_grid_signals(sim_prices, contracts_series, target_contracts, P)

        st.info(f"💡 **网格操作提示** (检测到最大持仓约 {int(max_contracts_sim)} 张，已自动将提示步长设为 **{step_size}** 张):")

        col_buy, col_sell = st.columns(2)
        with col_buy:
            st.markdown("##### 📉 下跌加仓参考")
            if not buy_points:
                st.write("无近期加仓点 (或已接近满仓)")
            else:
                for point in buy_points:
                    st.write(f"- 跌至 **${point['price']:.2f}** : 加至 **{int(point['target_hold'])}** 张 (+{point['step']}张)")

        with col_sell:
            st.markdown("##### ##### 📈 上涨减仓参考")
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
            risk_df = calculate_stress_test(f_cash, L, sigma_val, total_capital)

            # Format for display
            risk_df_display = risk_df.copy()
            risk_df_display['标的跌幅'] = risk_df_display['标的跌幅'].apply(lambda x: f"{x:.2%}")
            risk_df_display['LEAPS 预估跌幅'] = risk_df_display['LEAPS 预估跌幅'].apply(lambda x: f"{x:.2%}")
            risk_df_display['账户总净值回撤'] = risk_df_display['账户总净值回撤'].apply(lambda x: f"{x:.2%}")
            risk_df_display['预估亏损'] = risk_df_display['预估亏损'].apply(lambda x: f"${x:,.0f}" if x >= 0 else f"-${abs(x):,.0f}")

            st.table(risk_df_display)
            st.caption("*注：此处使用有效杠杆 (L) 进行线性估算，实际期权在暴跌中的跌幅可能因 Gamma/Vega 效应有所不同。仅供风控参考。")


        # --- Save to Portfolio Feature ---
        if opt_price > 0 and ERP > 0:
            st.markdown("---")
            st.subheader("💾 保存到组合")

            col_save1, col_save2 = st.columns([1, 3])
            with col_save1:
                # [自定义分组输入]
                group_name = st.text_input(
                    "自定义分组 (Group)",
                    value="默认分组",
                    help="您可以自由输入组名，例如：'核心持仓'、'AI赛道'、'观察仓'等。在 Step 2 中将按此分组展示。"
                )

            with col_save2:
                st.write("")
                st.write("")
                if st.button("➕ 保存当前配置到组合", type="primary"):
                    asset_record = {
                        'Ticker': ticker,
                        'Group': group_name, # 保存 Group 字段
                        'Raw_Kelly_Pct': f_cash,
                        'User_Confidence': alpha, # Use Alpha as initial confidence
                        'ERP': ERP,
                        'L': L,
                        'k_factor': k_factor,
                        'Alpha': alpha, # 保存 Alpha 作为参考
                        'P': P,
                        'V_target': V_target,
                        'V_hard': V_hard,
                        'V_fill': V_fill,
                        'Sigma_Leaps': sigma_leaps
                    }

                    if 'portfolio_data' not in st.session_state:
                        st.session_state['portfolio_data'] = []

                    existing_tickers = [item['Ticker'] for item in st.session_state['portfolio_data']]

                    if ticker in existing_tickers:
                        idx = existing_tickers.index(ticker)
                        st.session_state['portfolio_data'][idx] = asset_record
                        st.success(f"✅ 已更新 {ticker} 的组合数据 (分组: {group_name})")
                    else:
                        st.session_state['portfolio_data'].append(asset_record)
                        st.success(f"✅ 已将 {ticker} 添加到组合 (分组: {group_name})")

                    st.info(f"当前组合共有 {len(st.session_state['portfolio_data'])} 个标的")


def render_page_multi_asset_normalization(max_leverage_cap):
    st.title("⚖️ Step 2: 多标的组合管理")
    st.markdown("---")

    # --- USER REQUESTED CORRELATION GUIDANCE ---
    with st.expander("❓ 组合相关性与仓位上限 (C_max) 设定指南"):
        st.markdown(r"""
            组合中资产的相关性（Correlation）是确定最终总仓位上限 $C_{max}$ 的关键因素。
            * **低相关性 ($\rho \approx 0$):** 允许较高的 $C_{max}$ (例如 $100\%$ 或更高)。
            * **高相关性 ($\rho \approx 1$):** 必须将 $C_{max}$ 设定在较低水平 (例如 $25\% \sim 50\%$)，以避免黑天鹅事件导致账户清零。

            *本计算器采用**分层归一化方法**：组间按权重分配资金，组内按加权 Kelly 比例分配。*
        """)
    st.markdown("---")

    if 'portfolio_data' not in st.session_state or not st.session_state['portfolio_data']:
        st.warning("组合为空。请先在 Step 1 添加资产。")
        return

    # Prepare Data
    df = pd.DataFrame(st.session_state['portfolio_data'])

    # Initialize 'User_Confidence' if missing (using Alpha as initial default)
    if 'User_Confidence' not in df.columns:
        df['User_Confidence'] = df.get('Alpha', 1.0)
    df['User_Confidence'] = df['User_Confidence'].apply(lambda x: round(x, 2))

    # Initialize Group Info
    if 'Group' not in df.columns:
        df['Group'] = 'Default'
    df['Group'] = df['Group'].fillna('Default').replace('', 'Default')
    df = df.sort_values(by='Group')


    # ==========================================
    # 1. Group Configuration (New UI Block)
    # ==========================================
    st.subheader("1. 分组权重配置 (组间分配)")
    st.caption("设置每个分组的信心权重。资金将根据 `该组权重 / 总权重` 的比例在各组间分配。")

    unique_groups = df['Group'].unique()

    # Construct a dataframe for Group Settings
    # Use session_state to persist group confidence if available, otherwise default to 1.0
    if 'group_conf_state' not in st.session_state:
        st.session_state['group_conf_state'] = {g: 1.0 for g in unique_groups}

    # Ensure all current groups are in the state, initialized to 1.0 if new
    for g in unique_groups:
        if g not in st.session_state['group_conf_state']:
            st.session_state['group_conf_state'][g] = 1.0

    # Create DF from current state for editing
    group_conf_data = [{"Group": g, "Group_Confidence": st.session_state['group_conf_state'][g]} for g in unique_groups]
    df_groups_input = pd.DataFrame(group_conf_data)

    # Editor for Groups
    edited_groups = st.data_editor(
        df_groups_input,
        column_config={
            "Group": st.column_config.TextColumn("分组名称", disabled=True),
            "Group_Confidence": st.column_config.NumberColumn(
                "组信心权重",
                help="权重越高，分得的总资金比例越大。 (例如 1.0 = 标准， 2.0 = 双倍)",
                min_value=0.0, max_value=10.0, step=0.1, format="%.1f"
            )
        },
        hide_index=True,
        use_container_width=True,
        key="group_conf_editor_widget"
    )

    # Convert to Dictionary and update session state
    group_conf_map = dict(zip(edited_groups['Group'], edited_groups['Group_Confidence']))
    st.session_state['group_conf_state'] = group_conf_map

    # ==========================================
    # 2. Asset Configuration
    # ==========================================
    st.subheader("2. 资产配置 (组内分配)")
    st.caption("调整单个资产的信心。组内资金将按 `原始建议 * 信心` 的比例分配。")

    column_config = {
        "Ticker": st.column_config.TextColumn("代码", disabled=True),
        "Group": st.column_config.TextColumn("分组", disabled=True),
        "Raw_Kelly_Pct": st.column_config.NumberColumn("原始建议 %", format="%.2f", disabled=True),
        "User_Confidence": st.column_config.NumberColumn(
            "资产信心权重",
            min_value=0.0, max_value=5.0, step=0.05, format="%.2f"
        ),
        "Alpha": st.column_config.NumberColumn("参考 Alpha", format="%.3f", disabled=True),
        "ERP": st.column_config.NumberColumn("ERP", format="%.1f%%", disabled=True),
        "L": st.column_config.NumberColumn("杠杆", format="%.2fx", disabled=True),
    }

    display_columns = ['Group', 'Ticker', 'Raw_Kelly_Pct', 'User_Confidence', 'Alpha', 'ERP', 'L']

    edited_df = st.data_editor(
        df[display_columns],
        column_config=column_config,
        column_order=display_columns,
        use_container_width=True,
        hide_index=True,
        key='portfolio_editor'
    )

    # Update main DF with edited asset confidence and sync back
    df['User_Confidence'] = edited_df['User_Confidence']
    st.session_state['portfolio_data'] = df.to_dict('records')

    # ==========================================
    # 3. Calculation & Display
    # ==========================================
    # Pass group_conf_map to the logic function
    df_result, total_raw, scale_factor, group_stats = normalize_portfolio_with_grouping(df, max_leverage_cap, group_conf_map)

    if df_result.empty:
        st.warning("计算结果为空。")
        return

    total_final_alloc = df_result['Final_Pct'].sum()
    st.markdown("---")

    # Results Display (Validation)
    st.subheader("3. 结果验证")

    with st.expander("📊 分组统计验证 (含权重检查)", expanded=True):
        st.caption("验证逻辑：注意观察 **'组权重'** 和 **'组获配资金'** 之间的比例关系。")
        group_display = group_stats.copy()

        # Visualization Formatting
        group_display['Group_Allocated'] = group_display['Group_Allocated'].apply(lambda x: f"**{x:.2%}**")
        group_display['Group_Confidence'] = group_display['Group_Confidence'].apply(lambda x: f"{x:.1f}")

        st.dataframe(
            group_display[['Group', 'Asset_Count', 'Group_Confidence', 'Group_Allocated']],
            column_config={
                "Group": "分组名称",
                "Asset_Count": "资产数",
                "Group_Confidence": "组权重",
                "Group_Allocated": "组获配资金 (结果)"
            },
            hide_index=True,
            use_container_width=True
        )

        if total_raw > max_leverage_cap:
             st.info(f"💡 原始总需求 ({total_raw:.2%}) 超过上限 ({max_leverage_cap:.2%})，系统已按分组权重进行缩放。")
        elif total_final_alloc < max_leverage_cap * 0.9999:
             st.info("🎯 组合占用低于上限，可继续增加低相关性资产或提高信心权重。")


    col_res1, col_res2 = st.columns([1, 1])
    with col_res1:
        st.write("##### 资产分配明细")
        df_final_display = df_result[['Ticker', 'Group', 'Final_Pct']].copy()
        df_final_display.rename(columns={'Final_Pct': '最终仓位 %', 'Ticker': '代码', 'Group': '分组'}, inplace=True)
        st.dataframe(
            df_final_display.style.format({'最终仓位 %': '{:.2%}'})
                             .applymap(lambda x: 'background-color: #d4edda' if isinstance(x, float) and x > 0.05 else None, subset=['最终仓位 %']),
            use_container_width=True,
            hide_index=True
        )

        st.metric("总资金占用", f"{total_final_alloc:.2%}", f"上限: {max_leverage_cap:.2%}")

        if total_final_alloc > max_leverage_cap * 1.0001:
             st.error("⚠️ 依然超限，请检查分组计算是否有误。")
        elif total_final_alloc < max_leverage_cap * 0.9999:
             st.success("✅ 组合占用在合理范围内。")
        else:
             st.success("✅ 组合占用达到目标上限。")


    with col_res2:
        st.write("##### 资金饼图")
        if total_final_alloc > 0:
            plot_df = df_result[df_result['Final_Pct'] > 0.001].copy()
            labels = plot_df['Ticker'].tolist()
            sizes = plot_df['Final_Pct'].tolist()
            remaining = max_leverage_cap - total_final_alloc

            # Prepare colors and labels for plotting
            colors = plt.cm.Paired(np.arange(len(labels)))

            if remaining > 0.001:
                labels.append(f'现金 / 剩余额度 ({remaining:.1%})')
                sizes.append(remaining)
                # Assign a distinct color to cash/remaining
                new_colors = list(colors) + [(0.7, 0.7, 0.7, 1.0)]
                colors = new_colors

            fig, ax = plt.subplots(figsize=(6, 6))
            wedges, texts, autotexts = ax.pie(sizes, autopct='%1.1f%%', startangle=90, colors=colors, wedgeprops={'linewidth': 0.5, 'edgecolor': 'white'})

            ax.legend(wedges, labels,
                      title="标的",
                      loc="center left",
                      bbox_to_anchor=(1, 0, 0.5, 1))

            ax.set_title(f"组合资金分配 (Cap={max_leverage_cap:.0%})", fontsize=14)
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("暂无分配结果")

    st.markdown("---")
    if st.button("清空组合", type="secondary"):
        st.session_state['portfolio_data'] = []
        if 'group_conf_state' in st.session_state:
             del st.session_state['group_conf_state']
        st.rerun()