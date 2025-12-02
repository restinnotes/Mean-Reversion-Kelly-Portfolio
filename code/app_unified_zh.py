import streamlit as st
import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.ticker as mtick
import matplotlib.font_manager as fm

# ==========================================
# 1. SETUP: Path & Imports
# ==========================================
def get_resource_root():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    else:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.abspath(os.path.join(current_dir, ".."))

project_root = get_resource_root()

sys.path.append(os.path.join(project_root, "code", "utils"))
sys.path.append(os.path.join(project_root, "code", "strategies"))

try:
    from lambda_tools import get_ou_for_ticker, calculate_ou_params
    from sigma_tools import get_sigma
    from optimal_expiry_solver import bs_greek_calculator, calculate_single_asset_kelly_ratio
except ImportError as e:
    st.error(f"Module import error. Please ensure dependency files (lambda_tools.py, sigma_tools.py, optimal_expiry_solver.py) are accessible relative to the app structure: {e}")
    pass


# ==========================================
# 2. Matplotlib Font Configuration (Chinese Support)
# ==========================================
def configure_chinese_font():
    """
    配置中文字体。使用项目内上传的 SimHei.ttf 文件。
    NOTE: 请确保 SimHei.ttf 文件位于项目根目录下的 fonts/ 文件夹内。
    """
    # 1. 定义字体路径
    font_name = "SimHei.ttf"
    # 假设 SimHei.ttf 位于项目根目录的 fonts/ 文件夹中
    font_path = os.path.join(project_root, "fonts", font_name)

    if os.path.exists(font_path):
        try:
            # 2. 注册并加载字体
            fm.fontManager.addfont(font_path)
            prop = fm.FontProperties(fname=font_path)
            custom_font_name = prop.get_name()

            # 3. 应用配置
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = [custom_font_name, 'DejaVu Sans', 'Arial']
            plt.rcParams['axes.unicode_minus'] = False
        except Exception as e:
            st.warning(f"❌ 字体加载失败: {e}。请检查文件是否损坏或路径是否正确。")
    else:
        st.warning(f"⚠️ 未找到字体文件：{font_path}。虽然不影响计算，但图表中文可能显示为方框。请确认 'fonts/' 目录下有 SimHei.ttf。")


# 在脚本启动时立即执行配置
configure_chinese_font()

# ==========================================
# 3. HELPER FUNCTIONS FOR MONTE CARLO
# ==========================================

def run_simulation(current_pe, target_pe, lambda_annual, sigma_daily, days_to_simulate=252, num_paths=10000):
    dt = 1/252
    paths = np.zeros((days_to_simulate + 1, num_paths))
    paths[0] = current_pe
    theta = target_pe

    for t in range(1, days_to_simulate + 1):
        X_t = paths[t-1]
        drift = lambda_annual * (theta - X_t) * dt
        shock = sigma_daily * np.random.normal(0, 1, num_paths)
        paths[t] = X_t + drift + shock
    return paths

def analyze_probabilities(paths, target_pe, current_pe):
    days_simulated = paths.shape[0] - 1
    check_points = [21, 42, 63, 126, 189, 252]
    results = []
    is_long = target_pe > current_pe

    for day in check_points:
        if day > days_simulated: continue
        final_values = paths[day]

        if is_long: prob_end = np.mean(final_values >= target_pe)
        else: prob_end = np.mean(final_values <= target_pe)

        path_slice = paths[:day+1, :]
        if is_long: has_hit = np.any(path_slice >= target_pe, axis=0)
        else: has_hit = np.any(path_slice <= target_pe, axis=0)
        prob_touch = np.mean(has_hit)

        approx_cal_days = int(day * (365/252))

        results.append({
            "交易日": day,
            "~日历日": f"{approx_cal_days}d",
            "触摸目标概率": prob_touch,
            "结束时保持概率": prob_end,
            "预期PE": np.mean(final_values)
        })
    return pd.DataFrame(results)


# ==========================================
# 4. PAGE LOGIC FUNCTIONS
# ==========================================

# --- Page 1: Diagnosis (Rolling Analysis) ---
def page_diagnosis(ticker, window_days):
    st.title("📈 Step 0: 市场诊断 - 滚动分析")
    st.subheader(f"资产: {ticker} | 滚动窗口: {window_days} 交易日")
    st.markdown("---")

    # --- User Guide ---
    with st.expander("❓ Step 0：市场诊断指引 (验证均值回归)"):
        st.markdown("""
            这是**风险控制的第一步**，用于验证均值回归假设是否成立，以及评估回归动力 ($\lambda$) 的可靠性。
            **核心目标：**
            1.  **判断低估是否真实：** 查看 PE Ratio 曲线是否明显低于滚动均线，确认存在回归空间。
            2.  **评估 $\lambda$ 质量：** 检查 Lambda 曲线最右端的值是否远高于其历史平均水平（虚高）。如果是，后续 Step 1 中应**手动调低 $\lambda$**。
            3.  **确认时间可行性：** 检查 Monte Carlo 模拟，确认 90% 概率触摸目标所需的最短时间，以此作为 **LEAPS 选品的期限底线**。
        """)
    st.markdown("---")
    # ----------------------------

    # --- Data Loading uses the consistent project_root ---
    pe_csv_path = os.path.join(project_root, "pe_csv", f"{ticker}_pe.csv")
    if not os.path.exists(pe_csv_path):
        st.warning(f"警告: 找不到 {ticker}_pe.csv 文件进行滚动分析。请确保数据位于: {os.path.basename(project_root)}/pe_csv/")
        return

    try:
        df = pd.read_csv(pe_csv_path, parse_dates=['date'])
        df = df.sort_values('date').reset_index(drop=True)
    except Exception as e:
        st.error(f"读取 PE 数据失败: {e}")
        return

    # --- 1. Calculate Rolling Metrics ---
    if len(df) < window_days:
        st.warning("数据不足，无法进行滚动指标计算。")
        return

    df['rolling_mean'] = df['value'].rolling(window=window_days).mean()

    dates_hist = []; lambdas_annual_hist = []; half_lives_hist = []; sigmas_daily_hist = []

    start_index = window_days - 1

    if 'calculate_ou_params' in globals():
        for i in range(start_index, len(df)):
            window_series = df.iloc[i-window_days+1 : i+1].set_index('date')['value']
            try:
                ou_hist = calculate_ou_params(window_series)
                if ou_hist:
                    dates_hist.append(df.iloc[i]['date'])
                    lambdas_annual_hist.append(ou_hist['lambda'] * 252)
                    half_lives_hist.append(ou_hist['half_life'])
                    sigmas_daily_hist.append(ou_hist['sigma'])
            except Exception:
                continue
    else:
        st.error("依赖模块 (lambda_tools.py) 未导入，无法进行 OU 参数滚动计算。")
        return


    if not lambdas_annual_hist:
        st.warning("数据不足，无法进行滚动指标计算。")
        return

    current_lambda = lambdas_annual_hist[-1]
    current_hl = half_lives_hist[-1]
    current_pe = df['value'].iloc[-1]
    current_mean = df['rolling_mean'].iloc[-1]
    current_sigma_daily = sigmas_daily_hist[-1]

    if st.session_state.ticker == ticker:
        st.session_state['lambda'] = current_lambda

    # --- 2. Diagnosis Report ---
    st.subheader("诊断报告与 Monte Carlo 模拟")
    st.markdown("---")
    col_d1, col_d2 = st.columns(2)

    with col_d1:
        st.markdown("**PE 估值状态**")
        st.code(f"当前 PE: {current_pe:.2f}")
        st.code(f"{window_days}日均值: {current_mean:.2f}")

    with col_d2:
        st.markdown("**回归与波动率**")
        st.code(f"年化 Lambda (λ): {current_lambda:.4f}")
        st.code(f"半衰期: {current_hl:.2f} 天")
        st.code(f"日波动率 (σ_PE): {current_sigma_daily:.4f}")

    st.markdown("---")

    # ------------------------------------------

    # --- 3. Monte Carlo Simulation ---
    st.markdown("##### Monte Carlo 模拟结果")
    st.caption(f"目标: PE {current_pe:.2f} 修复到均值 PE {current_mean:.2f} | 模拟路径: 10,000条")

    paths = run_simulation(current_pe, current_mean, current_lambda, current_sigma_daily)
    df_probs = analyze_probabilities(paths, current_mean, current_pe)

    safe_days = 0
    safe_cal_days = 0
    found_safe_zone = False

    for idx, row in df_probs.iterrows():
        if row['触摸目标概率'] > 0.9:
            safe_days = int(row['交易日'])
            safe_cal_days = int(row['~日历日'].replace('d',''))
            found_safe_zone = True
            break

    df_probs['触摸目标概率'] = df_probs['触摸目标概率'].apply(lambda x: f"{x:.1%}")
    df_probs['结束时保持概率'] = df_probs['结束时保持概率'].apply(lambda x: f"{x:.1%}")
    df_probs['预期PE'] = df_probs['预期PE'].apply(lambda x: f"{x:.2f}")
    st.dataframe(df_probs, hide_index=True)

    if found_safe_zone:
        st.success(f"**[推荐行动计划]**: 90% 概率触摸目标所需的最短时间为 **{safe_days} 交易日 (~{safe_cal_days} 日历日)**。")
        st.info(f"选品建议：购买到期日 **大于等于** {safe_cal_days} 日历日的 LEAPS 期权。")
    else:
        st.warning(f"**[警告]**: 在 1 年内无法达到 90% 的目标触摸概率。回归缓慢/不确定。建议购买 > 1 年的 LEAPS 或保持现金。")

    st.markdown("---")

    # --- 4. Plotting ---
    plot_df = df.iloc[start_index:].copy()
    plot_df['Lambda'] = lambdas_annual_hist
    plot_df['Half_Life'] = half_lives_hist
    plot_df.set_index('date', inplace=True)

    lambda_80 = np.percentile(lambdas_annual_hist, 80)
    lambda_20 = np.percentile(lambdas_annual_hist, 20)
    hl_90 = np.percentile(half_lives_hist, 90)

    # Plot 1: PE Context
    fig1, ax0 = plt.subplots(figsize=(10, 3))
    ax0.plot(plot_df.index, plot_df['value'], 'k', alpha=0.8, label='市盈率')
    ax0.plot(plot_df.index, plot_df['rolling_mean'], 'b--', label=f'{window_days}日滚动均值')
    ax0.set_title(f'{ticker} 市盈率与 {window_days}日滚动均值 (估值偏离度)', fontsize=10)
    ax0.set_xlabel("日期")
    ax0.set_ylabel("市盈率")
    ax0.legend(loc='upper left'); ax0.grid(True, alpha=0.3)
    st.pyplot(fig1)
    plt.close(fig1)

    # Plot 2: Lambda
    fig2, ax1 = plt.subplots(figsize=(10, 3))
    ax1.plot(plot_df.index, plot_df['Lambda'], color='#1f77b4', label='年化 Lambda')
    ax1.axhline(lambda_80, color='r', linestyle='--', label=f'80%分位 ({lambda_80:.1f})')
    ax1.axhline(lambda_20, color='g', linestyle='--', label=f'20%分位 ({lambda_20:.1f})')
    ax1.set_title('均值回归速度 (Lambda)', fontsize=10)
    ax1.set_xlabel("日期")
    ax1.set_ylabel("Lambda (年化)")
    ax1.legend(loc='upper left'); ax1.grid(True, alpha=0.3)
    st.pyplot(fig2)
    plt.close(fig2)

    # Plot 3: Half-Life
    fig3, ax2 = plt.subplots(figsize=(10, 3))
    ax2.plot(plot_df.index, plot_df['Half_Life'], color='#ff7f0e', label='半衰期 (交易日)')
    ax2.axhline(hl_90, color='purple', linestyle='--', label=f'90%分位风险 ({hl_90:.1f}日)')
    ax2.set_ylim(0, max(300, hl_90 * 1.5))
    ax2.set_title('隐含半衰期 (风险指标)', fontsize=10)
    ax2.set_xlabel("日期")
    ax2.set_ylabel("半衰期 (交易日)")
    ax2.legend(loc='upper left'); ax2.grid(True, alpha=0.3)
    st.pyplot(fig3)
    plt.close(fig3)

    # --- 5. Sigma Plot (Added) ---
    st.markdown("---")
    st.subheader("历史波动率诊断 (Sigma Tools)")

    if st.session_state.get('sigma_rolling_data') and ticker in st.session_state.sigma_rolling_data:
        roll_vol = st.session_state.sigma_rolling_data[ticker]
        sigma_val = st.session_state.sigma_dict[ticker]
        window = 252
        percentile = 0.85

        if isinstance(roll_vol.index, pd.DatetimeIndex):
            index_for_plot = roll_vol.index
        else:
            index_for_plot = roll_vol.index.values

        if not roll_vol.empty:
            current = roll_vol.iloc[-1]
            pval = roll_vol.quantile(percentile)

            fig4, ax3 = plt.subplots(figsize=(10, 4))

            ax3.plot(index_for_plot, roll_vol.values, linewidth=1.4, label=f'{window}日滚动年化波动率')
            ax3.axhline(pval, linestyle='--', linewidth=1.5, color='orange', label=f'{percentile*100:.0f}%分位 = {pval:.2%}')

            final_sigma = max(current, pval)
            ax3.axhline(final_sigma, linestyle='-', linewidth=1.5, color='green', label=f'最终稳健 Sigma = {final_sigma:.2%}')

            ax3.scatter(index_for_plot[-1], current, color='red', s=50, zorder=5, label=f'当前波动率 = {current:.2%}')


            ax3.set_title(f"{ticker} 滚动年化波动率 ({window}日) — 稳健 Sigma", fontsize=10)
            ax3.set_xlabel("日期")
            ax3.set_ylabel("年化波动率")
            ax3.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
            ax3.legend(loc='upper left')
            ax3.grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig4)
            plt.close(fig4)
        else:
            st.warning("无足够的历史数据来绘制滚动波动率图表。")
    else:
        st.warning("请在侧边栏点击 '获取历史统计数据' 以加载波动率历史数据。")


# --- Page 2: Optimal Expiry Solver ---
def page_solver(P_CURRENT, V_TARGET, V_HARD_FLOOR, V_FILL_PLAN, LAMBDA, SIGMA_ASSET, IV_PRICING, R_RISKFREE, ticker, K_FACTOR, BETA):
    st.title("🎯 Step 0.5: 最优期限求解器")
    st.subheader(f"资产: {ticker} | P={P_CURRENT}")
    st.markdown("---")

    # --- User Guide ---
    with st.expander("❓ Step 0.5：求解器原理与下一步行动"):
        st.markdown("""
            求解器旨在找到一个**攻守平衡点**：即在满足凯利增长速度要求的同时，预留出在计划补仓价 ($V_{fill}$) 进行 **1:1 补仓的充足现金**。
            * **进攻曲线 (Offense)**：基于 Kelly 理论，期限越长，波动率惩罚越低，建议仓位越高。**注意：进攻曲线使用当前设定的 k 值（例如 0.5）来计算初始仓位。**
            * **防守上限 (Defense)**：基于补仓现金约束，期限越长，期权越贵，可用的初始仓位越低。
            两条曲线的**交点即为最优期限 (Sweet Spot)**。
        """)
    st.markdown("---")
    # ----------------------------

    if 'bs_greek_calculator' not in globals() or 'calculate_single_asset_kelly_ratio' not in globals():
        st.error("依赖模块 (optimal_expiry_solver.py) 未导入，无法进行求解。")
        return

    results = []
    for days in range(30, 1100, 7):
        T = days / 365.0
        c_price, c_delta, c_theta_annual = bs_greek_calculator(P_CURRENT, V_HARD_FLOOR, T, R_RISKFREE, IV_PRICING)

        # Calculates full Kelly (k=1.0) ratio first
        kelly_full = calculate_single_asset_kelly_ratio(
            P_CURRENT, c_price, c_delta, c_theta_annual, V_TARGET, V_HARD_FLOOR, LAMBDA, SIGMA_ASSET, R_RISKFREE, beta=BETA
        )
        # Apply the user's k-factor (e.g., 0.5) for the initial target allocation
        kelly_target = kelly_full * K_FACTOR

        c0, _, _ = bs_greek_calculator(P_CURRENT, V_HARD_FLOOR, T, R_RISKFREE, IV_PRICING)
        c_fill, _, _ = bs_greek_calculator(V_FILL_PLAN, V_HARD_FLOOR, T, R_RISKFREE, IV_PRICING)
        # Cap limit is the available space for the initial position given the fill budget
        cap_limit = c0 / (c0 + c_fill) if c0 + c_fill > 0 else 0.0

        results.append({
            "Days": days,
            "Option_Price": c_price,
            "Kelly_Target": kelly_target,
            "Pilot_Cap": cap_limit,
            "Diff": kelly_target - cap_limit
        })

    df = pd.DataFrame(results)

    if df.empty:
        st.warning("未找到有效数据进行求解。请检查输入参数。")
        return

    best_idx = df['Diff'].abs().idxmin()
    best_row = df.loc[best_idx]

    st.success("✅ 最优期限计算完成。")

    col_r1, col_r2, col_r3 = st.columns(3)
    with col_r1:
        st.metric("最优期限", f"{int(best_row['Days'])} 天", f"~{best_row['Days']/30.4:.1f} 月")
    with col_r2:
        st.metric("建议分配比例 (Cap)", f"{best_row['Pilot_Cap']:.2%}")
    with col_r3:
        st.metric("期权价格 (BS 估值)", f"${best_row['Option_Price']:.2f}")

    # --- Plotting ---
    st.markdown("---")
    st.markdown("##### 攻守平衡曲线图")
    st.caption("最优解为进攻曲线 (Target Kelly) 与防守上限 (Pilot Cash Cap) 的交点。")

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(df['Days'], df['Kelly_Target'], label=f'进攻曲线: K={K_FACTOR:.2f} Kelly 比例',
             color='blue', linewidth=2, linestyle='--')

    ax.plot(df['Days'], df['Pilot_Cap'], label='防守上限: 初始补仓容量 (1:1)',
             color='red', linewidth=2)

    ax.scatter(best_row['Days'], best_row['Pilot_Cap'], color='green', s=150, zorder=5, label='最优期限点')

    ax.annotate(
        f"最优平衡点\n{int(best_row['Days'])} 天\n{best_row['Pilot_Cap']:.1%} 仓位",
        xy=(best_row['Days'], best_row['Pilot_Cap']),
        xytext=(best_row['Days']+100, best_row['Pilot_Cap']+0.1),
        arrowprops=dict(facecolor='black', shrink=0.05),
        fontsize=10, fontweight='bold'
    )

    ax.set_title(f"最优期限求解器: {ticker}", fontsize=14)
    ax.set_xlabel("距离到期日 (天)", fontsize=12)
    ax.set_ylabel("头寸分配百分比", fontsize=12)

    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # --- Next Step Warning ---
    st.markdown("---")
    st.warning(f"""
        ⚠️ **下一步行动 (关键)**：请将最优期限对应的 **真实期权合约价格**、**Delta** 和 **Theta 绝对值**，
        作为 **Step 1** 主仓位计算器的最终输入，进行精确的仓位测算。
    """)


# --- Page 3: Main Calculator (Dashboard) ---
def page_dashboard(ticker, lambda_val, sigma_val, r_f, k_factor, beta, P, V_target, V_hard, opt_price, delta, theta, V_fill, iv_pricing, days_to_expiry):
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
    L = delta * (P / opt_price) if opt_price > 0 else 0
    theta_annual = (theta / opt_price) * 252.0 if opt_price > 0 else 0

    # B. Returns
    mu_stock = lambda_val * np.log(V_target / P)
    mu_leaps = mu_stock * L
    ERP = mu_leaps - r_f - theta_annual

    # C. Risk
    sigma_leaps = sigma_val * L
    variance_leaps = sigma_leaps ** 2

    # D. Alpha
    range_len = max(1e-9, V_target - V_hard)
    dist_from_floor = P - V_hard
    risk_ratio = max(0.0, min(1.0, dist_from_floor / range_len))
    alpha = 1.0 - (beta * risk_ratio)

    # E. Kelly Cash
    # Calculate the initial allocation based on user's k_factor (e.g., k=0.5)
    f_cash = (k_factor * alpha * ERP) / variance_leaps if (ERP > 0 and variance_leaps > 0) else 0.0
    f_cash = max(0.0, f_cash)

    # --- Display Results ---
    col_d, col_m = st.columns([1, 2])
    with col_d:
        st.subheader("核心结果")
        if ERP > 0:
            st.metric(
                label=f"初始 Kelly 分配 ({k_factor:.2f}K)",
                value=f"{f_cash:.2%}",
                delta=f"有效杠杆: {L:.2f}x"
            )
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
                **Alpha (信心折扣系数)** 是一个动态调节因子，用于根据当前股价**距离硬底的远近**来调整仓位。

                $$\alpha_i = 1 - \beta \cdot \left( \frac{P_i - P_{\text{floor}, i}}{V_i - P_{\text{floor}, i}} \right)$$

                * **当股价接近硬底 ($V_{\text{hard}}$) 时:** $\alpha \to 1.0$，信心最高，推荐分配全部 Kelly 仓位。
                * **当股价接近目标价 ($V_{\text{target}}$) 时:** $\alpha \to (1-\beta)$，折扣生效，Kelly 仓位被缩减，以保留利润。
            """)

        st.write(f"**LEAPS 年化波动率:** {sigma_leaps:.2%}")

    with col_m:
        st.subheader("情景分析 (固定杠杆)")
        st.caption("当价格跌向硬底时，仓位如何变化。")

        prices = np.linspace(V_hard, P, 50)
        allocations = []

        for p_sim in prices:
            dist = p_sim - V_hard
            rr = max(0.0, min(1.0, dist / range_len))
            a_sim = 1.0 - (beta * rr)
            mu_s = lambda_val * np.log(V_target / p_sim)
            mu_l = mu_s * L
            # Note: ERP here uses the current fixed L and theta_annual
            erp_sim = mu_l - r_f - theta_annual
            # We use k_factor for the chart to show current strategy's response
            if erp_sim > 0:
                val = (k_factor * a_sim * erp_sim) / variance_leaps
            else:
                val = 0
            allocations.append(max(0, val))

        chart_data = pd.DataFrame({
            "股价": prices,
            "建议分配比例": allocations
        })
        st.line_chart(chart_data, x="股价", y="建议分配比例", color="#FF4B4B")
        st.caption(f"曲线变化由 Alpha 信心系数 (Beta={beta:.2f}) 驱动，确保越接近硬底 ($V_{{hard}}$) 信心越高。")

    st.markdown("---")

    # --- F. Dynamic K-Factor Strategy Visualizer (NEW) ---
    st.subheader("💡 动态 K 值策略推演 (Dynamic K-Factor Matrix)")
    st.info(f"此图展示了当股价从当前 ${P} 下跌至补仓价 ${V_fill} 时，若设定不同的【最终目标 K 值】，总仓位将如何变化。")
    st.caption(f"假设：K 值随股价下跌线性递增。起点为当前设定的 K={k_factor}，终点为图例中的目标 K。")

    if P <= V_fill:
        st.warning(f"当前价格 ${P} 已低于或等于补仓价 ${V_fill}。建议直接采用目标 K 值进行配置，无需动态推演。")
    else:
        # Simulation Parameters
        sim_steps = 30
        sim_prices = np.linspace(P, V_fill, sim_steps)

        # Generate Target Ks: Start from current k_factor, step up to 1.0
        # Example: if k=0.5, we want [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        # We use a small epsilon to ensure 1.0 is included if step aligns
        step = 0.1
        if k_factor >= 1.0:
            target_ks = [1.0]
        else:
            # Create range
            targets = np.arange(k_factor, 1.0 + 1e-9, step)
            # Ensure 1.0 is strictly in the list if not already (due to float precision)
            if abs(targets[-1] - 1.0) > 1e-5:
                targets = np.append(targets, 1.0)
            target_ks = targets

        # Prepare Plot
        fig_sim, ax_sim = plt.subplots(figsize=(10, 6))

        # Color map - distinct colors
        colors = plt.cm.plasma(np.linspace(0, 0.9, len(target_ks)))

        for idx, target_k in enumerate(target_ks):
            allocations = []

            # Label generation
            if abs(target_k - k_factor) < 0.01:
                label_str = f"保持恒定 K={target_k:.1f}"
            else:
                label_str = f"目标 K={target_k:.1f}"

            hit_100_idx = -1

            for i, p_sim in enumerate(sim_prices):
                # 1. Linear Interpolation of K
                # Progress: 0.0 at Start(P), 1.0 at End(V_fill)
                progress = (P - p_sim) / (P - V_fill)
                current_sim_k = k_factor + (target_k - k_factor) * progress

                # 2. Recalculate Option/Greeks
                T_sim = days_to_expiry / 365.0
                c_sim, delta_sim, theta_sim_ann = bs_greek_calculator(p_sim, V_hard, T_sim, r_f, iv_pricing)

                val = 0.0
                if c_sim > 0:
                    theta_yield_sim = abs(theta_sim_ann) / c_sim
                    L_sim = delta_sim * (p_sim / c_sim)

                    mu_stock_sim = lambda_val * np.log(V_target / p_sim)
                    mu_leaps_sim = mu_stock_sim * L_sim
                    ERP_sim = mu_leaps_sim - r_f - theta_yield_sim

                    sigma_leaps_sim = sigma_val * L_sim
                    var_leaps_sim = sigma_leaps_sim ** 2

                    dist_sim = p_sim - V_hard
                    risk_ratio_sim = max(0.0, min(1.0, dist_sim / range_len))
                    alpha_sim = 1.0 - (beta * risk_ratio_sim)

                    if ERP_sim > 0 and var_leaps_sim > 0:
                        val = (current_sim_k * alpha_sim * ERP_sim) / var_leaps_sim

                val = max(0.0, val)
                allocations.append(val)

                # Track when it hits 100%
                if val >= 1.0 and hit_100_idx == -1:
                    hit_100_idx = i

            # Plot Logic
            # If hits 100%, we can truncate or just let it go high but clip visually
            # The prompt asks: "什么时候会加满（占到100%的话就停止循环进入下一个）"
            # Visually showing it capping at 100% is often better than stopping the line

            # Clip values for plotting but keep data integrity?
            # Let's plot actual values but limit Y axis to e.g. 1.2

            ax_sim.plot(sim_prices, allocations, label=label_str, color=colors[idx], linewidth=2)



        # Formatting
        ax_sim.set_title(f"不同 K 值递增策略下的仓位路径 (${P} -> ${V_fill})", fontsize=12)
        ax_sim.set_xlabel("股价 ($)", fontsize=10)
        ax_sim.set_ylabel("建议总仓位 % (f)", fontsize=10)

        # X Axis: Invert to show price dropping from Left to Right?
        # Standard financial charts usually have time/lower prices depending on context.
        # User wants "Drop". High -> Low.
        ax_sim.invert_xaxis()

        ax_sim.set_ylim(0, 1.5) # Allow seeing a bit above 100%
        ax_sim.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

        # 100% Line
        ax_sim.axhline(1.0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label="满仓线 (100%)")

        ax_sim.legend(loc='upper right', fontsize=8, framealpha=0.9)
        ax_sim.grid(True, alpha=0.3)

        st.pyplot(fig_sim)
        plt.close(fig_sim)

        st.markdown(f"""
        **图表说明：**
        * **X轴**：股价从当前价格逐渐下跌至补仓价。
        * **Y轴**：凯利公式计算出的建议总仓位。
        * **彩色线条**：代表不同的策略。例如 "目标 K=1.0" 意味着随着股价下跌，K 值从当前的 {k_factor} 线性增加到 1.0。
        * **交点/红线**：当线条触及 100% 红线时，意味着该策略下建议满仓，后续应停止加仓或仅维持仓位。
        """)

    st.markdown("---")

    # --- G. Stress Test (NEW FEATURE) ---
    st.subheader("⚠️ 压力测试 (Stress Test) - 账户净值模拟")
    st.caption(f"基于当前建议仓位 ({f_cash:.2%}) 的次日盈亏模拟")

    with st.expander("📊 点击展开：如果明天发生暴跌，我的账户将承受？", expanded=True):

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

        # We use Delta Approximation for simplicity: LEAPS Drop % ≈ Leverage * Stock Drop %
        # Assume a nominal account size of $100,000 for dollar loss display (optional but illustrative)
        NOMINAL_ACCOUNT_VALUE = 100000.0

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
                "标的跌幅": f"{stock_drop:.2%}",
                "LEAPS 预估跌幅": f"{leaps_drop_pct:.2%}",
                "账户总净值回撤": f"{account_impact_pct:.2%}",
                "预估亏损 (10万账户)": f"${account_loss_usd:,.0f}" if f_cash > 0 else "$0",
            })

        risk_df = pd.DataFrame(risk_table)
        st.table(risk_df)
        st.caption("*注：此处使用有效杠杆 (L) 进行线性估算，实际期权在暴跌中的跌幅可能因 Gamma/Vega 效应有所不同。仅供风控参考。如果 $3\sigma$ 亏损额让你感到恐慌，请在侧边栏调低 $k$ 值。")


    # --- Save to Portfolio Feature ---
    if opt_price > 0 and ERP > 0:
        st.markdown("---")
        st.subheader("💾 保存到组合")

        if st.button("➕ 保存当前配置到组合", type="primary"):
            asset_record = {
                'Ticker': ticker,
                'Raw_Kelly_Pct': f_cash,
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


# --- Page for Multi-Asset Normalization ---
def page_multi_asset_normalization(max_leverage_cap):
    st.title("⚖️ Step 2: 多标的组合管理 - 简单归一化")
    st.markdown("---")

    # --- USER REQUESTED CORRELATION GUIDANCE ---
    with st.expander("❓ 组合相关性与仓位上限 (C_max) 设定指南"):
        st.markdown(r"""
            组合中资产的相关性（Correlation）是确定最终总仓位上限 $C_{max}$ 的关键因素。

            **1. 高相关性资产 (例如：同板块股票或指数)**
            * **原则:** 当资产相关性高时，风险分散效果差。建议将原始 Kelly 值进行**内部加权平均**，而非简单相加，以此平均值作为 $C_{max}$ 或略高的上限。
            * **案例:** 如果资产A (Kelly $65\%$, 信心 $2$) 和资产B (Kelly $45\%$, 信心 $1$)，您可以考虑将最终上限 $C_{max}$ 设置为他们的**信心加权平均**：
                $$C_{max} \approx \frac{65\% \times 2 + 45\% \times 1}{2 + 1} \approx 58.33\%$$
            * **操作:** 将计算出的加权平均值（例如 $0.58$）作为 $C_{max}$ 阈值输入到左侧边栏的滑块中。

            **2. 低相关性资产 (例如：跨市场指数)**
            * **原则:** 风险分散效应显著，可以允许较高的总仓位。
            * **操作:** 可以将 $C_{max}$ 设置在 $80\%$ 到 $100\%$ 之间，让系统根据您设置的上限自动计算归一化后的仓位。

            *本计算器采用简单的线性归一化方法 (Final Pct = Raw Kelly $\times$ Scale Factor)，请根据您的组合相关性设置合理的 $C_{max}$。*
        """)
    st.markdown("---")
    # ------------------------------------------

    portfolio_data = st.session_state.get('portfolio_data')

    if not portfolio_data:
        st.warning("组合中没有资产。请回到 Step 1 计算并点击 '保存当前配置到组合'。")
        return

    df = pd.DataFrame(portfolio_data)

    # 1. Calculate Raw Exposure
    total_raw_exposure = df['Raw_Kelly_Pct'].sum()

    st.markdown(f"**总资产数量:** `{len(df)}`")
    st.markdown(f"**原始 Kelly 理论总仓位 (C_raw):** `{total_raw_exposure:.2%}`")
    st.markdown(f"**设置的现金上限 (C_max):** `{max_leverage_cap:.2%}`")

    # 2. Normalize Logic
    if total_raw_exposure > max_leverage_cap:
        scale_factor = max_leverage_cap / total_raw_exposure
        st.error(f"🚨 总仓位超限，已进行归一化缩放。缩放因子: {scale_factor:.4f}")
    else:
        scale_factor = 1.0
        st.success("✅ 总仓位在限制内。无需缩放。")

    # 3. Apply Normalization
    df['Final_Pct'] = df['Raw_Kelly_Pct'] * scale_factor

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
    # FIX: Use the NEW column name '最终仓位 %' because 'Final_Pct' was renamed in the previous step
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


# ==========================================
# 5. MAIN ROUTER
# ==========================================
st.set_page_config(page_title="统一凯利量化工具", layout="wide", page_icon="📈")


# --- Initialize Session State Defaults ---
default_vals = {
    'r_f': 0.037, 'k_factor': 0.50, 'beta': 0.20, 'P': 180.00,
    'V_target': 225.00, 'V_hard': 130.00, 'V_fill': 145.00,
    'iv_pricing': 0.5100, 'opt_price': 61.60, 'delta': 0.8446,
    'theta': 0.0425, 'ticker': "NVDA", 'lambda': 6.0393,
    'sigma': 0.6082, 'portfolio_data': [], 'window_days': 90,
    'days_to_expiry': 365 # Default 1 year
}

for key, default_val in default_vals.items():
    if key not in st.session_state:
        st.session_state[key] = default_val

# --- Sidebar Inputs ---
with st.sidebar:
    st.title("导航与全局参数")

    page = st.radio("选择工具页面",
                    ("Step 0: 市场诊断",
                     "Step 0.5: 最优期限求解",
                     "Step 1: 主仓位计算器",
                     "Step 2: 多标的组合管理"),
                    key='page_select', index=0)

    st.header("1. 资产与统计数据")
    ticker = st.text_input("股票代码 (Ticker)", value=st.session_state.ticker, key='ticker_global').upper()

    if st.button("获取历史统计数据"):
        if 'get_ou_for_ticker' in globals() and 'get_sigma' in globals():
            try:
                with st.spinner("Calculating OU Params & Volatility..."):
                    ou = get_ou_for_ticker(ticker, window=90)
                    new_lambda = ou["lambda"] * 252.0

                    sigma_dict, _, _, rolling_series_dict = get_sigma(
                        [ticker], period="5y", window=252, percentile=0.85, annualize=True, safety_lock=True
                    )
                    new_sigma = sigma_dict.get(ticker)

                    st.session_state['lambda'] = new_lambda
                    st.session_state['sigma'] = new_sigma
                    st.session_state['ticker'] = ticker

                    st.session_state['sigma_rolling_data'] = rolling_series_dict
                    st.session_state['sigma_dict'] = sigma_dict

                    st.info(f"✅ 已检测到滚动窗口统计值: Lambda (λ) = **{new_lambda:.4f}**, Sigma (σ) = **{new_sigma:.4f}**")
                    st.warning("⚠️ 请评估该值是否过于激进，确认后请手动输入到左侧边栏以应用到后续计算")

            except Exception as e:
                st.error(f"Error fetching data: {e}")
            finally:
                pass
        else:
            st.error("依赖模块 (lambda_tools.py / sigma_tools.py) 未导入，无法获取历史数据。")

    st.divider()
    lambda_val = st.number_input("年化 Lambda (λ)", value=st.session_state['lambda'], key='lambda_global', format="%.4f",
                               help="【均值回归动力】数值越大，修复越快。若图表显示 Lambda 处于历史极高位(>8.0)，建议手动调低至 5.0 左右以防噪音。")
    sigma_val = st.number_input("年化 Sigma (σ)", value=st.session_state['sigma'], key='sigma_global', format="%.4f",
                              help="【保守波动率】通常取历史 85% 分位数。用于计算凯利公式的分母(风险)。")

    st.header("2. 策略与市场参数 (动态)")

    current_lambda = lambda_val
    current_sigma = sigma_val
    current_r_f = st.session_state.r_f
    current_k_factor = st.session_state.k_factor
    current_beta = st.session_state.beta
    current_P = st.session_state.P
    current_V_target = st.session_state.V_target
    current_V_hard = st.session_state.V_hard
    current_V_fill = st.session_state.V_fill
    current_iv_pricing = st.session_state.iv_pricing
    current_opt_price = st.session_state.opt_price
    current_delta = st.session_state.delta
    current_theta = st.session_state.theta
    current_window_days = st.session_state.window_days
    current_max_cap = st.session_state.get('c_max_slider', 1.0)
    current_days_to_expiry = st.session_state.get('days_to_expiry', 365)


    if page == "Step 0: 市场诊断":
        st.subheader("诊断特有参数")
        window_days = st.slider("滚动窗口 (交易日)", min_value=30, max_value=252, value=st.session_state.window_days, key='window_days_diag')
        st.session_state['window_days'] = window_days
        current_window_days = window_days
    else:
        if page == "Step 1: 主仓位计算器":
            st.subheader("2.1 策略约束")
            current_r_f = st.number_input("无风险利率 (r_f)", value=st.session_state.r_f, key='r_f_dash', format="%.3f")
            current_k_factor = st.slider("凯利分数 (k)", 0.1, 1.0, st.session_state.k_factor, 0.05, key='k_dash',
                                         help="【激进程度】0.5 = 推荐标准 (半凯利)，最大化长期几何增长率。1.0 = 满凯利，仅建议在极度低估时用于回补。")
            current_beta = st.slider("估值折扣系数 (beta)", 0.0, 1.0, st.session_state.beta, 0.05, key='beta_dash',
                                     help="【止盈速率/信心衰减】0.2 = 推荐。股价接近目标价时，Alpha 保留 80% 权重。1.0 = 到达目标价即清仓。")

            st.subheader("2.2 市场与合约参数")
            current_P = st.number_input("当前股价 P ($)", value=st.session_state.P, key='P_dash', format="%.2f")
            current_V_target = st.number_input("目标价 V ($)", value=st.session_state.V_target, key='V_target_dash', format="%.2f",
                                               help="【公允价值】你认为标的最终应值多少钱？影响预期收益(Drift)。")
            current_V_hard = st.number_input("硬底 V_hard ($)", value=st.session_state.V_hard, key='V_hard_dash', format="%.2f",
                                             help="【止损锚点】极端悲观下绝对不会跌破的价格。建议买入 Strike 接近此价格的期权，物理锁死尾部风险。")

            # Added V_fill for dynamic calculation
            current_V_fill = st.number_input("计划补仓价 V_fill ($)", value=st.session_state.V_fill, key='V_fill_dash', format="%.2f",
                                            help="【满仓线】当股价跌至此价格时，总仓位将提升至 1.0K 的理论最大值。")


            st.divider()
            # Added Days to Expiry for BS calc
            current_days_to_expiry = st.number_input("距离到期日 (Days)", value=st.session_state.days_to_expiry, key='dte_dash', step=1)
            current_iv_pricing = st.number_input("期权定价 IV", value=st.session_state.iv_pricing, key='iv_dash', format="%.4f", help="用于在动态推演中重新计算期权价格。")

            current_opt_price = st.number_input("LEAPS Price ($)", value=st.session_state.opt_price, key='opt_price_dash', format="%.2f")
            current_delta = st.number_input("Delta", value=st.session_state.delta, key='delta_dash', format="%.4f")
            current_theta = st.number_input("Daily Theta (Abs)", value=st.session_state.theta, key='theta_dash', format="%.4f")

            st.session_state.r_f = current_r_f
            st.session_state.k_factor = current_k_factor
            st.session_state.beta = current_beta
            st.session_state.P = current_P
            st.session_state.V_target = current_V_target
            st.session_state.V_hard = current_V_hard
            st.session_state.V_fill = current_V_fill # Store V_fill
            st.session_state.opt_price = current_opt_price
            st.session_state.delta = current_delta
            st.session_state.theta = current_theta
            st.session_state.days_to_expiry = current_days_to_expiry # Store DTE
            st.session_state.iv_pricing = current_iv_pricing # Store IV

        elif page == "Step 0.5: 最优期限求解":
            st.subheader("2.1 策略约束")
            current_r_f = st.number_input("无风险利率 (r_f)", value=st.session_state.r_f, key='r_f_solver', format="%.3f")

            # --- ADDED K and Beta Inputs to Solver Sidebar ---
            current_k_factor = st.slider("凯利分数 (k)", 0.1, 1.0, st.session_state.k_factor, 0.05, key='k_solver_factor',
                                         help="【激进程度】影响进攻曲线 (Kelly) 的起始位置。")
            current_beta = st.slider("估值折扣系数 (beta)", 0.0, 1.0, st.session_state.beta, 0.05, key='beta_solver',
                                     help="【信心衰减】影响 Kelly 计算中 Alpha 的折扣率。")
            # --- END ADDED ---

            st.subheader("2.2 市场与定价参数")
            current_P = st.number_input("当前股价 P ($)", value=st.session_state.P, key='P_solver', format="%.2f")
            current_V_target = st.number_input("目标价 V ($)", value=st.session_state.V_target, key='V_target_solver', format="%.2f")
            current_V_hard = st.number_input("硬底 V_hard ($)", value=st.session_state.V_hard, key='V_hard_solver', format="%.2f")
            current_V_fill = st.number_input("计划补仓价 V_fill ($)", value=st.session_state.V_fill, key='V_fill_solver', format="%.2f")
            current_iv_pricing = st.number_input("期权定价波动率 (IV)", value=st.session_state.iv_pricing, key='iv_pricing_solver', format="%.4f")

            st.session_state.r_f = current_r_f
            st.session_state.k_factor = current_k_factor # Update k_factor and beta in session state for consistency
            st.session_state.beta = current_beta
            st.session_state.P = current_P
            st.session_state.V_target = current_V_target
            st.session_state.V_hard = current_V_hard
            st.session_state.V_fill = current_V_fill
            st.session_state.iv_pricing = current_iv_pricing

        elif page == "Step 2: 多标的组合管理":
            st.subheader("2.1 组合约束")
            max_leverage_cap = st.slider("总仓位上限 (C_max)", 0.5, 2.0, 1.0, 0.05, key='c_max_slider', help="控制总现金分配不超过 C_max * 100%")
            st.info("数据来源于 Step 1 中点击 '保存到组合' 的记录。")
            current_max_cap = max_leverage_cap

# --- Page Routing ---
if page == "Step 0: 市场诊断":
    page_diagnosis(ticker, current_window_days)

elif page == "Step 0.5: 最优期限求解":
    if current_V_target <= current_V_hard:
        st.error("错误: 目标价必须高于硬底。")
    elif current_lambda is None or current_sigma is None:
        st.error("请先在侧边栏获取 Lambda/Sigma 统计数据。")
    else:
        page_solver(current_P, current_V_target, current_V_hard, current_V_fill, current_lambda, current_sigma, current_iv_pricing, current_r_f, ticker, current_k_factor, current_beta)

elif page == "Step 1: 主仓位计算器":
    if current_lambda is None or current_sigma is None:
        st.error("请先在侧边栏获取 Lambda/Sigma 统计数据。")
    elif current_opt_price <= 0 or current_delta <= 0:
        st.warning("请在侧边栏输入有效的期权合约数据。")
    else:
        # Pass new arguments to page_dashboard
        page_dashboard(ticker, current_lambda, current_sigma, current_r_f, current_k_factor, current_beta, current_P, current_V_target, current_V_hard, current_opt_price, current_delta, current_theta, current_V_fill, current_iv_pricing, current_days_to_expiry)

elif page == "Step 2: 多标的组合管理":
    max_leverage_cap = st.session_state.get('c_max_slider', 1.0)
    page_multi_asset_normalization(max_leverage_cap)