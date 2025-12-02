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
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import os
import sys

def configure_chinese_font():
    """
    配置中文字体,兼容本地和 Streamlit Cloud 环境
    """
    try:
        # 方案 1: 尝试使用项目自带字体
        FONT_FILE_NAME = 'SimHei.ttf'
        FONT_PATH = os.path.join(os.getcwd(), "fonts", FONT_FILE_NAME)

        if os.path.exists(FONT_PATH):
            print(f"Found custom font at: {FONT_PATH}")
            fm.fontManager.addfont(FONT_PATH)
            prop = fm.FontProperties(fname=FONT_PATH)
            font_name = prop.get_name()
            plt.rcParams['font.sans-serif'] = [font_name, 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            print(f"Successfully loaded custom font: {font_name}")
            return

        # 方案 2: Streamlit Cloud - 使用系统中文字体
        print("Custom font not found, trying system fonts...")

        # Linux 系统常见中文字体列表
        chinese_fonts = [
            'WenQuanYi Micro Hei',    # 文泉驿微米黑
            'WenQuanYi Zen Hei',      # 文泉驿正黑
            'Noto Sans CJK SC',       # 思源黑体
            'Noto Sans CJK TC',
            'SimHei',                  # 黑体
            'Microsoft YaHei',         # 微软雅黑
            'STHeiti',                 # 华文黑体
            'Arial Unicode MS',
        ]

        # 获取系统可用字体
        available_fonts = set([f.name for f in fm.fontManager.ttflist])
        print(f"Available fonts on system: {len(available_fonts)}")

        # 查找可用的中文字体
        found_font = None
        for font in chinese_fonts:
            if font in available_fonts:
                found_font = font
                print(f"Found system Chinese font: {font}")
                break

        if found_font:
            plt.rcParams['font.sans-serif'] = [found_font, 'DejaVu Sans']
        else:
            # 方案 3: 安装 Noto Sans (最可靠)
            print("No Chinese font found, using fallback with Noto Sans SC")
            plt.rcParams['font.sans-serif'] = [
                'Noto Sans CJK SC',
                'DejaVu Sans',
                'Arial'
            ]

        plt.rcParams['axes.unicode_minus'] = False
        print("Font configuration completed")

    except Exception as e:
        print(f"Font configuration error: {e}")
        # 最终后备方案
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False

# 执行字体配置
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
# 4. PAGE LOGIC FUNCTIONS (封装各应用逻辑)
# ==========================================

# --- Page 1: 市场诊断 (Rolling Analysis) ---
def page_diagnosis(ticker, window_days):
    st.title("📈 Step 0: 市场诊断 - 滚动分析")
    st.subheader(f"资产: {ticker} | 滚动窗口: {window_days} 交易日")
    st.markdown("---")

    # --- 用户提示：Step 0 指引 ---
    with st.expander("❓ Step 0：市场诊断指引 (验证均值回归)"):
        st.markdown("""
            这是**风险控制的第一步**，用于验证均值回归假设是否成立，以及评估回归动力 ($\lambda$) 的可靠性。
            **核心目标：**
            1.  **判断低估是否真实：** 查看 PE Ratio 曲线是否明显低于滚动均线，确认存在回归空间。
            2.  **评估 $\lambda$ 质量：** 检查 Lambda 曲线最右端的值是否远高于其历史平均水平（虚高）。如果是，后续 Step 1 中应**手动调低 $\lambda$**。
            3.  **确认时间可行性：** 检查 Monte Carlo 模拟，确认 90% 概率触摸目标所需的最短时间，以此作为 **LEAPS 选品的期限底线**。
        """)
    st.markdown("---")
    # ----------------------------

    # --- Data Loading uses the consistent project_root ---
    pe_csv_path = os.path.join(project_root, "pe_csv", f"{ticker}_pe.csv")
    if not os.path.exists(pe_csv_path):
        st.warning(f"警告: 找不到 {ticker}_pe.csv 文件进行滚动分析。请确保数据位于: {os.path.join(os.path.basename(project_root), 'pe_csv/')}")
        return

    try:
        df = pd.read_csv(pe_csv_path, parse_dates=['date'])
        df = df.sort_values('date').reset_index(drop=True)
    except Exception as e:
        st.error(f"读取 PE 数据失败: {e}")
        return

    # --- 1. 计算滚动指标 ---
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

    # --- 2. 诊断报告 (简化) ---
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

    # --- 3. Monte Carlo 模拟 ---
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

    # --- 4. Plotting (三张图表) ---
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


# --- Page 2: 最优期限求解 (Optimal Expiry Solver) ---
def page_solver(P_CURRENT, V_TARGET, V_HARD_FLOOR, V_FILL_PLAN, LAMBDA, SIGMA_ASSET, IV_PRICING, R_RISKFREE, ticker):
    st.title("🎯 Step 0.5: 最优期限求解器")
    st.subheader(f"资产: {ticker} | P={P_CURRENT}")
    st.markdown("---")

    # --- 用户提示：Step 0.5 指引 ---
    with st.expander("❓ Step 0.5：求解器原理与下一步行动"):
        st.markdown("""
            求解器旨在找到一个**攻守平衡点**：即在满足凯利增长速度要求的同时，预留出在计划补仓价 ($V_{fill}$) 进行 **1:1 补仓的充足现金**。
            * **进攻曲线 (Offense)**：基于 Kelly 理论，期限越长，波动率惩罚越低，建议仓位越高。
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

        kelly_full = calculate_single_asset_kelly_ratio(
            P_CURRENT, c_price, c_delta, c_theta_annual, V_TARGET, V_HARD_FLOOR, LAMBDA, SIGMA_ASSET, R_RISKFREE, beta=0.2
        )
        kelly_target = kelly_full * 0.5

        c0, _, _ = bs_greek_calculator(P_CURRENT, V_HARD_FLOOR, T, R_RISKFREE, IV_PRICING)
        c_fill, _, _ = bs_greek_calculator(V_FILL_PLAN, V_HARD_FLOOR, T, R_RISKFREE, IV_PRICING)
        cap_limit = c0 / (c0 + c_fill) if c0 + c_fill > 0 else 0.0

        results.append({
            "Days": days,
            "Option_Price": c_price,
            "Kelly_Half": kelly_target,
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

    # --- Plotting (修复中文标签) ---
    st.markdown("---")
    st.markdown("##### 攻守平衡曲线图")
    st.caption("最优解为进攻曲线 (0.5 * Kelly) 与防守上限 (Pilot Cash Cap) 的交点。")

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(df['Days'], df['Kelly_Half'], label='进攻曲线: 0.5 * Kelly 比例',
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


# --- Page 3: 主仓位计算器 (App Dashboard) ---
def page_dashboard(ticker, lambda_val, sigma_val, r_f, k_factor, beta, P, V_target, V_hard, opt_price, delta, theta):
    st.title("🌌 Step 1: 凯利 LEAPS 仓位主计算器")
    st.markdown("---")

    # --- 用户提示：Step 1 指引 ---
    with st.expander("❓ Step 1：核心目标 (买多少？)"):
        st.markdown("""
            本计算器是系统的**核心步骤**。它将**均值回归动力** ($\lambda$) 与 **LEAPS 的杠杆风险** ($L^2\sigma^2$) 相结合，计算出在您设定的风险偏好 (k) 和信心 ($\\alpha$) 下，能够**最大化长期几何增长率**的现金投入比例。
            **核心判断：** 确保 **净优势 (ERP)** 为正值。如果 ERP < 0，即使是理论上最优的杠杆，也无法覆盖期权的租金成本 ($\\theta$) 和无风险利率 ($r_f$)，应避免开仓。
            *输入前，请确保您已从 Step 0 或券商处获取了**准确的合约数据**。*
        """)
    st.markdown("---")
    # ----------------------------

    # --- A. Leverage & Cost ---
    if opt_price > 0:
        L = delta * (P / opt_price)
        theta_annual = (theta / opt_price) * 252.0

        # --- B. Returns ---
        mu_stock = lambda_val * np.log(V_target / P)
        mu_leaps = mu_stock * L
        ERP = mu_leaps - r_f - theta_annual

        # --- C. Risk ---
        sigma_leaps = sigma_val * L
        variance_leaps = sigma_leaps ** 2

        # --- D. Alpha ---
        range_len = max(1e-9, V_target - V_hard)
        dist_from_floor = P - V_hard
        risk_ratio = max(0.0, min(1.0, dist_from_floor / range_len))
        alpha = 1.0 - (beta * risk_ratio)

        # --- E. Kelly ---
        if ERP > 0 and variance_leaps > 0:
            f_cash = (k_factor * alpha * ERP) / variance_leaps
        else:
            f_cash = 0.0

        f_cash = max(0.0, f_cash)

        # --- Display Results ---
        col_d, col_m = st.columns([1, 2])
        with col_d:
            st.subheader("核心结果")
            if ERP > 0:
                st.metric(
                    label="Kelly Allocation %",
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

                    $$ \text{ERP}_i = (\mu_{\text{stock}, i} \cdot L_i) - r_f - \theta_{\text{annual}, i} $$

                    * **进攻端:** 均值回归预期收益 $\times$ 杠杆 $L$
                    * **防守端:** 减去资金成本 $r_f$ 和时间损耗 $\theta_{\text{annual}}$

                    **如果 ERP > 0，则表明这是一笔具有正期望值的交易。**
                """)

            # --- Alpha Explanation ---
            st.write(f"**信心系数 (Alpha):** {alpha:.3f}")
            with st.expander("❓ 信心系数 (Alpha) 解读"):
                st.markdown(r"""
                    **Alpha (信心折扣系数)** 是一个动态调节因子，用于根据当前股价**距离硬底的远近**来调整仓位。

                    $$ \alpha_i = 1 - \beta \cdot \left( \frac{P_i - P_{\text{floor}, i}}{V_i - P_{\text{floor}, i}} \right) $$

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
                erp_sim = mu_l - r_f - theta_annual
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
                $$ C_{max} \approx \frac{65\% \times 2 + 45\% \times 1}{2 + 1} \approx 58.33\% $$
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

    # 1. 计算原始总风险暴露
    total_raw_exposure = df['Raw_Kelly_Pct'].sum()

    st.markdown(f"**总资产数量:** `{len(df)}`")
    st.markdown(f"**原始 Kelly 理论总仓位 (C_raw):** `{total_raw_exposure:.2%}`")
    st.markdown(f"**设置的现金上限 (C_max):** `{max_leverage_cap:.2%}`")

    # 2. 归一化逻辑
    if total_raw_exposure > max_leverage_cap:
        scale_factor = max_leverage_cap / total_raw_exposure
        st.error(f"🚨 总仓位超限，已进行归一化缩放。缩放因子: {scale_factor:.4f}")
    else:
        scale_factor = 1.0
        st.success("✅ 总仓位在限制内。无需缩放。")

    # 3. 应用归一化
    df['Final_Pct'] = df['Raw_Kelly_Pct'] * scale_factor

    # 4. 格式化输出
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


# ==========================================
# 5. MAIN ROUTER (统一入口)
# ==========================================
st.set_page_config(page_title="统一凯利量化工具", layout="wide", page_icon="📈")


# --- 初始化 Session State 中的默认值 ---
default_vals = {
    'r_f': 0.037, 'k_factor': 0.50, 'beta': 0.20, 'P': 180.00,
    'V_target': 225.00, 'V_hard': 130.00, 'V_fill': 145.00,
    'iv_pricing': 0.5100, 'opt_price': 61.60, 'delta': 0.8446,
    'theta': 0.0425, 'ticker': "NVDA", 'lambda': 6.0393,
    'sigma': 0.6082, 'portfolio_data': [], 'window_days': 90
}

for key, default_val in default_vals.items():
    if key not in st.session_state:
        st.session_state[key] = default_val

# --- 侧边栏统一输入 (Global Inputs) ---
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
    lambda_val = st.number_input("年化 Lambda (λ)", value=st.session_state['lambda'], key='lambda_global', format="%.4f")
    sigma_val = st.number_input("年化 Sigma (σ)", value=st.session_state['sigma'], key='sigma_global', format="%.4f")

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


    if page == "Step 0: 市场诊断":
        st.subheader("诊断特有参数")
        window_days = st.slider("滚动窗口 (交易日)", min_value=30, max_value=252, value=st.session_state.window_days, key='window_days_diag')
        st.session_state['window_days'] = window_days
        current_window_days = window_days
    else:
        if page == "Step 1: 主仓位计算器":
            st.subheader("2.1 策略约束")
            current_r_f = st.number_input("无风险利率 (r_f)", value=st.session_state.r_f, key='r_f_dash', format="%.3f")
            current_k_factor = st.slider("凯利分数 (k)", 0.1, 1.0, st.session_state.k_factor, 0.05, key='k_dash')
            current_beta = st.slider("估值折扣系数 (beta)", 0.0, 1.0, st.session_state.beta, 0.05, key='beta_dash')

            st.subheader("2.2 市场与合约参数")
            current_P = st.number_input("当前股价 P ($)", value=st.session_state.P, key='P_dash', format="%.2f")
            current_V_target = st.number_input("目标价 V ($)", value=st.session_state.V_target, key='V_target_dash', format="%.2f")
            current_V_hard = st.number_input("硬底 V_hard ($)", value=st.session_state.V_hard, key='V_hard_dash', format="%.2f")

            st.divider()
            current_opt_price = st.number_input("LEAPS Price ($)", value=st.session_state.opt_price, key='opt_price_dash', format="%.2f")
            current_delta = st.number_input("Delta", value=st.session_state.delta, key='delta_dash', format="%.4f")
            current_theta = st.number_input("Daily Theta (Abs)", value=st.session_state.theta, key='theta_dash', format="%.4f")

            st.session_state.r_f = current_r_f
            st.session_state.k_factor = current_k_factor
            st.session_state.beta = current_beta
            st.session_state.P = current_P
            st.session_state.V_target = current_V_target
            st.session_state.V_hard = current_V_hard
            st.session_state.opt_price = current_opt_price
            st.session_state.delta = current_delta
            st.session_state.theta = current_theta

        elif page == "Step 0.5: 最优期限求解":
            st.subheader("2.1 策略约束")
            current_r_f = st.number_input("无风险利率 (r_f)", value=st.session_state.r_f, key='r_f_solver', format="%.3f")

            st.subheader("2.2 市场与定价参数")
            current_P = st.number_input("当前股价 P ($)", value=st.session_state.P, key='P_solver', format="%.2f")
            current_V_target = st.number_input("目标价 V ($)", value=st.session_state.V_target, key='V_target_solver', format="%.2f")
            current_V_hard = st.number_input("硬底 V_hard ($)", value=st.session_state.V_hard, key='V_hard_solver', format="%.2f")
            current_V_fill = st.number_input("计划补仓价 V_fill ($)", value=st.session_state.V_fill, key='V_fill_solver', format="%.2f")
            current_iv_pricing = st.number_input("期权定价波动率 (IV)", value=st.session_state.iv_pricing, key='iv_pricing_solver', format="%.4f")

            st.session_state.r_f = current_r_f
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
        page_solver(current_P, current_V_target, current_V_hard, current_V_fill, current_lambda, current_sigma, current_iv_pricing, current_r_f, ticker)

elif page == "Step 1: 主仓位计算器":
    if current_lambda is None or current_sigma is None:
        st.error("请先在侧边栏获取 Lambda/Sigma 统计数据。")
    elif current_opt_price <= 0 or current_delta <= 0:
        st.warning("请在侧边栏输入有效的期权合约数据。")
    else:
        page_dashboard(ticker, current_lambda, current_sigma, current_r_f, current_k_factor, current_beta, current_P, current_V_target, current_V_hard, current_opt_price, current_delta, current_theta)

elif page == "Step 2: 多标的组合管理":
    max_leverage_cap = st.session_state.get('c_max_slider', 1.0)
    page_multi_asset_normalization(max_leverage_cap)