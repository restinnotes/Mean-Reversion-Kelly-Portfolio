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
    # Define placeholder functions to prevent crashes if imports fail completely
    # This is crucial for Streamlit apps running in environments without custom module support
    def get_ou_for_ticker(*args, **kwargs): return None
    def calculate_ou_params(*args, **kwargs): return None
    def get_sigma(*args, **kwargs): return ({}, {}, {}, {})
    def bs_greek_calculator(*args, **kwargs): return (0, 0, 0)
    def calculate_single_asset_kelly_ratio(*args, **kwargs): return 0.0
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
def analyze_risk_reward(paths, current_pe, days_map):
    """
    计算 Hold (持有到底) 和 Touch (触碰高点) 的风险收益分布
    """
    results = []
    max_sim_days = paths.shape[0] - 1

    for label, day in days_map.items():
        if day > max_sim_days: continue

        # --- A. HOLD 逻辑 (持有到底) ---
        final_values = paths[day]
        # 1. 亏损概率
        prob_loss = np.mean(final_values < current_pe)
        # 2. 10% 底线 (Worst Case)
        worst_10_val = np.percentile(final_values, 10)
        worst_10_pnl = (worst_10_val - current_pe) / current_pe
        # 3. 预期收益
        expected_val = np.mean(final_values)
        expected_pnl = (expected_val - current_pe) / current_pe

        # --- B. TOUCH 逻辑 (触碰高点) ---
        # 路径切片: [0..day]
        path_slice = paths[:day+1, :]
        # 每条路径在期间的最高点
        max_values = np.max(path_slice, axis=0)
        # 4. 10% 高点 (Best Case / Lucky Case)
        lucky_10_val = np.percentile(max_values, 90)
        lucky_10_pnl = (lucky_10_val - current_pe) / current_pe

        results.append({
            "时间窗口": label,
            "亏损概率 (Loss%)": prob_loss,
            "10%底线 (Hold)": worst_10_pnl,
            "预期收益 (Exp)": expected_pnl,
            "10%高点 (Touch)": lucky_10_pnl
        })

    return pd.DataFrame(results)

# ---------------------------------------------------------
# 请用此【逻辑重构版】替换 code/app_unified_zh.py 中的 page_diagnosis
# ---------------------------------------------------------

def page_diagnosis(ticker, window_days):
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

    # --- Data Loading ---
    pe_csv_path = os.path.join(project_root, "pe_csv", f"{ticker}_pe.csv")
    if not os.path.exists(pe_csv_path):
        st.warning(f"警告: 找不到 {ticker}_pe.csv 文件。")
        return

    try:
        df = pd.read_csv(pe_csv_path, parse_dates=['date'])
        df = df.sort_values('date').reset_index(drop=True)
    except Exception as e:
        st.error(f"读取 PE 数据失败: {e}")
        return

    # --- 1. Calculate Rolling Metrics ---
    if len(df) < window_days:
        st.warning("数据不足。")
        return

    df['rolling_mean'] = df['value'].rolling(window=window_days).mean()

    dates_hist = []; lambdas_annual_hist = []; half_lives_hist = []; sigmas_daily_hist = []
    start_index = window_days - 1

    if 'calculate_ou_params' in globals() and calculate_ou_params:
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
        st.error("依赖模块 (lambda_tools.py) 未导入或函数缺失。")
        return

    if not lambdas_annual_hist:
        st.warning("数据不足，无法计算指标。")
        return

    current_lambda = lambdas_annual_hist[-1]
    current_hl = half_lives_hist[-1]
    current_pe = df['value'].iloc[-1]
    current_mean = df['rolling_mean'].iloc[-1]
    current_sigma_daily = sigmas_daily_hist[-1]

    # New code for T-stat calculation and display logic START

    # 获取 t_stat
    # 获取用于计算 current_lambda 的最新窗口数据
    ou_last_snapshot = calculate_ou_params(df['value'].iloc[-window_days:])
    current_t_stat = ou_last_snapshot.get('t_stat', 0.0) if ou_last_snapshot else 0.0

    # 判定置信度文案
    if current_t_stat < -2.86:
        conf_label = "⭐⭐⭐ 极高 (Strong)"
        conf_color = "green"
        conf_help = "统计显著性 > 95%。拒绝随机游走假设，均值回归特征非常明显。"
    elif current_t_stat < -1.90:
        conf_label = "⭐⭐ 较高 (Moderate)"
        conf_color = "orange"
        conf_help = "统计显著性 > 90%。均值回归特征存在，但需留意。"
    else:
        conf_label = "⚠️ 存疑 (Weak)"
        conf_color = "red"
        conf_help = f"T-Stat ({current_t_stat:.2f}) 不显著。当前走势可能接近随机游走，Lambda 值参考意义下降。"

    # New code for T-stat calculation and display logic END

    # 将 Lambda 存入 Session 供后续使用
    if st.session_state.ticker == ticker:
        st.session_state['lambda'] = current_lambda

    # =========================================================
    # Part 1: 参数验证与历史回溯 (The Gatekeeper)
    # 这一部分必须放在前面，作为“体检报告”
    # =========================================================
    st.subheader("1. 核心参数验证 (Diagnosis)")

    col_d1, col_d2, col_d3 = st.columns(3) # 改为 3 列

    with col_d1:
        st.markdown("**估值偏离度**")
        st.code(f"当前 PE: {current_pe:.2f}")
        st.metric("均值偏离", f"{(current_pe - current_mean)/current_mean:.1%}")

    with col_d2:
        st.markdown("**回归动力 (Lambda)**")
        st.code(f"λ: {current_lambda:.4f}")
        st.caption(f"半衰期: {current_hl:.1f} 天")

    with col_d3:
        st.markdown("**统计可信度 (ADF Test)**")
        st.markdown(f":{conf_color}[**{conf_label}**]")
        st.caption(f"T-Stat: {current_t_stat:.2f}", help=conf_help)

    # --- 历史图表 (Visual Verification) ---
    # Plot 1: PE Context
    plot_df = df.iloc[start_index:].copy()
    plot_df['Lambda'] = lambdas_annual_hist
    plot_df['Half_Life'] = half_lives_hist
    plot_df.set_index('date', inplace=True)

    fig1, ax0 = plt.subplots(figsize=(10, 3))
    ax0.plot(plot_df.index, plot_df['value'], 'k', alpha=0.8, label='PE')
    ax0.plot(plot_df.index, plot_df['rolling_mean'], 'b--', label=f'{window_days}日均线')
    ax0.set_title(f'{ticker} 估值偏离度 (验证: 低估是否真实？)', fontsize=10)
    ax0.legend(loc='upper left'); ax0.grid(True, alpha=0.3)
    st.pyplot(fig1)
    plt.close(fig1)

    # Plot 2: Lambda History (Critical Check)
    lambda_80 = np.percentile(lambdas_annual_hist, 80)

    fig2, ax1 = plt.subplots(figsize=(10, 3))
    ax1.plot(plot_df.index, plot_df['Lambda'], color='#1f77b4', label='Lambda')
    ax1.axhline(lambda_80, color='r', linestyle='--', label=f'80%分位 ({lambda_80:.1f})')
    ax1.set_title('Lambda 历史走势 (验证: 是否处于不可持续的极高位？)', fontsize=10)
    ax1.legend(loc='upper left'); ax1.grid(True, alpha=0.3)
    st.pyplot(fig2)
    plt.close(fig2)

    # Plot 3: Sigma (Volatility Check)
    st.markdown("**波动率验证 (Sigma Check)**")
    # 仅当 session state 中有数据时才绘制图表
    if st.session_state.get('sigma_rolling_data') and ticker in st.session_state.sigma_rolling_data:
        roll_vol = st.session_state.sigma_rolling_data[ticker]
        sigma_val = st.session_state.sigma_dict[ticker]

        fig4, ax3 = plt.subplots(figsize=(10, 3))
        # 简单绘制即可，核心是确认当前使用的 Sigma 足够稳健
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
    # 这一部分放在后面，作为基于上述参数的推演结果
    # =========================================================
    st.subheader("2. 盈亏分布推演 (Simulation)")
    st.caption(f"👉 **前提假设**：如果估值回归真的遵循上述 Lambda={current_lambda:.2f} 的历史规律，那么正态分布下的结局是：")

    # 定义关键时间窗口
    check_points_map = {
        "1个月 (21交易日)": 21,
        "3个月 (63交易日)": 63,
        "6个月 (126交易日)": 126,
        "9个月 (189交易日)": 189
    }

    # 运行模拟
    # Check if sigma_daily is available and sensible before running simulation
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

# --- Page 2: Optimal Expiry Solver ---
# ---------------------------------------------------------
# 请复制以下代码，替换 code/app_unified_zh.py 中的 page_solver 函数
# ---------------------------------------------------------

def page_solver(P_CURRENT, V_TARGET, V_HARD_FLOOR, V_FILL_PLAN, LAMBDA, SIGMA_ASSET, IV_PRICING, R_RISKFREE, ticker, K_FACTOR, BETA):
    st.title("🎯 Step 0.5: 最优期限求解 (动态 K 值版)")
    st.subheader(f"资产: {ticker} | 目标: 在 {V_FILL_PLAN} 时打满子弹")
    st.markdown("---")

    # [FIX] Move k_fill_target definition before the expander that uses it
    # 新增：目标 K 值输入
    col_k1, col_k2 = st.columns(2)
    with col_k1:
        # 显示当前的起始 K (从左侧边栏继承)
        st.metric("起始 K 值 (Start)", f"{K_FACTOR:.2f}", help="当前左侧边栏设定的 K 值")
    with col_k2:
        # 允许用户设定补仓时的 K
        # MODIFIED: Default value set to 0.5 per user request (Constant K strategy by default)
        k_fill_target = st.number_input("满仓 K 值 (Target at Fill)",
                                     min_value=K_FACTOR, max_value=2.0, value=0.5, step=0.1,
                                     help="当股价跌到 V_fill 时，你愿意使用多大的 K 值？通常设为 0.5 (保持不变) 或 1.0 (激进加仓)。")


    # --- 1. 策略配置区 (新增) ---
    with st.expander("❓ 什么是“动态 K 值”求解？", expanded=True):
        st.markdown(f"""
            **核心思想**：
            通常我们在建仓时比较谨慎（使用较小的 $k$，如 0.5），但随着股价下跌，安全边际变大，我们的信心会增强（使用较大的 $k$，如 1.0）。

            **本工具的目标**：
            寻找一张合约，使得：
            1.  **现在 ($P={P_CURRENT}$)**：应用 **起始 K={K_FACTOR}** 时，仓位适中。
            2.  **到底 ($P={V_FILL_PLAN}$)**：应用 **最终 K={k_fill_target}** 时，建议仓位 **恰好为 100%**。

            这样你就能设计出一个“越跌越买，到底正好满仓”的完美加仓路径。
        """)

    st.markdown("---")

    if 'bs_greek_calculator' not in globals() or not bs_greek_calculator or 'calculate_single_asset_kelly_ratio' not in globals() or not calculate_single_asset_kelly_ratio:
        st.error("依赖模块 (optimal_expiry_solver.py) 未导入，无法进行求解。")
        return

    # 检查输入合理性
    if V_FILL_PLAN >= P_CURRENT:
        st.error(f"错误：补仓价 V_fill ({V_FILL_PLAN}) 必须低于当前价格 ({P_CURRENT})。")
        return

    results = []

    # --- 2. 求解循环 ---
    # 我们遍历期限，寻找那张能在 V_fill 配合 k_fill_target 达到 100% 的合约
    # MODIFIED: Start range from 90 days to avoid volatile short-term structures
    for days in range(90, 1100, 7):
        T = days / 365.0

        # A. 计算【当前】状态 (P_CURRENT, k=K_FACTOR)
        c_price, c_delta, c_theta_annual = bs_greek_calculator(P_CURRENT, V_HARD_FLOOR, T, R_RISKFREE, IV_PRICING)

        kelly_full_now = calculate_single_asset_kelly_ratio(
            P_CURRENT, c_price, c_delta, c_theta_annual, V_TARGET, V_HARD_FLOOR, LAMBDA, SIGMA_ASSET, R_RISKFREE, beta=BETA
        )
        kelly_alloc_now = kelly_full_now * K_FACTOR  # Apply Start K

        # B. 计算【补仓】状态 (V_FILL_PLAN, k=k_fill_target)
        # 假设：忽略时间损耗（考察即时弹性）
        c_fill_price, c_fill_delta, c_fill_theta_fill = bs_greek_calculator(V_FILL_PLAN, V_HARD_FLOOR, T, R_RISKFREE, IV_PRICING)

        kelly_full_at_fill = calculate_single_asset_kelly_ratio(
            V_FILL_PLAN, c_fill_price, c_fill_delta, c_fill_theta_fill,
            V_TARGET, V_HARD_FLOOR, LAMBDA, SIGMA_ASSET, R_RISKFREE, beta=BETA
        )
        kelly_alloc_at_fill = kelly_full_at_fill * k_fill_target # Apply Target K (关键变化点)

        # C. 记录结果
        # 目标：kelly_alloc_at_fill == 1.0
        diff = abs(kelly_alloc_at_fill - 1.0)

        results.append({
            "Days": days,
            "Kelly_Now": kelly_alloc_now,
            "Kelly_At_Fill": kelly_alloc_at_fill,
            "Diff_From_100": diff,
            "Price_Now": c_price
        })

    df = pd.DataFrame(results)

    if df.empty:
        st.warning("无法计算。请检查参数。")
        return

    # --- 3. 寻找最优解 ---
    best_idx = df['Diff_From_100'].idxmin()
    best_row = df.loc[best_idx]

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

    # --- 4. 动态路径推演 (Dynamic K Simulation) ---
    st.markdown("---")
    st.subheader("📉 动态 K 值加仓路径推演")
    st.caption(f"模拟：股价下跌，K 值从 {K_FACTOR} 线性增加至 {k_fill_target}。")

    sim_prices = np.linspace(P_CURRENT, V_FILL_PLAN, 50)
    sim_allocations = []
    sim_ks = []

    T_best = best_row['Days'] / 365.0

    for p in sim_prices:
        # 1. 动态计算当前的 K 值 (线性插值)
        # progress: 0.0 (Top) -> 1.0 (Bottom)
        progress = (P_CURRENT - p) / (P_CURRENT - V_FILL_PLAN)
        k_dynamic = K_FACTOR + (k_fill_target - K_FACTOR) * progress

        # 2. 计算期权和凯利
        c, d, t_val = bs_greek_calculator(p, V_HARD_FLOOR, T_best, R_RISKFREE, IV_PRICING)
        kelly_ratio_raw = calculate_single_asset_kelly_ratio(
            p, c, d, t_val, V_TARGET, V_HARD_FLOOR, LAMBDA, SIGMA_ASSET, R_RISKFREE, beta=BETA
        )

        final_alloc = kelly_ratio_raw * k_dynamic
        sim_allocations.append(final_alloc)
        sim_ks.append(k_dynamic)

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

    plt.title(f"加仓路径: 价格下跌 {P_CURRENT}->{V_FILL_PLAN} | 信心增强 k={K_FACTOR}->{k_fill_target}", fontsize=12)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.info(f"""
        **图表解读**：
        * **蓝色实线**：你应该持有的总仓位。它现在的斜率更陡峭了，因为不仅期权在变便宜，你的 K 值（虚线）也在变大。
        * **灰色虚线**：K 值的变化路径。这代表了你的心态——股价越低，下注越重。
        * **结果**：这张 {int(best_row['Days'])} 天的合约，完美配合了你的心态，在 $V_{{fill}}$ 处精准达到满仓。
    """)


# --- Page 3: Main Calculator (Dashboard) ---
# MODIFIED: Added k_fill to the function signature
def page_dashboard(ticker, lambda_val, sigma_val, r_f, k_factor, beta, P, V_target, V_hard, opt_price, delta, theta, V_fill, iv_pricing, days_to_expiry, k_fill, total_capital):
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

    # MODIFIED: CAPPED at 100% logic
    if P <= V_fill:
        # When price hits or breaches V_fill, use k_fill as the factor, and cap at 1.0
        # Recalculate f_cash using k_fill, then cap at 1.0, ensuring dynamic k is used at fill point
        f_cash_raw = (k_fill * alpha * ERP) / variance_leaps if (ERP > 0 and variance_leaps > 0) else 0.0
        f_cash = min(1.0, max(0.0, f_cash_raw))
    else:
        # When price is above V_fill, use the standard k_factor (Start K) and cap at 1.0
        f_cash = min(1.0, f_cash)


    # --- Calculate Contracts ---
    contract_cost = opt_price * 100
    if contract_cost > 0:
        target_contracts_float = (f_cash * total_capital) / contract_cost
        target_contracts = int(target_contracts_float)
    else:
        target_contracts = 0
        target_contracts_float = 0.0

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

        # --- Alpha Explanation (Updated per user request) ---
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

        # 1. 生成全景价格序列 (从硬底 V_hard 到 目标价 V_target)
        # Extend the range slightly past V_target for visualization
        sim_prices = np.linspace(V_hard, V_target * 1.05, 100)
        allocations = []
        k_values = []
        contracts_series = []

        T_year = days_to_expiry / 365.0

        # --- Iteration ---
        for p_sim in sim_prices:
            # --- A. Dynamic K Factor Calculation ---
            k_dynamic = k_factor
            if p_sim <= P:
                # Downside: Linear interpolation for K from k_factor (Start) to k_fill (Max)
                # Interpolate only between P and V_fill
                progress = (P - p_sim) / max(1e-9, (P - V_fill))
                # Clamp progress: K increases only until V_fill is reached (i.e., progress >= 1.0)
                progress_clamped = min(1.0, max(0.0, progress))

                k_dynamic = k_factor + (k_fill - k_factor) * progress_clamped

                # If price falls below V_fill, K remains k_fill
                if p_sim < V_fill:
                    k_dynamic = k_fill

            else:
                # Upside: Maintain initial K (k_factor), letting Alpha and ERP reduce position
                k_dynamic = k_factor

            k_values.append(k_dynamic)

            # --- B. Full Dynamic Kelly Calculation ---
            # NOTE: We use the full, proper Kelly calculation for each price point p_sim
            # This ensures ERP and Alpha are recalculated based on the new p_sim

            c_sim, d_sim, t_val_sim = bs_greek_calculator(p_sim, V_hard, T_year, r_f, iv_pricing)

            kelly_ratio_raw = calculate_single_asset_kelly_ratio(
                p_sim, c_sim, d_sim, t_val_sim, V_target, V_hard, lambda_val, sigma_val, r_f, beta=beta
            )

            final_alloc = kelly_ratio_raw * k_dynamic

            # MODIFIED: Cap logic in chart
            if p_sim <= V_fill:
                 # Re-calculate allocation using k_fill and cap at 1.0
                k_fill_dynamic = k_fill
                kelly_ratio_raw_at_fill = calculate_single_asset_kelly_ratio(
                    p_sim, c_sim, d_sim, t_val_sim, V_target, V_hard, lambda_val, sigma_val, r_f, beta=beta
                )
                final_alloc = kelly_ratio_raw_at_fill * k_fill_dynamic
                final_alloc = min(1.0, final_alloc)
            else:
                final_alloc = min(1.0, final_alloc)


            # Ensure allocation is non-negative
            final_alloc = max(0.0, final_alloc)
            allocations.append(final_alloc)

            # Calculate contracts at this price point
            # Note: Option price c_sim changes, so contract cost changes
            cost_sim = c_sim * 100
            if cost_sim > 0:
                num_c = (final_alloc * total_capital) / cost_sim
            else:
                num_c = 0
            contracts_series.append(num_c)

        # --- C. Plotting (Dual Axis) ---
        fig, ax1 = plt.subplots(figsize=(10, 5))

        # 绘制区域填充
        ax1.axvspan(V_hard, P, color='#d4edda', alpha=0.5, label='加仓区')
        ax1.axvspan(P, V_target * 1.05, color='#f8d7da', alpha=0.5, label='减仓区') # Up to 105% of V_target

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
        # Find the allocation near V_fill
        v_fill_alloc_index = np.argmin(np.abs(sim_prices - V_fill))
        ax1.scatter([V_fill], allocations[v_fill_alloc_index], color='red', s=100, zorder=5, label=f'补仓点 V_fill (${V_fill:.2f})')
        # Find the allocation near V_hard
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

        # --- NEW: Grid Trading Advice ---
        # Determine step size based on magnitude
        max_c = max(contracts_series) if contracts_series else 0
        if max_c > 50:
             step_size = max(1, int(max_c / 20)) # e.g. if 100 contracts, step every 5
        else:
             step_size = 1

        st.info(f"💡 **网格操作提示** (检测到最大持仓约 {int(max_c)} 张，已自动将提示步长设为 **{step_size}** 张):")

        # Simple scan to find price points where contract count crosses integer multiples of step_size
        current_c = target_contracts

        # Look Down (Buying)
        buy_points = []
        # [FIX 1] 修复循环范围，确保能扫描到 V_hard (数组最底端，index=0)
        # Iterate backwards through prices (from highest to lowest)
        for i in range(len(sim_prices)):
            idx = len(sim_prices) - 1 - i # Start from the lowest price V_hard
            p_val = sim_prices[idx]
            c_val = contracts_series[idx]

            if p_val > P: continue # Only look at prices below current

            # We want to find p where contracts >= current + step, current + 2*step...
            # Look for the contract count where it crosses the next threshold
            next_threshold = current_c + (len(buy_points) + 1) * step_size
            if c_val >= next_threshold:
                 buy_points.append((p_val, c_val)) # 这里存 c_val 没问题
                 if len(buy_points) >= 3: break # Show top 3

        # Look Up (Selling)
        sell_points = []
        # Iterate forwards through prices (from lowest to highest)
        for i in range(len(sim_prices)):
            p_val = sim_prices[i]
            c_val = contracts_series[i]

            if p_val < P: continue

            # We want to find p where contracts <= current - step
            next_threshold = current_c - (len(sell_points) + 1) * step_size
            if c_val <= next_threshold and next_threshold >= 0:
                sell_points.append((p_val, c_val))
                if len(sell_points) >= 3: break

        col_buy, col_sell = st.columns(2)
        with col_buy:
            st.markdown("##### 📉 下跌加仓参考")
            if not buy_points:
                st.write("无近期加仓点 (或已接近满仓)")
            else:
                for p_val, c_val in buy_points:
                    # 重新计算目标持仓数量用于显示
                    idx_buy = buy_points.index((p_val, c_val))
                    target_hold = current_c + (idx_buy + 1) * step_size

                    st.write(f"- 跌至 **${p_val:.2f}** : 加至 **{int(target_hold)}** 张 (+{step_size}张)")

        with col_sell:
            st.markdown("##### 📈 上涨减仓参考")
            if not sell_points:
                st.write("无近期减仓点 (或已空仓)")
            else:
                 for p_val, c_val in sell_points:
                     # [FIX 2] 计算目标台阶，而不是显示计算出的浮动值
                     # 也就是：当前持仓 - (这是第几次卖出 * 步长)
                     idx_sell = sell_points.index((p_val, c_val))
                     target_hold = current_c - (idx_sell + 1) * step_size

                     st.write(f"- 涨至 **${p_val:.2f}** : 减至 **{int(target_hold)}** 张 (-{step_size}张)")


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
        NOMINAL_ACCOUNT_VALUE = total_capital

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
                "预估亏损": f"${account_loss_usd:,.0f}" if f_cash > 0 else "$0",
            })

        risk_df = pd.DataFrame(risk_table)
        st.table(risk_df)
        st.caption("*注：此处使用有效杠杆 (L) 进行线性估算，实际期权在暴跌中的跌幅可能因 Gamma/Vega 效应有所不同。仅供风控参考。如果 $3\\sigma$ 亏损额让你感到恐慌，请在侧边栏调低 $k$ 值。")


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
    'days_to_expiry': 365, # Default 1 year
    'k_fill': 1.0, # NEW Default Max K for Step 1
    'total_capital': 100000.0 # NEW Default Capital
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

    # --- 1. 输入框 ---
    # 使用 key='ticker_global' 绑定状态
    ticker = st.text_input("股票代码 (Ticker)", value=st.session_state.ticker, key='ticker_global').upper()

    # --- 2. 自动获取数据逻辑 (Auto-Fetch) ---
    # 定义判断条件：
    # A. 刚打开 App，还没有 fetch 过 (last_fetched_ticker 不存在)
    # B. 用户修改了输入框内容 (ticker != last_fetched_ticker)
    # C. 数据丢失 (sigma_dict 不在 session 中)
    need_refresh = (ticker != st.session_state.get('last_fetched_ticker')) or \
                   ('sigma_dict' not in st.session_state)

    if need_refresh:
        # 检查依赖是否存在
        if 'get_ou_for_ticker' in globals() and 'get_sigma' in globals():
            try:
                # 使用 spinner 提示用户正在后台计算
                with st.spinner(f"正在自动计算 {ticker} 的历史波动率与回归参数..."):

                    # === 以下是原按钮内的获取逻辑 (原封不动搬运) ===
                    ou_window = st.session_state.get('window_days', 90)
                    ou = get_ou_for_ticker(ticker, window=ou_window)
                    # Handle None from OU calculation
                    new_lambda = ou["lambda"] * 252.0 if ou and ou["lambda"] is not None else st.session_state.get('lambda', 6.0393)

                    sigma_dict, _, _, rolling_series_dict = get_sigma(
                        [ticker], period="5y", window=252, percentile=0.85, annualize=True, safety_lock=True
                    )
                    new_sigma = sigma_dict.get(ticker)

                    # === 更新 Session State ===
                    st.session_state['lambda'] = new_lambda
                    st.session_state['sigma'] = new_sigma
                    st.session_state['ticker'] = ticker

                    # 关键：保存图表所需的详细数据
                    st.session_state['sigma_rolling_data'] = rolling_series_dict
                    st.session_state['sigma_dict'] = sigma_dict

                    # === 标记：记录当前已获取的 Ticker，防止重复刷新 ===
                    st.session_state['last_fetched_ticker'] = ticker

            except Exception as e:
                st.error(f"❌ 数据获取失败: {e}")
        else:
            st.error("依赖模块 (lambda_tools.py / sigma_tools.py) 未导入，无法获取历史数据。")

    # --- 3. 如果数据已就绪，显示简报 ---
    if st.session_state.get('last_fetched_ticker') == ticker and 'lambda' in st.session_state:
         # 保持用户要求的显示精度
         st.caption(f"✅ 已加载: λ={st.session_state['lambda']:.2f}, σ={st.session_state['sigma']:.1%}")

    st.divider()

    lambda_val = st.number_input("年化 Lambda (λ)", value=st.session_state['lambda'], key='lambda_global', format="%.4f",
                                 help="【均值回归动力】数值越大，修复越快。若图表显示 Lambda 处于历史极高位(>80分位)，建议手动调低以提高安全边际。")
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
    current_k_fill = st.session_state.get('k_fill', 1.0) # Retrieve new k_fill value
    current_total_capital = st.session_state.get('total_capital', 100000.0)


    if page == "Step 0: 市场诊断":
        st.subheader("诊断特有参数")
        window_days = st.slider("滚动窗口 (交易日)", min_value=30, max_value=252, value=st.session_state.window_days, key='window_days_diag')
        st.session_state['window_days'] = window_days
        current_window_days = window_days
    else:
        if page == "Step 1: 主仓位计算器":
            st.subheader("2.1 策略约束")
            # NEW INPUT: Total Capital
            current_total_capital = st.number_input("账户本金 ($)", value=st.session_state.total_capital, step=10000.0, key='capital_dash')

            current_r_f = st.number_input("无风险利率 (r_f)", value=st.session_state.r_f, key='r_f_dash', format="%.3f")

            # MODIFIED: Changed label to Start K
            current_k_factor = st.slider("起始 K (Start)", 0.1, 1.0, st.session_state.k_factor, 0.05, key='k_dash',
                                         help="【激进程度】0.5 = 推荐标准 (半凯利)，最大化长期几何增长率。1.0 = 满凯利，仅建议在极度低估时用于回补。")

            # NEW INPUT: Max K at Fill
            current_k_fill = st.number_input("满仓 K (Max at Fill)",
                                     min_value=current_k_factor, max_value=2.0, value=st.session_state.k_fill, step=0.1,
                                     key='k_fill_dash',
                                     help="当股价跌至 V_fill 时，信心增强，K 值线性增加至此值。")

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
            st.session_state.k_fill = current_k_fill # Store k_fill
            st.session_state.total_capital = current_total_capital # Store total_capital
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
        # Pass NEW arguments to page_dashboard, including k_fill
        page_dashboard(ticker, current_lambda, current_sigma, current_r_f, current_k_factor, current_beta, current_P, current_V_target, current_V_hard, current_opt_price, current_delta, current_theta, current_V_fill, current_iv_pricing, current_days_to_expiry, current_k_fill, current_total_capital)

elif page == "Step 2: 多标的组合管理":
    max_leverage_cap = st.session_state.get('c_max_slider', 1.0)
    page_multi_asset_normalization(max_leverage_cap)