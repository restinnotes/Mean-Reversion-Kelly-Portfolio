import streamlit as st
import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ==========================================
# 1. SETUP: Path & Imports
# ==========================================
# 使用 os.path.dirname(sys.executable) 来获取打包后的运行目录，
# 否则在未打包环境下使用 os.path.abspath(__file__)
def get_resource_root():
    """Determines the correct root path for resources (e.g., pe_csv) in both development and PyInstaller environments."""
    if getattr(sys, 'frozen', False):
        # Running as a PyInstaller bundle
        return os.path.dirname(sys.executable)
    else:
        # Running in development environment
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up one level from 'code' directory to get the project root
        return os.path.abspath(os.path.join(current_dir, ".."))

project_root = get_resource_root()

# Add necessary paths for imports
# In the executable, these relative paths will be handled by PyInstaller hooks,
# but we keep them for development convenience.
sys.path.append(os.path.join(project_root, "code", "utils"))
sys.path.append(os.path.join(project_root, "code", "strategies")) # Use project_root to resolve path consistently

# Import all core functions
try:
    from utils.lambda_tools import get_ou_for_ticker, calculate_ou_params
    from utils.sigma_tools import get_sigma
    from optimal_expiry_solver import bs_greek_calculator, calculate_single_asset_kelly_ratio
except ImportError as e:
    st.error(f"Module import error. Please check utils/ and strategies/ directories: {e}")
    st.stop()


# ==========================================
# 2. Matplotlib Font Configuration (English for consistency)
# ==========================================
# We use English labels in the plots to avoid common Matplotlib Chinese font issues.
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Verdana']
plt.rcParams['axes.unicode_minus'] = True


# ==========================================
# 3. PAGE LOGIC FUNCTIONS (封装各应用逻辑)
# ==========================================

# --- Page 1: 市场诊断 (Rolling Analysis) ---
def page_diagnosis(ticker, window_days):
    st.title("📈 Step 0: 市场诊断 - 滚动分析")
    st.subheader(f"资产: {ticker} | 滚动窗口: {window_days} 交易日")
    st.markdown("---")

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
    df['rolling_mean'] = df['value'].rolling(window=window_days).mean()
    dates = []; pe_values = []; pe_means = []; lambdas_annual = []; half_lives = []; sigmas_daily = []

    if len(df) >= window_days:
        window_data = df.iloc[-window_days:]
        series = window_data.set_index('date')['value']
        try:
            ou = calculate_ou_params(series)
            if ou:
                dates = df['date'].iloc[window_days:].tolist()
                pe_values = df['value'].iloc[window_days:].tolist()
                pe_means = df['rolling_mean'].iloc[window_days:].tolist()
                lambdas_annual.extend([ou['lambda'] * 252] * (len(dates) or 1))
                half_lives.extend([ou['half_life']] * (len(dates) or 1))
                sigmas_daily.extend([ou['sigma']] * (len(dates) or 1))
        except Exception:
            st.warning("滚动计算发生异常，请检查数据质量。")

    if not lambdas_annual:
        st.warning("数据不足，无法进行滚动指标计算。")
        return

    current_lambda = lambdas_annual[-1]
    current_hl = half_lives[-1]
    current_pe = pe_values[-1]
    current_mean = pe_means[-1]

    # --- 2. 诊断报告 (简化) ---
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown("**PE Valuation Status**")
        st.code(f"Current PE: {current_pe:.2f}")
        st.code(f"{window_days}d Moving Avg: {current_mean:.2f}")
    with col_d2:
        st.markdown("**Reversion & Volatility**")
        st.code(f"Annualized Lambda (λ): {current_lambda:.4f}")
        st.code(f"Half-Life: {current_hl:.2f} Days")

    st.markdown("---")

    # --- 3. Plotting (使用英文标签) ---
    if len(plot_df := pd.DataFrame({'Date': dates, 'PE_Ratio': pe_values, 'MA': pe_means}).set_index('Date')) > 0:
        st.markdown("##### Visual Diagnosis (Simplified Plot)")
        fig1, ax0 = plt.subplots(figsize=(10, 3))
        ax0.plot(plot_df.index, plot_df['PE_Ratio'], 'k', alpha=0.8, label='PE Ratio')
        ax0.plot(plot_df.index, plot_df['MA'], 'b--', label=f'{window_days}d Moving Avg')
        ax0.set_title(f'{ticker} PE Ratio vs MA', fontsize=10)
        ax0.legend(loc='upper left'); ax0.grid(True, alpha=0.3)
        st.pyplot(fig1)
        plt.close(fig1)

# --- Page 2: 最优期限求解 (Optimal Expiry Solver) ---
def page_solver(P_CURRENT, V_TARGET, V_HARD_FLOOR, V_FILL_PLAN, LAMBDA, SIGMA_ASSET, IV_PRICING, R_RISKFREE, ticker):
    st.title("🎯 Step 0.5: 最优期限求解器")
    st.subheader(f"资产: {ticker} | P={P_CURRENT}")
    st.markdown("---")

    # The actual find_perfect_expiry function from optimal_expiry_solver.py is complex.
    # We rely on the core functions being correctly imported and adapted here.

    results = []
    for days in range(30, 1100, 7):
        T = days / 365.0
        # Use bs_greek_calculator
        c_price, c_delta, c_theta = bs_greek_calculator(P_CURRENT, V_HARD_FLOOR, T, R_RISKFREE, IV_PRICING)

        # Calculate Kelly (k=0.5, beta=0.2 assumed for solver logic)
        kelly_full = calculate_single_asset_kelly_ratio(
            P_CURRENT, c_price, c_delta, c_theta, V_TARGET, V_HARD_FLOOR, LAMBDA, SIGMA_ASSET, R_RISKFREE, beta=0.2
        )
        kelly_target = kelly_full * 0.5

        # Calculate Pilot Cap
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

    # Find intersection point (Diff closest to 0)
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

    # --- Plotting (使用英文标签) ---
    st.markdown("---")
    st.markdown("##### Offense-Defense Balance Chart")

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(df['Days'], df['Kelly_Half'], label='Offense: 0.5 * Kelly Ratio',
             color='blue', linewidth=2, linestyle='--')

    ax.plot(df['Days'], df['Pilot_Cap'], label='Defense: Pilot Cash Cap (1:1 Refill)',
             color='red', linewidth=2)

    ax.scatter(best_row['Days'], best_row['Pilot_Cap'], color='green', s=150, zorder=5, label='Optimal Expiry')

    ax.annotate(
        f"Sweet Spot\n{int(best_row['Days'])} Days\n{best_row['Pilot_Cap']:.1%} Alloc",
        xy=(best_row['Days'], best_row['Pilot_Cap']),
        xytext=(best_row['Days']+100, best_row['Pilot_Cap']+0.1),
        arrowprops=dict(facecolor='black', shrink=0.05),
        fontsize=10, fontweight='bold'
    )

    ax.set_title(f"Optimal Expiry Solver: {ticker}", fontsize=14)
    ax.set_xlabel("Days to Expiration", fontsize=12)
    ax.set_ylabel("Position Allocation %", fontsize=12)

    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# --- Page 3: 主仓位计算器 (App Dashboard) ---
def page_dashboard(ticker, lambda_val, sigma_val, r_f, k_factor, beta, P, V_target, V_hard, opt_price, delta, theta):
    st.title("🌌 Step 1: 凯利 LEAPS 仓位主计算器")
    st.markdown("---")

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
                    delta=f"Leverage: {L:.2f}x"
                )
            else:
                st.error("Negative Edge (ERP < 0).")

            st.divider()
            st.write(f"**Net Edge (ERP):** {ERP:.2%}")
            st.write(f"**Confidence (Alpha):** {alpha:.3f}")
            st.write(f"**LEAPS Volatility:** {sigma_leaps:.2%}")

        with col_m:
            st.subheader("Scenario Analysis (Fixed Leverage)")
            st.caption("How allocation changes as price drops towards the hard floor.")

            # Generate scenarios (From Floor to Current Price)
            prices = np.linspace(V_hard, P, 50)
            allocations = []

            for p_sim in prices:
                # Recalculate Alpha
                dist = p_sim - V_hard
                rr = max(0.0, min(1.0, dist / range_len))
                a_sim = 1.0 - (beta * rr)
                # Recalc Drift
                mu_s = lambda_val * np.log(V_target / p_sim)
                mu_l = mu_s * L
                # Kelly Calc
                erp_sim = mu_l - r_f - theta_annual
                if erp_sim > 0:
                    val = (k_factor * a_sim * erp_sim) / variance_leaps
                else:
                    val = 0
                allocations.append(max(0, val))

            chart_data = pd.DataFrame({
                "Stock Price": prices,
                "Suggested Allocation": allocations
            })
            st.line_chart(chart_data, x="Stock Price", y="Suggested Allocation", color="#FF4B4B")
    else:
        st.warning("请输入有效的期权价格进行计算。")


# ==========================================
# 4. MAIN ROUTER (统一入口)
# ==========================================
st.set_page_config(page_title="统一凯利量化工具", layout="wide", page_icon="📈")


# --- 侧边栏统一输入 (Global Inputs) ---
with st.sidebar:
    st.title("导航与全局参数")
    page = st.radio("选择工具页面",
                    ("Step 0: 市场诊断",
                     "Step 0.5: 最优期限求解",
                     "Step 1: 主仓位计算器"),
                    index=2) # 默认选中主计算器

    st.header("1. 资产与统计数据")
    ticker = st.text_input("股票代码 (Ticker)", value=st.session_state.get('ticker', "NVDA")).upper()

    # 历史数据获取/展示
    if st.button("获取历史统计数据"):
        try:
            with st.spinner("Calculating OU Params & Volatility..."):
                # Use os.path.join(project_root, "pe_csv", f"{ticker}_pe.csv") inside get_ou_for_ticker
                ou = get_ou_for_ticker(ticker, window=90)
                st.session_state['lambda'] = ou["lambda"] * 252.0
                sigma_dict, _, _, _ = get_sigma(
                    [ticker], period="5y", window=252, percentile=0.85, annualize=True, safety_lock=True
                )
                st.session_state['sigma'] = sigma_dict.get(ticker)
                st.session_state['ticker'] = ticker
                st.success("Data Fetched!")
        except Exception as e:
            st.error(f"Error fetching data: {e}")

    # 统计参数
    st.divider()
    lambda_val = st.number_input("年化 Lambda (λ)", value=st.session_state.get('lambda', 5.8930), format="%.4f")
    sigma_val = st.number_input("年化 Sigma (σ)", value=st.session_state.get('sigma', 0.6082), format="%.4f")

    st.header("2. 策略约束")
    r_f = st.number_input("无风险利率 (r_f)", value=0.041, format="%.3f")
    k_factor = st.slider("凯利分数 (k)", 0.1, 1.0, 0.50, 0.05)
    beta = st.slider("估值折扣系数 (beta)", 0.0, 1.0, 0.20, 0.05)

    # 市场估值与期权参数 (作为全局输入，避免重复输入)
    st.header("3. 市场与合约参数")
    P = st.number_input("当前股价 P ($)", value=st.session_state.get('P', 182.00), key='P', format="%.2f")
    V_target = st.number_input("目标价 V ($)", value=st.session_state.get('V_target', 225.00), key='V_target', format="%.2f")
    V_hard = st.number_input("硬底 V_hard ($)", value=st.session_state.get('V_hard', 130.00), key='V_hard', format="%.2f")
    V_fill = st.number_input("计划补仓价 V_fill ($)", value=st.session_state.get('V_fill', 145.00), key='V_fill', format="%.2f") # Solver专用

    st.divider()
    opt_price = st.number_input("LEAPS Price ($)", value=st.session_state.get('opt_price', 64.63), key='opt_price', format="%.2f")
    delta = st.number_input("Delta", value=st.session_state.get('delta', 0.8460), key='delta', format="%.4f")
    theta = st.number_input("Daily Theta (Abs)", value=st.session_state.get('theta', 0.0432), key='theta', format="%.4f")



# --- 页面路由 ---
if page == "Step 0: 市场诊断":
    # 诊断页不需要期权数据
    page_diagnosis(ticker, 90)

elif page == "Step 0.5: 最优期限求解":
    if V_target <= V_hard:
        st.error("错误: 目标价必须高于硬底。")
    elif lambda_val is None or sigma_val is None:
         st.error("请先在侧边栏获取 Lambda/Sigma 统计数据。")
    else:
        # Note: IV_PRICING is usually close to Sigma_ASSET for long-term options
        page_solver(P, V_target, V_hard, V_fill, lambda_val, sigma_val, sigma_val, r_f, ticker)

elif page == "Step 1: 主仓位计算器":
    if lambda_val is None or sigma_val is None:
        st.error("请先在侧边栏获取 Lambda/Sigma 统计数据。")
    elif opt_price <= 0 or delta <= 0:
        st.warning("请在侧边栏输入有效的期权合约数据。")
    else:
        page_dashboard(ticker, lambda_val, sigma_val, r_f, k_factor, beta, P, V_target, V_hard, opt_price, delta, theta)