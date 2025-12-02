import streamlit as st
import numpy as np
import pandas as pd
import os
import sys

# ==========================================
# 1. SETUP: Path & Imports
# ==========================================
# 路径设置，确保可以导入 utils 模块
current_dir = os.path.dirname(os.path.abspath(__file__))
# 向上走一级到项目根目录，以便导入 utils
sys.path.append(os.path.join(current_dir, ".."))

from utils.lambda_tools import get_ou_for_ticker
from utils.sigma_tools import get_sigma

# ==========================================
# 2. Streamlit Page Configuration
# ==========================================
st.set_page_config(page_title="莫顿-凯利 LEAPS 优化器", layout="wide", page_icon="🌌")

st.title("🌌 莫顿-凯利 LEAPS 仓位优化器")
st.markdown("---")

# ==========================================
# 3. SIDEBAR: Global Settings
# ==========================================
with st.sidebar:
    st.header("1. 资产与统计数据")

    # 股票代码输入
    ticker = st.text_input("股票代码 (Ticker)", value="NVDA").upper()

    if st.button("获取历史统计数据"):
        try:
            # 路径修复：确保 lambda_tools 和 sigma_tools 能够找到 PE CSV
            project_root = os.path.abspath(os.path.join(current_dir, ".."))
            sys.path.append(project_root)

            with st.spinner("正在计算 OU 参数与波动率..."):
                # 1. 获取 Lambda (回归动力)
                ou = get_ou_for_ticker(ticker, window=90)
                st.session_state['lambda'] = ou["lambda"] * 252.0

                # 2. 获取 Sigma (稳健历史波动率)
                sigma_dict, _, _, _ = get_sigma(
                    [ticker], period="5y", window=252, percentile=0.85, annualize=True, safety_lock=True
                )
                st.session_state['sigma'] = sigma_dict[ticker]
                st.success("数据获取成功！")
        except Exception as e:
            st.error(f"错误: {e}")

    # 使用 Session State 或默认值
    lambda_val = st.number_input("年化 Lambda (回归速度 λ)",
                                 value=st.session_state.get('lambda', 5.8930),
                                 format="%.4f", help="从 Step 0 市场诊断获取")

    sigma_val = st.number_input("年化 Sigma (稳健波动率 σ)",
                                value=st.session_state.get('sigma', 0.6082),
                                format="%.4f", help="通常取历史 85% 分位数")

    st.divider()

    st.header("2. 策略约束与风险控制")

    # 风险控制参数
    r_f = st.number_input("无风险利率 (r_f)", value=0.041, format="%.3f")

    k_factor = st.slider("凯利分数 (k)", 0.1, 1.0, 0.50, 0.05, help="0.5 为半凯利，最安全推荐值")

    beta = st.slider("估值折扣系数 (beta)", 0.0, 1.0, 0.20, 0.05, help="股价接近目标价时，Alpha 降低的程度")

# ==========================================
# 4. MAIN AREA: Inputs (中文标签)
# ==========================================
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("市场与估值输入")
    # 估值输入
    P = st.number_input("当前股价 P ($)", value=182.00, format="%.2f")
    V_target = st.number_input("目标价 V ($)", value=225.00, format="%.2f", help="你的公允价值估值")
    V_hard = st.number_input("硬底 V_hard ($)", value=130.00, format="%.2f", help="极限支撑位，通常与期权行权价接近")

with col2:
    st.subheader("期权合约数据")
    # 期权链数据输入 (需手动从券商软件获取或 Step 0.5 求解器回填)
    opt_price = st.number_input("LEAPS 合约价格 ($)", value=64.63, format="%.2f")
    delta = st.number_input("Delta 值", value=0.8460, format="%.4f")
    theta = st.number_input("每日 Theta 损耗 (绝对值 $)", value=0.0432, format="%.4f", help="期权每日时间损耗的绝对值")

# ==========================================
# 5. CALCULATION ENGINE
# ==========================================
if opt_price > 0:
    # --- A. 杠杆与成本 ---
    L = delta * (P / opt_price)
    # 年化 Theta 衰减率 (252个交易日)
    theta_annual = (theta / opt_price) * 252.0

    # --- B. 预期收益 ---
    # 正股预期年化回报率 (根据 OU 均值回归)
    mu_stock = lambda_val * np.log(V_target / P)
    # LEAPS 杠杆后年化回报率
    mu_leaps = mu_stock * L
    # LEAPS 净优势 (ERP) = 回报 - 无风险利率成本 - 时间损耗成本
    ERP = mu_leaps - r_f - theta_annual

    # --- C. 风险 ---
    sigma_leaps = sigma_val * L
    variance_leaps = sigma_leaps ** 2

    # --- D. Alpha (信心折扣系数) ---
    range_len = max(1e-9, V_target - V_hard)
    dist_from_floor = P - V_hard
    risk_ratio = max(0.0, min(1.0, dist_from_floor / range_len))
    alpha = 1.0 - (beta * risk_ratio)

    # --- E. 凯利公式 ---
    if ERP > 0:
        # f_cash = k * Alpha * ERP / Variance
        f_cash = (k_factor * alpha * ERP) / variance_leaps
    else:
        f_cash = 0.0

    f_cash = max(0.0, f_cash)

    # ==========================================
    # 6. DISPLAY RESULTS (中文输出)
    # ==========================================
    with col3:
        st.subheader("📊 实时计算结果")

        st.caption("凯利建议仓位占比")
        if ERP > 0:
            st.metric(
                label="仓位",
                value=f"{f_cash:.2%}",
                label_visibility="collapsed",
                delta=f"有效杠杆: {L:.2f}x"
            )
        else:
            st.error("净优势为负 (ERP < 0)，不建议开仓。")

        st.divider()
        st.write(f"**净优势 (ERP):** {ERP:.2%}")
        st.write(f"**信心系数 (Alpha):** {alpha:.3f}")
        st.write(f"**LEAPS 年化波动率:** {sigma_leaps:.2%}")

    st.divider()

    # ==========================================
    # 7. VISUALIZATION: Sensitivity (敏感性分析)
    # ==========================================
    st.subheader("情景分析：若股价下跌，建议仓位如何变化？")
    st.caption("假设在下跌时，你会换仓购买新的深度实值合约，以维持当前杠杆率，从而隔离期权噪音，仅显示估值吸引力。")

    # 生成情景 (从硬底到当前价格)
    prices = np.linspace(V_hard, P, 50)
    allocations = []

    for p_sim in prices:
        # 1. 重新计算 Alpha (越接近硬底，信心越高)
        dist = p_sim - V_hard
        rr = max(0.0, min(1.0, dist / range_len))
        a_sim = 1.0 - (beta * rr)

        # 2. 假设恒定杠杆 (L_sim = L)
        L_sim = L

        # 3. 重新计算预期回报 (Mu_stock 越高)
        mu_s = lambda_val * np.log(V_target / p_sim)
        mu_l = mu_s * L_sim

        # 4. 风险与成本 (假设不变)
        theta_annual_sim = theta_annual
        sigma_l_sim = sigma_val * L_sim
        var_l_sim = sigma_l_sim ** 2

        # 5. 凯利计算
        erp_sim = mu_l - r_f - theta_annual_sim

        if erp_sim > 0:
            val = (k_factor * a_sim * erp_sim) / var_l_sim
        else:
            val = 0
        allocations.append(max(0, val))

    chart_data = pd.DataFrame({
        "股价": prices,
        "建议仓位比例": allocations
    })

    # 使用中文标签绘图
    st.line_chart(chart_data, x="股价", y="建议仓位比例", color="#FF4B4B")
    st.caption(f"当前杠杆率 L = {L:.2f}x (固定)。曲线显示的是纯粹的均值回归吸引力。")