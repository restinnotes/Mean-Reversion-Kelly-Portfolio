import streamlit as st
import numpy as np
import pandas as pd
import os
import sys

# ==========================================
# 1. SETUP: Path & Imports
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, ".."))

from utils.lambda_tools import get_ou_for_ticker
from utils.sigma_tools import get_sigma

st.set_page_config(page_title="Merton-Kelly Optimizer", layout="wide", page_icon="🌌")

st.title("🌌 Merton-Kelly LEAPS Optimizer")

# ==========================================
# 2. SIDEBAR: Global Settings
# ==========================================
with st.sidebar:
    st.header("1. Asset & Statistics")

    # Default Ticker from screenshot
    ticker = st.text_input("Ticker Symbol", value="NVDA").upper()

    if st.button("Fetch Historical Stats"):
        try:
            with st.spinner("Calculating OU Params & Volatility..."):
                # 1. Get Lambda
                ou = get_ou_for_ticker(ticker, window=90)
                st.session_state['lambda'] = ou["lambda"] * 252.0

                # 2. Get Sigma
                sigma_dict, _, _, _ = get_sigma(
                    [ticker], period="5y", window=252, percentile=0.85, annualize=True, safety_lock=True
                )
                st.session_state['sigma'] = sigma_dict[ticker]
                st.success("Data Fetched!")
        except Exception as e:
            st.error(f"Error: {e}")

    # Use session state if available, otherwise use Screenshot Defaults
    # Default Lambda: 5.8930
    lambda_val = st.number_input("Annualized Lambda (Reg)",
                                 value=st.session_state.get('lambda', 5.8930),
                                 format="%.4f")

    # Default Sigma: 0.6082
    sigma_val = st.number_input("Annualized Sigma (Hist)",
                                value=st.session_state.get('sigma', 0.6082),
                                format="%.4f")

    st.divider()

    st.header("2. Strategy Constraints")

    # Default Risk Free: 0.041
    r_f = st.number_input("Risk Free Rate", value=0.041, format="%.3f")

    # Default Kelly: 0.50
    k_factor = st.slider("Kelly Fraction (k)", 0.1, 1.0, 0.50, 0.05)

    # Default Beta: 0.20
    beta = st.slider("Valuation Discount (beta)", 0.0, 1.0, 0.20, 0.05)

# ==========================================
# 3. MAIN AREA: Inputs
# ==========================================
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Market Price")
    # Defaults: P=182.00, V=225.00, Floor=130.00
    P = st.number_input("Current Stock Price ($)", value=182.00, format="%.2f")
    V_target = st.number_input("Target Price V ($)", value=225.00, format="%.2f")
    V_hard = st.number_input("Hard Floor V_hard ($)", value=130.00, format="%.2f")

with col2:
    st.subheader("Option Chain")
    # Defaults: Price=64.63, Delta=0.8460, Theta=0.0432
    opt_price = st.number_input("LEAPS Ask Price ($)", value=64.63, format="%.2f")
    delta = st.number_input("Delta", value=0.8460, format="%.4f")
    theta = st.number_input("Daily Theta (Abs)", value=0.0432, format="%.4f")

# ==========================================
# 4. CALCULATION ENGINE
# ==========================================
if opt_price > 0:
    # --- A. Leverage & Cost ---
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
    if ERP > 0:
        f_cash = (k_factor * alpha * ERP) / variance_leaps
    else:
        f_cash = 0.0

    f_cash = max(0.0, f_cash)

    # ==========================================
    # 5. DISPLAY RESULTS
    # ==========================================
    with col3:
        st.subheader("📊 Live Output")

        st.caption("Kelly Allocation %")
        if ERP > 0:
            st.metric(
                label="size",
                value=f"{f_cash:.2%}",
                label_visibility="collapsed",
                delta=f"Lev: {L:.2f}x"
            )
        else:
            st.error("Negative Edge (ERP < 0)")

        st.divider()
        st.write(f"**Net Edge (ERP):** {ERP:.2%}")
        st.write(f"**Confidence (Alpha):** {alpha:.3f}")
        st.write(f"**LEAPS Volatility:** {sigma_leaps:.2%}")

    st.divider()

    # ==========================================
    # 6. VISUALIZATION: Sensitivity (Fixed Logic)
    # ==========================================
    st.subheader("Scenario Analysis: What if price drops?")

    # Generate scenarios (From Floor to Current Price)
    prices = np.linspace(V_hard, P, 50)
    allocations = []

    for p_sim in prices:
        # 1. Recalculate Alpha (Confidence increases as we drop)
        dist = p_sim - V_hard
        rr = max(0.0, min(1.0, dist / range_len))
        a_sim = 1.0 - (beta * rr)

        # 2. Assume "Constant Leverage" Strategy
        # Logic: If price drops to $130, we wouldn't buy this 5x leverage ATM option.
        # We would buy a NEW deep ITM option that restores our target leverage (e.g. ~2.4x).
        # This isolates the "Valuation Alpha" from "Option Greeks Noise".
        L_sim = L

        # 3. Recalc Drift (Higher Potential Return)
        mu_s = lambda_val * np.log(V_target / p_sim)
        mu_l = mu_s * L_sim

        # 4. Theta & Risk (Constant assumption for strategic view)
        theta_annual_sim = theta_annual
        sigma_l_sim = sigma_val * L_sim
        var_l_sim = sigma_l_sim ** 2

        # 5. Kelly Calc
        erp_sim = mu_l - r_f - theta_annual_sim

        if erp_sim > 0:
            val = (k_factor * a_sim * erp_sim) / var_l_sim
        else:
            val = 0
        allocations.append(max(0, val))

    chart_data = pd.DataFrame({
        "Stock Price": prices,
        "Suggested Allocation": allocations
    })

    # Use a custom key to force refresh
    st.line_chart(chart_data, x="Stock Price", y="Suggested Allocation", color="#FF4B4B")
    st.caption("Assumption: Maintaining constant effective leverage (fresh entry) at each price point.")