import streamlit as st
import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt

# ===============================
# 1. SETUP: Path & Imports
# ===============================
current_dir = os.path.dirname(os.path.abspath(__file__))
# Add 'code/strategies' directory to Python search path
strategies_dir = os.path.join(current_dir, "strategies")
if strategies_dir not in sys.path:
    sys.path.append(strategies_dir)

# Import the core solver functions
try:
    from optimal_expiry_solver import find_perfect_expiry
except ImportError as e:
    st.error(f"Module import failed: {e}. Ensure optimal_expiry_solver.py is located in the code/strategies/ directory.")
    st.stop()


# ===============================
# 2. Streamlit App Layout
# ===============================
st.set_page_config(page_title="Selection Assistant - Optimal Expiry Solver", layout="wide", page_icon="🎯")
st.title("🎯 Step 0.5: Selection Assistant - Optimal Expiry Solver")

st.sidebar.header("Input Parameters")

# --- Default Values (NVDA example from optimal_expiry_solver.py) ---
DEFAULT_PARAMS = {
    "TICKER": "NVDA",
    "P_CURRENT": 182.14,
    "V_TARGET": 225.00,
    "V_HARD_FLOOR": 130.00,
    "V_FILL_PLAN": 145.00,
    "LAMBDA": 4.46,
    "SIGMA_ASSET": 0.51,
    "IV_PRICING": 0.45,
    "R_RISKFREE": 0.041,
}

# --- Sidebar Inputs: Asset ---
st.sidebar.subheader("Asset & Valuation")
ticker = st.sidebar.text_input("Ticker Symbol", value=DEFAULT_PARAMS['TICKER'])
P_CURRENT = st.sidebar.number_input("Current Stock Price P ($)", value=DEFAULT_PARAMS['P_CURRENT'], format="%.2f")
V_TARGET = st.sidebar.number_input("Target Price V_target ($)", value=DEFAULT_PARAMS['V_TARGET'], format="%.2f")
V_HARD_FLOOR = st.sidebar.number_input("Hard Floor (Strike) V_hard ($)", value=DEFAULT_PARAMS['V_HARD_FLOOR'], format="%.2f")
V_FILL_PLAN = st.sidebar.number_input("Refill Plan Price V_fill ($)", value=DEFAULT_PARAMS['V_FILL_PLAN'], format="%.2f")

# --- Sidebar Inputs: Statistical ---
st.sidebar.subheader("Statistical & Risk Parameters")
LAMBDA = st.sidebar.number_input("Annualized Lambda (λ)", value=DEFAULT_PARAMS['LAMBDA'], format="%.4f", help="Get from Step 0 Market Diagnosis")
SIGMA_ASSET = st.sidebar.number_input("Asset Real Volatility (σ)", value=DEFAULT_PARAMS['SIGMA_ASSET'], format="%.4f", help="Get from Volatility Calculator")
IV_PRICING = st.sidebar.number_input("Option Pricing Volatility (IV)", value=DEFAULT_PARAMS['IV_PRICING'], format="%.4f", help="Used for Black-Scholes Pricing")
R_RISKFREE = st.sidebar.number_input("Risk Free Rate (r_f)", value=DEFAULT_PARAMS['R_RISKFREE'], format="%.4f")


if st.sidebar.button("Run Optimal Expiry Solver", type="primary"):
    st.session_state['run_solver'] = True
else:
    st.session_state['run_solver'] = False

# --- Main Content Execution ---
if st.session_state.get('run_solver', False):

    st.subheader("⚠️ Note: The current solver uses the original code's hardcoded defaults of K=0.5 and Beta=0.2 for calculation.")

    try:
        # Call the core solver function
        best, data = find_perfect_expiry(
            ticker, P_CURRENT, V_TARGET, V_HARD_FLOOR, V_FILL_PLAN,
            LAMBDA, SIGMA_ASSET, IV_PRICING, R_RISKFREE
        )

        st.success("✅ Optimal Expiry Calculation Complete.")

        st.subheader("Analysis Results")
        col_r1, col_r2, col_r3 = st.columns(3)

        with col_r1:
            st.metric("Optimal Expiry", f"{int(best['Days'])} Days", f"~{best['Days']/30.4:.1f} Months")
        with col_r2:
            st.metric("Suggested Allocation (Cap)", f"{best['Pilot_Cap']:.2%}")
        with col_r3:
            st.metric("Option Price (BS Valuation)", f"${best['Option_Price']:.2f}")

        st.markdown("---")
        st.markdown("##### Offense vs. Defense Curve Plot")
        st.caption(f"The optimal point is the intersection of the Offense Curve (0.5 * Kelly) and the Defense Cap (Pilot Cash Cap).")

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(data['Days'], data['Kelly_Half'], label='Offense: 0.5 * Kelly Ratio',
                 color='blue', linewidth=2, linestyle='--')

        ax.plot(data['Days'], data['Pilot_Cap'], label='Defense: Pilot Cash Cap (1:1 Refill)',
                 color='red', linewidth=2)

        # Mark optimal point
        ax.scatter(best['Days'], best['Pilot_Cap'], color='green', s=150, zorder=5, label='Optimal Expiry')

        ax.annotate(
            f"Sweet Spot\n{int(best['Days'])} Days\n{best['Pilot_Cap']:.1%} Alloc",
            xy=(best['Days'], best['Pilot_Cap']),
            xytext=(best['Days']+100, best['Pilot_Cap']+0.1),
            arrowprops=dict(facecolor='black', shrink=0.05),
            fontsize=10, fontweight='bold'
        )

        ax.set_title(f"Optimal Expiry Solver: {ticker} (Strike=${V_HARD_FLOOR}, Fill @ ${V_FILL_PLAN})", fontsize=14)
        ax.set_xlabel("Days to Expiration", fontsize=12)
        ax.set_ylabel("Position Allocation %", fontsize=12)
        ax.axhline(best['Pilot_Cap'], color='gray', linestyle=':', alpha=0.5)
        ax.axvline(best['Days'], color='gray', linestyle=':', alpha=0.5)

        ax.set_xticks(np.arange(0, 1100, 180))

        ax.legend(fontsize=12, loc='best')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

        st.markdown("---")
        st.warning("⚠️ **Next Step** : Use the **Actual Option Price**, **Delta**, and **Theta** corresponding to the optimal expiry date to calculate the final position in the main calculator (`app_dashboard.py`).")


    except Exception as e:
        st.error(f"An error occurred while running the solver: {e}")

st.info("Please enter the asset information and statistical parameters (Lambda, Sigma, IV) in the sidebar and click Run.")
st.caption("Note: This tool executes the functionality of Step 0.5 outlined in the User Guide.")