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
# Correct path: go up one level '..' to the project root
project_root = os.path.abspath(os.path.join(current_dir, ".."))

# Add utilities path
sys.path.append(os.path.join(project_root, "code", "utils"))

try:
    from utils.lambda_tools import calculate_ou_params
except ImportError:
    st.error("Could not import utils.lambda_tools. Please check the util/ path and file integrity.")
    st.stop()


PE_CSV_DIR_NAME = "pe_csv"

def get_pe_data(ticker):
    """Loads PE data file"""
    # Use the correct project_root to construct the CSV path
    csv_path = os.path.join(project_root, PE_CSV_DIR_NAME, f"{ticker}_pe.csv")

    if not os.path.exists(csv_path):
        st.error(f"[ERROR] PE data file not found: {ticker}_pe.csv.")
        st.markdown(f"**Expected Search Path**: `{os.path.join(os.path.basename(project_root), PE_CSV_DIR_NAME)}/`")
        st.markdown(f"Please ensure Streamlit is run from the project root directory and the file structure is correct.")
        return None

    try:
        df = pd.read_csv(csv_path, parse_dates=['date'])
        df = df.sort_values('date').reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"[ERROR] Failed to read CSV file: {e}")
        return None


# Monte Carlo Simulation Function (Translated from rolling_analysis.py)
def run_simulation(current_pe, target_pe, lambda_annual, sigma_daily, days_to_simulate=252, num_paths=10000):
    dt = 1/252
    paths = np.zeros((days_to_simulate + 1, num_paths))
    paths[0] = current_pe
    theta = target_pe # Long-term mean target

    for t in range(1, days_to_simulate + 1):
        X_t = paths[t-1]
        drift = lambda_annual * (theta - X_t) * dt
        shock = sigma_daily * np.random.normal(0, 1, num_paths)
        paths[t] = X_t + drift + shock
    return paths

# Probability Analysis Function (Translated from rolling_analysis.py)
def analyze_probabilities(paths, target_pe, current_pe):
    days_simulated = paths.shape[0] - 1
    check_points = [21, 42, 63, 126, 189, 252]
    results = []
    is_long = target_pe > current_pe

    for day in check_points:
        if day > days_simulated: continue
        final_values = paths[day]

        # Hold Prob: ended at or beyond the target
        if is_long: prob_end = np.mean(final_values >= target_pe)
        else: prob_end = np.mean(final_values <= target_pe)

        # Touch Prob: hit the target at any point
        path_slice = paths[:day+1, :]
        if is_long: has_hit = np.any(path_slice >= target_pe, axis=0)
        else: has_hit = np.any(path_slice <= target_pe, axis=0)
        prob_touch = np.mean(has_hit)

        approx_cal_days = int(day * (365/252))

        results.append({
            "Trading Days": day,
            "~Calendar Days": f"{approx_cal_days}d",
            "Touch Probability": prob_touch,
            "Hold Probability": prob_end,
            "Expected PE": np.mean(final_values)
        })
    return pd.DataFrame(results)


def run_rolling_analysis_gui(ticker, window_days=90):
    df = get_pe_data(ticker)
    if df is None:
        return

    # --- 1. Calculate Rolling Metrics ---
    df['rolling_mean'] = df['value'].rolling(window=window_days).mean()
    dates = []; pe_values = []; pe_means = []; lambdas_annual = []; half_lives = []; sigmas_daily = []

    status_placeholder = st.empty()
    progress_bar = st.progress(0)
    num_iterations = len(df)

    for i in range(num_iterations):
        if i < window_days - 1:
            progress_bar.progress((i + 1) / num_iterations)
            continue

        window_data = df.iloc[i-window_days+1 : i+1]
        series = window_data.set_index('date')['value']
        try:
            ou = calculate_ou_params(series)
        except Exception:
            # Skip iterations where calculation fails
            progress_bar.progress((i + 1) / num_iterations)
            continue

        if ou:
            dates.append(df.iloc[i]['date'])
            pe_values.append(df.iloc[i]['value'])
            pe_means.append(df.iloc[i]['rolling_mean'])
            lambdas_annual.append(ou['lambda'] * 252)
            half_lives.append(ou['half_life'])
            sigmas_daily.append(ou['sigma'])

        progress_bar.progress((i + 1) / num_iterations)

    progress_bar.empty()
    status_placeholder.text("Rolling metrics calculation complete.")


    if not lambdas_annual:
        st.warning("Not enough historical data points to calculate rolling metrics.")
        return

    # --- 2. Diagnosis Report ---
    current_lambda = lambdas_annual[-1]
    current_hl = half_lives[-1]
    current_pe = pe_values[-1]
    current_mean = pe_means[-1]
    current_sigma = sigmas_daily[-1]

    st.subheader("Diagnosis Report & Monte Carlo Simulation")
    st.markdown("---")
    col_d1, col_d2 = st.columns(2)

    with col_d1:
        st.markdown("**PE Valuation Status**")
        st.code(f"Current PE: {current_pe:.2f}")
        st.code(f"{window_days}d Moving Avg: {current_mean:.2f}")

    with col_d2:
        st.markdown("**Reversion & Volatility**")
        st.code(f"Annualized Lambda (λ): {current_lambda:.4f}")
        st.code(f"Half-Life: {current_hl:.2f} Days")
        st.code(f"Daily Volatility (σ): {current_sigma:.4f}")

    st.markdown("---")

    # --- 3. Monte Carlo Simulation ---
    st.markdown("##### Monte Carlo Simulation Results")
    st.caption(f"Goal: PE {current_pe:.2f} reverts to mean PE {current_mean:.2f} | Paths: 10,000")

    paths = run_simulation(current_pe, current_mean, current_lambda, current_sigma)
    df_probs = analyze_probabilities(paths, current_mean, current_pe)

    safe_days = 0
    safe_cal_days = 0
    found_safe_zone = False

    for idx, row in df_probs.iterrows():
        if row['Touch Probability'] > 0.9:
            safe_days = int(row['Trading Days'])
            safe_cal_days = int(row['~Calendar Days'].replace('d',''))
            found_safe_zone = True
            break

    df_probs['Touch Probability'] = df_probs['Touch Probability'].apply(lambda x: f"{x:.1%}")
    df_probs['Hold Probability'] = df_probs['Hold Probability'].apply(lambda x: f"{x:.1%}")
    df_probs['Expected PE'] = df_probs['Expected PE'].apply(lambda x: f"{x:.2f}")
    st.dataframe(df_probs, hide_index=True)

    if found_safe_zone:
        st.success(f"**[Recommended Action Plan]**: The minimum time for 90% probability to touch the target is **{safe_days} Trading Days (~{safe_cal_days} Calendar Days)**.")
        st.info(f"Selection Advice: Purchase LEAPS options with Expiry **Greater Than or Equal To** {safe_cal_days} Calendar Days.")
    else:
        st.warning(f"**[WARNING]**: 90% target touch probability not reached within 1 year. Reversion is slow/uncertain. Recommend buying > 1 Year LEAPS or staying in cash.")

    st.markdown("---")

    # --- 4. Plotting ---
    plot_df = pd.DataFrame({
        'Date': dates,
        'PE_Ratio': pe_values,
        'MA': pe_means,
        'Lambda': lambdas_annual,
        'Half_Life': half_lives,
    }).set_index('Date')

    if len(lambdas_annual) > 1:
        fast_threshold = np.percentile(lambdas_annual, 80)
        slow_threshold = np.percentile(lambdas_annual, 20)
    else:
        fast_threshold = current_lambda * 1.1
        slow_threshold = current_lambda * 0.9

    percentile_90_hl = np.percentile(half_lives, 90)

    # Plot 1: PE Context
    fig1, ax0 = plt.subplots(figsize=(10, 3))
    ax0.plot(plot_df.index, plot_df['PE_Ratio'], 'k', alpha=0.8, label='PE Ratio')
    ax0.plot(plot_df.index, plot_df['MA'], 'b--', label=f'{window_days}d Moving Avg')
    ax0.set_title(f'{ticker} PE Ratio vs {window_days}d MA', fontsize=10)
    ax0.legend(loc='upper left'); ax0.grid(True, alpha=0.3)
    st.pyplot(fig1)

    # Plot 2: Lambda
    fig2, ax1 = plt.subplots(figsize=(10, 3))
    ax1.plot(plot_df.index, plot_df['Lambda'], color='#1f77b4', label='Annualized Lambda')
    ax1.axhline(fast_threshold, color='r', linestyle='--', label=f'Fast >{fast_threshold:.1f}')
    ax1.axhline(slow_threshold, color='g', linestyle='--', label=f'Slow <{slow_threshold:.1f}')
    ax1.set_title('Reversion Speed (Lambda)', fontsize=10)
    ax1.legend(loc='upper left'); ax1.grid(True, alpha=0.3)
    st.pyplot(fig2)

    # Plot 3: Half-Life
    fig3, ax2 = plt.subplots(figsize=(10, 3))
    ax2.plot(plot_df.index, plot_df['Half_Life'], color='#ff7f0e', label='Half-Life (Days)')
    ax2.axhline(percentile_90_hl, color='purple', linestyle='--', label=f'90%ile Risk ({percentile_90_hl:.1f}d)')
    ax2.set_ylim(0, max(300, percentile_90_hl * 1.5))
    ax2.set_title('Implied Half-Life (Risk)', fontsize=10)
    ax2.legend(loc='upper left'); ax2.grid(True, alpha=0.3)
    st.pyplot(fig3)

    plt.close('all')


# --- Streamlit Boilerplate ---
st.set_page_config(page_title="Market Diagnosis - Rolling Analysis", layout="wide", page_icon="📈")
st.title("📈 Step 0: Market Diagnosis - Rolling Analysis")

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("Configuration")
    ticker = st.text_input("Ticker Symbol", value="NVDA").upper()
    window_days = st.slider("Rolling Window (Trading Days)", min_value=30, max_value=252, value=90, step=10)

    if st.button("Run Analysis & Diagnosis", type="primary"):
        st.session_state['run_analysis'] = True

# --- Main Content ---
if st.session_state.get('run_analysis', False):
    run_rolling_analysis_gui(ticker, window_days)
    st.session_state['run_analysis'] = False