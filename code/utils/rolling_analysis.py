import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# ==========================================
# 1. SETUP: Path & Fonts
# ==========================================
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Verdana']
plt.rcParams['axes.unicode_minus'] = False

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Ensure utils is in path if lambda_tools is there
utils_dir = os.path.join(current_dir, "utils")
if os.path.exists(utils_dir) and utils_dir not in sys.path:
    sys.path.append(utils_dir)

try:
    from lambda_tools import calculate_ou_params
except ImportError:
    try:
        sys.path.append(os.path.join(current_dir, "utils"))
        from lambda_tools import calculate_ou_params
    except ImportError:
        print(f"[Error] Cannot import lambda_tools.")
        sys.exit(1)

project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
pe_csv_dir = os.path.join(project_root, "pe_csv")

# ==========================================
# 2. MONTE CARLO ENGINE
# ==========================================
def run_simulation(current_pe, target_pe, lambda_annual, sigma_daily, days_to_simulate=252, num_paths=10000):
    """
    Runs Monte Carlo simulation for P/E ratio using the Ornstein-Uhlenbeck process.

    Parameters:
    - current_pe (float): Starting value (X_0).
    - target_pe (float): Long-term mean (Theta).
    - lambda_annual (float): Annualized mean-reversion rate (Lambda).
    - sigma_daily (float): Daily volatility (Sigma).
    - days_to_simulate (int): Number of time steps.
    - num_paths (int): Number of simulated paths.

    Returns:
    - numpy.ndarray: Simulated paths.
    """
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
    """
    Analyzes simulated paths to calculate Touch and Hold probabilities.
    """
    days_simulated = paths.shape[0] - 1
    # Check points in TRADING DAYS
    check_points = [21, 42, 63, 126, 189, 252]
    results = []
    is_long = target_pe > current_pe

    for day in check_points:
        if day > days_simulated: continue
        final_values = paths[day]

        # Hold Prob: probability of ending at or beyond the target
        if is_long: prob_end = np.mean(final_values >= target_pe)
        else: prob_end = np.mean(final_values <= target_pe)

        # Touch Prob: probability of hitting or crossing the target at any point
        path_slice = paths[:day+1, :]
        if is_long: has_hit = np.any(path_slice >= target_pe, axis=0)
        else: has_hit = np.any(path_slice <= target_pe, axis=0)
        prob_touch = np.mean(has_hit)

        # Convert Trading Days to Approx Calendar Months for Clarity
        approx_months = day / 21
        approx_cal_days = int(day * (365/252))

        results.append({
            "Trading Days": day,
            "~Calendar Days": f"{approx_cal_days}d",
            "~Months": f"{approx_months:.1f} Mo",
            "Touch Prob": prob_touch,
            "Hold Prob": prob_end,
            "Expected PE": np.mean(final_values)
        })
    return pd.DataFrame(results)

# ==========================================
# 3. ROLLING ANALYSIS
# ==========================================
def run_rolling_analysis(ticker, window_days=90):
    """
    Performs rolling OU parameter estimation, plots diagnostics, and runs MC simulation.
    """
    csv_path = os.path.join(pe_csv_dir, f"{ticker}_pe.csv")
    if not os.path.exists(csv_path):
        print(f"[Error] Data not found: {csv_path}")
        return

    print(f"Reading data: {csv_path} ...")
    df = pd.read_csv(csv_path, parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # Context Data
    df['rolling_mean'] = df['value'].rolling(window=window_days).mean()

    dates = []
    pe_values = []
    pe_means = []
    lambdas_annual = []
    half_lives = []
    sigmas_daily = []

    print(f"Calculating rolling metrics (Window={window_days}d)...")

    for i in range(len(df)):
        if i < window_days: continue

        window_data = df.iloc[i-window_days+1 : i+1]
        series = window_data.set_index('date')['value']
        ou = calculate_ou_params(series)

        if ou:
            dates.append(df.iloc[i]['date'])
            pe_values.append(df.iloc[i]['value'])
            pe_means.append(df.iloc[i]['rolling_mean'])
            lambdas_annual.append(ou['lambda'] * 252)
            half_lives.append(ou['half_life'])
            sigmas_daily.append(ou['sigma'])

    if not lambdas_annual: return

    # --- Plotting the 3-Pane Diagnosis Chart ---
    fast_threshold = np.percentile(lambdas_annual, 80)
    slow_threshold = np.percentile(lambdas_annual, 20)
    percentile_90_hl = np.percentile(half_lives, 90)

    current_lambda = lambdas_annual[-1]
    current_hl = half_lives[-1]
    current_pe = pe_values[-1]
    current_mean = pe_means[-1]
    current_sigma = sigmas_daily[-1]

    # Draw Plots
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(12, 14), sharex=True)

    # Plot 0: PE Context
    ax0.plot(dates, pe_values, 'k', alpha=0.8, label='PE Ratio')
    ax0.plot(dates, pe_means, 'b--', label=f'{window_days}d Moving Avg')
    ax0.set_title(f'{ticker} PE Ratio vs {window_days}d MA', fontsize=12, fontweight='bold')
    ax0.legend(loc='upper left'); ax0.grid(True, alpha=0.3)

    # Plot 1: Lambda
    ax1.plot(dates, lambdas_annual, color='#1f77b4', label='Annualized Lambda')
    ax1.axhline(fast_threshold, color='r', linestyle='--', label=f'Fast >{fast_threshold:.1f}')
    ax1.axhline(slow_threshold, color='g', linestyle='--', label=f'Slow <{slow_threshold:.1f}')
    ax1.set_title('Reversion Speed (Lambda)', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left'); ax1.grid(True, alpha=0.3)

    # Plot 2: Half-Life
    ax2.plot(dates, half_lives, color='#ff7f0e', label='Half-Life (Days)')
    ax2.axhline(percentile_90_hl, color='purple', linestyle='--', label=f'90%ile Risk ({percentile_90_hl:.1f}d)')
    ax2.set_ylim(0, 300)
    ax2.set_title('Implied Half-Life (Risk)', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left'); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # --- Diagnostic Report ---
    print("\n" + "="*60)
    print(f"   MARKET REGIME DIAGNOSIS: {ticker} ({dates[-1].date()})")
    print("="*60)
    print(f"Current PE          : {current_pe:.2f} (Avg: {current_mean:.2f})")
    print(f"Current Lambda      : {current_lambda:.4f}")
    print(f"Current Half-Life   : {current_hl:.2f} Days")
    print("-" * 30)

    # --- Trigger Monte Carlo Simulation ---
    print("\n>>> STARTING VALUATION REPAIR SIMULATION <<<")
    # Define Target: Let's assume Mean Reversion to the 90d Average
    P_target = current_mean
    # Or user manual override:
    # P_target = 33.0

    print(f"Goal: Repair from PE {current_pe:.2f} -> PE {P_target:.2f}")
    print(f"Using Stats: Lambda={current_lambda:.2f}, Sigma={current_sigma:.4f}")

    paths = run_simulation(current_pe, P_target, current_lambda, current_sigma)
    df_probs = analyze_probabilities(paths, P_target, current_pe)

    print("\n[WIN RATE TABLE]")
    print(df_probs.to_string(index=False, formatters={
        "Touch Prob": "{:.1%}".format,
        "Hold Prob": "{:.1%}".format,
        "Expected PE": "{:.2f}".format
    }))

    print("-" * 60)
    print(f"RECOMMENDATION:")
    # Find timeframe with > 90% Touch Prob
    safe_days = 0
    safe_cal_days = 0
    for idx, row in df_probs.iterrows():
        if row['Touch Prob'] > 0.9:
            print(f"   > 90% Probability to touch target within: {row['Trading Days']} Trading Days (~{row['~Calendar Days']})")
            safe_days = int(row['Trading Days'])
            safe_cal_days = int(row['~Calendar Days'].replace('d',''))
            break

    if safe_days > 0:
        # Suggest 3x margin
        rec_expiry = safe_cal_days * 3
        print(f"   > Suggested Expiry (3x Safety): > {rec_expiry} Calendar Days")
        print(f"     (Buy options expiring in approx {rec_expiry/30:.1f} Months)")
    else:
        print(f"   > Reversion is slow/uncertain. Buy > 1 Year LEAPS.")
    print("="*60)

if __name__ == "__main__":
    run_rolling_analysis('NVDA', window_days=90)