import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import sys

def get_sigma(tickers, period="5y", window=252, percentile=0.80,
              annualize=True, plot=False, safety_lock=False):
    """
    Calculates the target volatility (sigma) based on the specified percentile
    of historical rolling volatility.

    Parameters
    ----------
    tickers : str or list[str]
        One or more ticker symbols.
    period : str
        Data history period (e.g., "5y", "1y", "max").
    window : int
        Rolling window size for volatility calculation (e.g., 252 for 1 year).
    percentile : float (0-1)
        The percentile of historical rolling volatility to use as sigma.
        e.g. 0.80 -> 80th percentile.
    annualize : bool
        If True, annualize the volatility using sqrt(252).
    plot : bool
        If True, display a diagnostic plot for each ticker.
    safety_lock : bool
        If True, sigma is set to max(current_vol, percentile_vol).

    Returns
    -------
    sigma_dict : dict
        Calculated sigma for each ticker.
    corr_matrix : pd.DataFrame
        Correlation matrix of returns over the full period.
    cov_matrix : pd.DataFrame
        Covariance matrix of returns over the full period.
    rolling_series_dict : dict
        Rolling volatility series for each ticker.
    """
    # normalize tickers to list
    if isinstance(tickers, str):
        tickers_list = [tickers]
    else:
        tickers_list = list(tickers)

    # Download close prices
    data = yf.download(tickers_list, period=period, progress=False)
    if data is None or data.empty:
        raise RuntimeError("No data returned by yfinance. Check tickers / network.")

    # yfinance multi-index when multiple tickers; safe access to 'Close'
    if 'Close' in data.columns:
        data = data['Close']

    # Ensure DataFrame even for single ticker
    if isinstance(data, pd.Series):
        data = data.to_frame(name=tickers_list[0])

    # drop rows with all NaN
    data = data.dropna(how="all")
    # drop columns with all NaN (tickers not downloaded)
    data = data.dropna(axis=1, how="all")

    if data.empty:
        raise RuntimeError("No valid close price data after dropping NaNs.")

    # Compute returns
    returns = data.pct_change().dropna()
    if returns.empty:
        raise RuntimeError("Not enough data to compute returns.")

    # Cov / Corr computed on entire returns window
    cov_matrix = returns.cov()
    corr_matrix = returns.corr()

    annual_factor = np.sqrt(252) if annualize else 1.0

    sigma_dict = {}
    rolling_series_dict = {}

    for t in tickers_list:
        if t not in returns.columns:
            print(f"[Warning] {t} not found in downloaded data — skipping.")
            continue

        r = returns[t].dropna()
        if r.empty:
            print(f"[Warning] no returns for {t} — skipping.")
            continue

        # rolling vol
        roll_vol = r.rolling(window=window).std() * annual_factor
        roll_vol = roll_vol.dropna()
        rolling_series_dict[t] = roll_vol

        if len(roll_vol) == 0:
            # fallback: static vol of whole period
            static_vol = r.std() * annual_factor
            print(f"[{t}] WARNING: not enough points for rolling window -> using static vol {static_vol:.2%}")
            sigma_val = float(static_vol)
        else:
            pval = float(roll_vol.quantile(percentile))
            current = float(roll_vol.iloc[-1])

            if safety_lock:
                sigma_val = max(current, pval)
                print(f"[{t}] current={current:.2%}, {percentile*100:.0f}%ile={pval:.2%} -> using max => {sigma_val:.2%}")
            else:
                sigma_val = pval
                print(f"[{t}] {percentile*100:.0f}%ile Sigma = {pval:.2%} (current {current:.2%})")

        sigma_dict[t] = sigma_val

        # Plot if requested
        if plot and len(roll_vol) > 0:
            plt.figure(figsize=(11,4))
            plt.plot(roll_vol.index, roll_vol, linewidth=1.4, label=f'{window}d Rolling Vol')
            plt.axhline(pval, linestyle='--', linewidth=1.5, label=f'{percentile*100:.0f}%ile = {pval:.2%}')
            plt.scatter(roll_vol.index[-1], current, color='red', s=50, zorder=5, label=f'Current = {current:.2%}')
            plt.title(f"{t} Rolling Volatility ({window}d) — {percentile*100:.0f}%ile Sigma")
            plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
            plt.legend(loc='upper left')
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()

    return sigma_dict, corr_matrix, cov_matrix, rolling_series_dict


if __name__ == "__main__":
    # --- Simple Test Case ---
    # In a GUI environment, plot=True will display figures; in a headless environment,
    # it won't display but will still return the data.
    TEST_TICKERS = ["NVDA", "MSFT"]
    print("Downloading and computing... (this may take a few seconds)")

    sigs, corr, cov, rseries = get_sigma(
        TEST_TICKERS,
        period="5y",
        window=252,
        percentile=0.80,
        annualize=True,
        plot=True,
        safety_lock=False
    )

    print("\nReturned sigma_dict (percentile-based):")
    for k,v in sigs.items():
        print(f"   {k}: {v:.4f}  ({v:.2%})")

    print("\nCorrelation matrix (head):")
    print(corr.round(3))