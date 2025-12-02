# code/data/sigma_tools.py

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.ticker as mtick # Keep import but remove usage
import sys

def get_sigma(tickers, period="5y", window=252, percentile=0.80,
              annualize=True, safety_lock=False):
    """
    Calculates the target volatility (sigma) based on the specified percentile
    of historical rolling volatility.
    (Modified to remove plotting logic.)

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

    if 'Close' in data.columns:
        data = data['Close']

    if isinstance(data, pd.Series):
        data = data.to_frame(name=tickers_list[0])

    data = data.dropna(how="all")
    data = data.dropna(axis=1, how="all")

    if data.empty:
        raise RuntimeError("No valid close price data after dropping NaNs.")

    # Compute returns
    returns = data.pct_change().dropna()
    if returns.empty:
        raise RuntimeError("Not enough data to compute returns.")

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
            static_vol = r.std() * annual_factor
            sigma_val = float(static_vol)
        else:
            pval = float(roll_vol.quantile(percentile))
            current = float(roll_vol.iloc[-1])

            if safety_lock:
                sigma_val = max(current, pval)
            else:
                sigma_val = pval

        sigma_dict[t] = sigma_val

    return sigma_dict, corr_matrix, cov_matrix, rolling_series_dict