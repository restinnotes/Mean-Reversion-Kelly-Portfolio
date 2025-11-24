import yfinance as yf
import pandas as pd
import numpy as np

def get_sigma(tickers, period="3y", annualize=True):
    """
    获取标的波动率和相关性矩阵
    Get asset volatilities, correlation and covariance matrices

    Parameters
    ----------
    tickers : list[str]
        股票代码列表 / List of ticker symbols
    period : str
        数据周期，例如 '3y', '1y', '6mo'
        Data period, e.g., '3y', '1y', '6mo'
    annualize : bool
        是否年化波动率（默认True，按252交易日）
        Whether to annualize daily volatility (default True, 252 trading days)

    Returns
    -------
    sigma_dict : dict
        每个标的的波动率 {'NVDA': 0.45, ...}
        Annualized volatility for each ticker
    corr_matrix : pd.DataFrame
        收益率相关系数矩阵 / Correlation matrix of returns
    cov_matrix : pd.DataFrame
        收益率协方差矩阵 / Covariance matrix of returns
    """
    # 下载收盘价数据 / Download adjusted close price
    data = yf.download(tickers, period=period)['Close'].dropna()

    # 计算日收益率 / Compute daily returns
    returns = data.pct_change().dropna()

    # 计算协方差矩阵 / Covariance matrix
    cov_matrix = returns.cov()

    # 计算相关系数矩阵 / Correlation matrix
    corr_matrix = returns.corr()

    # 计算每个标的波动率 / Compute individual volatilities
    sigma_dict = {}
    for t in tickers:
        vol = returns[t].std()
        if annualize:
            vol *= np.sqrt(252)
        sigma_dict[t] = vol

    return sigma_dict, corr_matrix, cov_matrix


# --------------------------
# 使用示例 / Example
# --------------------------
if __name__ == "__main__":
    tickers = ['NVDA', 'META', 'MSFT']
    sigma_dict, corr_matrix, cov_matrix = get_sigma(tickers)

    print("年化波动率 / Annualized volatility:")
    print(sigma_dict)
    print("\n相关系数矩阵 / Correlation matrix:")
    print(corr_matrix)