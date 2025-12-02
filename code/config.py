# code/config.py

# Default values for lambda and sigma (from app_unified_zh.py)
DEFAULT_LAMBDA = 6.0393
DEFAULT_SIGMA = 0.6082

# Default constants for Streamlit app
DEFAULT_APP_PARAMS = {
    'r_f': 0.037, 'k_factor': 0.50, 'beta': 0.20, 'P': 180.00,
    'V_target': 225.00, 'V_hard': 130.00, 'V_fill': 145.00,
    'iv_pricing': 0.5100, 'opt_price': 61.60, 'delta': 0.8446,
    'theta': 0.0425, 'ticker': "NVDA",
    'lambda': DEFAULT_LAMBDA,
    'sigma': DEFAULT_SIGMA,
    'portfolio_data': [], 'window_days': 90,
    'days_to_expiry': 365,
    'k_fill': 1.0,
    'total_capital': 100000.0,
    'P_anchor_global': 180.00
}