# code/ui/zh/sidebar.py

import streamlit as st
import os
import sys
import json

# Import Data/Config modules
from data.fetcher import get_ou_for_ticker, get_sigma
from config import DEFAULT_LAMBDA, DEFAULT_SIGMA
from ui.plot_utils import get_resource_root

project_root = get_resource_root()

def render_sidebar():
    """
    Renders the entire Streamlit sidebar and handles data fetching.
    (Extracted and refactored from app_unified_zh.py)
    """
    # --- Sidebar Inputs ---
    with st.sidebar:
        st.title("导航与全局参数")

        # ==========================================
        # [NEW] 配置管理 (导入/导出)
        # ==========================================
        with st.expander("💾 配置管理 (导入/导出)", expanded=False):
            st.caption("保存当前 Step 2 的组合和所有参数设置。")

            # 定义需要保存的 Session State 键名
            config_keys = [
                'ticker', 'lambda', 'sigma', 'r_f', 'k_factor', 'beta',
                'P', 'V_target', 'V_hard', 'V_fill',
                'iv_pricing', 'opt_price', 'delta', 'theta',
                'window_days', 'days_to_expiry', 'k_fill', 'total_capital',
                'P_anchor_global', 'portfolio_data', 'max_leverage_cap'
            ]

            # 1. 导出
            current_config = {}
            for k in config_keys:
                if k in st.session_state:
                    val = st.session_state[k]
                    try:
                        json.dumps(val)
                        current_config[k] = val
                    except:
                        pass

            json_str = json.dumps(current_config, indent=4, ensure_ascii=False)

            st.download_button(
                label="📤 导出配置 (JSON)",
                data=json_str,
                file_name="kelly_config.json",
                mime="application/json"
            )

            # 2. 导入
            uploaded_file = st.file_uploader("📥 导入配置", type=['json'])
            if uploaded_file is not None:
                try:
                    data = json.load(uploaded_file)
                    for k, v in data.items():
                        st.session_state[k] = v

                    st.success("配置已加载！")
                    if st.button("🔄 刷新页面生效"):
                        st.rerun()
                except Exception as e:
                    st.error(f"导入失败: {e}")

        st.divider()
        # ==========================================

        page = st.radio("选择工具页面",
                        ("Step 0: 市场诊断",
                         "Step 0.5: 最优期限求解",
                         "Step 1: 主仓位计算器",
                         "Step 2: 多标的组合管理",
                         "📚 术语与原理速查"),
                        key='page_select', index=0)

        st.header("1. 资产与统计数据")

        # --- 1. 输入框 ---
        ticker = st.text_input("股票代码 (Ticker)", value=st.session_state.ticker, key='ticker_global').upper()

        # [MODIFIED] 修改了说明，强调这是唯一的全局价格输入
        current_P_anchor_global = st.number_input(
            "当前股价 P ($)",
            value=st.session_state.P_anchor_global,
            key='P_anchor_global',
            format="%.2f",
            help="全局统一的当前标的价格。用于 Step 0 的估值计算以及 Step 1 的凯利公式计算。"
        )

        # --- 2. 自动获取数据逻辑 (Auto-Fetch) ---
        need_refresh = (ticker != st.session_state.get('last_fetched_ticker')) or \
                       ('sigma_dict' not in st.session_state) or \
                       (st.session_state.get('lambda') is None)

        if need_refresh:
            if 'get_ou_for_ticker' in globals() or 'get_sigma' in globals():
                try:
                    with st.spinner(f"正在自动计算 {ticker} 的历史波动率与回归参数..."):

                        ou_window = st.session_state.get('window_days', 90)
                        ou = get_ou_for_ticker(ticker, project_root, window=ou_window)

                        new_lambda = DEFAULT_LAMBDA
                        if ou and ou["lambda"] is not None:
                             new_lambda = ou["lambda"] * 252.0

                        sigma_dict, _, _, rolling_series_dict = get_sigma(
                            [ticker], period="5y", window=252, percentile=0.85, annualize=True, safety_lock=True
                        )
                        new_sigma = sigma_dict.get(ticker, DEFAULT_SIGMA)

                        # === 更新 Session State ===
                        st.session_state['lambda'] = new_lambda
                        st.session_state['sigma'] = new_sigma
                        st.session_state['ticker'] = ticker

                        st.session_state['sigma_rolling_data'] = rolling_series_dict
                        st.session_state['sigma_dict'] = sigma_dict
                        st.session_state['last_fetched_ticker'] = ticker

                except Exception as e:
                    st.error(f"❌ 数据获取失败: {e}")
            else:
                st.error("依赖模块未导入，无法获取历史数据。")

        # --- 3. 如果数据已就绪，显示简报 ---
        current_lambda_val = st.session_state.get('lambda')
        current_sigma_val = st.session_state.get('sigma')

        lambda_display = 'N/A'
        sigma_display = 'N/A'

        if current_lambda_val is not None:
            lambda_display = f"{current_lambda_val:.4f}"

        if current_sigma_val is not None:
            sigma_display = f"{current_sigma_val:.2%}"

        if st.session_state.get('last_fetched_ticker') == ticker:
            st.caption(f"✅ 已加载: λ={lambda_display}, σ={sigma_display}")

        st.divider()

        # Update Session State from inputs
        lambda_val = st.number_input("年化 Lambda (λ)", value=current_lambda_val if current_lambda_val is not None else DEFAULT_LAMBDA, key='lambda_global', format="%.4f",
                                     help="【均值回归动力】数值越大，修复越快。若图表显示 Lambda 处于历史极高位(>80分位)，建议手动调低以提高安全边际。")
        sigma_val = st.number_input("年化 Sigma (σ)", value=current_sigma_val if current_sigma_val is not None else DEFAULT_SIGMA, key='sigma_global', format="%.4f",
                                     help="【保守波动率】通常取历史 85% 分位数。用于计算凯利公式的分母(风险)。")

        st.session_state['lambda'] = lambda_val
        st.session_state['sigma'] = sigma_val


        # --- Page-specific Input Logic ---
        st.header("2. 策略与市场参数 (动态)")

        # Get current values for dynamic updating
        current_r_f = st.session_state.get('r_f', 0.037)
        current_k_factor = st.session_state.get('k_factor', 0.50)
        current_beta = st.session_state.get('beta', 0.20)

        # [MODIFIED] 这里的 current_P 默认取全局 anchor
        current_P = current_P_anchor_global

        current_V_target = st.session_state.get('V_target', 225.00)
        current_V_hard = st.session_state.get('V_hard', 130.00)
        current_V_fill = st.session_state.get('V_fill', 145.00)
        current_iv_pricing = st.session_state.get('iv_pricing', 0.5100)
        current_opt_price = st.session_state.get('opt_price', 61.60)
        current_delta = st.session_state.get('delta', 0.8446)
        current_theta = st.session_state.get('theta', 0.0425)
        current_window_days = st.session_state.get('window_days', 90)
        current_days_to_expiry = st.session_state.get('days_to_expiry', 365)
        current_k_fill = st.session_state.get('k_fill', 1.0)
        current_total_capital = st.session_state.get('total_capital', 100000.0)

        # NOTE: Only expose relevant inputs based on the selected page.
        if page == "Step 0: 市场诊断":
            st.subheader("诊断特有参数")
            window_days = st.slider("滚动窗口 (交易日)", min_value=30, max_value=252, value=current_window_days, key='window_days_diag')
            st.session_state['window_days'] = window_days

        elif page in ["Step 1: 主仓位计算器", "Step 0.5: 最优期限求解"]:
            st.subheader("2.1 策略约束")
            if page == "Step 1: 主仓位计算器":
                current_total_capital = st.number_input("账户本金 ($)", value=current_total_capital, step=10000.0, key='capital_dash')

            current_r_f = st.number_input("无风险利率 (r_f)", value=current_r_f, key='r_f_dash', format="%.3f")

            current_k_factor = st.slider("起始 K (Start)", 0.1, 1.0, current_k_factor, 0.05, key='k_dash',
                                         help="【激进程度】0.5 = 推荐标准 (半凯利)，最大化长期几何增长率。1.0 = 满凯利，仅建议在极度低估时用于回补。")
            current_k_fill = st.number_input("满仓 K (Max at Fill)", min_value=current_k_factor, max_value=2.0, value=current_k_fill, step=0.1, key='k_fill_dash',
                                       help="当股价跌至 V_fill 时，信心增强，K 值线性增加至此值。")
            current_beta = st.slider("估值折扣系数 (beta)", 0.0, 1.0, current_beta, 0.05, key='beta_dash',
                                         help="【止盈速率/信心衰减】0.2 = 推荐。股价接近目标价时，Alpha 保留 80% 权重。1.0 = 到达目标价即清仓。")

            st.subheader("2.2 市场与合约参数")

            # [MODIFIED] 移除了原本在这里的 current_P 输入框 (key='P_dash')
            # 强制使用全局 Anchor 价格，消除冗余
            current_P = current_P_anchor_global

            current_V_target = st.number_input("目标价 V ($)", value=current_V_target, key='V_target_dash', format="%.2f", help="【公允价值】你认为标的最终应值多少钱？影响预期收益(Drift)。")
            current_V_hard = st.number_input("硬底 V_hard ($)", value=current_V_hard, key='V_hard_dash', format="%.2f", help="【止损锚点】极端悲观下绝对不会跌破的价格。建议买入 Strike 接近此价格的期权，物理锁死尾部风险。")
            current_V_fill = st.number_input("计划补仓价 V_fill ($)", value=current_V_fill, key='V_fill_dash', format="%.2f", help="【满仓线】当股价跌至此价格时，总仓位将提升至 1.0K 的理论最大值。")

            if page == "Step 1: 主仓位计算器":
                st.divider()
                current_days_to_expiry = st.number_input("距离到期日 (Days)", value=current_days_to_expiry, key='dte_dash', step=1)
                current_iv_pricing = st.number_input("期权定价 IV", value=current_iv_pricing, key='iv_dash', format="%.4f", help="用于在动态推演中重新计算期权价格。")
                current_opt_price = st.number_input("LEAPS Price ($)", value=current_opt_price, key='opt_price_dash', format="%.2f")
                current_delta = st.number_input("Delta", value=current_delta, key='delta_dash', format="%.4f")
                current_theta = st.number_input("Daily Theta (Abs)", value=current_theta, key='theta_dash', format="%.4f")

            # Save all inputs to session state
            st.session_state.r_f = current_r_f
            st.session_state.k_factor = current_k_factor
            st.session_state.beta = current_beta

            # [MODIFIED] 确保保存到 Session State 的 P 是全局 Anchor
            st.session_state.P = current_P

            st.session_state.V_target = current_V_target
            st.session_state.V_hard = current_V_hard
            st.session_state.V_fill = current_V_fill
            st.session_state.k_fill = current_k_fill
            st.session_state.total_capital = current_total_capital
            st.session_state.days_to_expiry = current_days_to_expiry
            st.session_state.iv_pricing = current_iv_pricing
            st.session_state.opt_price = current_opt_price
            st.session_state.delta = current_delta
            st.session_state.theta = current_theta

        elif page == "Step 2: 多标的组合管理":
            st.subheader("2.1 组合约束")
            max_leverage_cap = st.slider("总仓位上限 (C_max)", 0.5, 2.0, st.session_state.get('max_leverage_cap', 1.0), 0.05, key='c_max_slider', help="控制总现金分配不超过 C_max * 100%")
            st.info("数据来源于 Step 1 中点击 '保存到组合' 的记录。")
            st.session_state['max_leverage_cap'] = max_leverage_cap


    # Return current values needed for page routing and calculation
    current_params = {
        'page': page,
        'ticker': ticker,
        'lambda_val': lambda_val,
        'sigma_val': sigma_val,
        'r_f': st.session_state.r_f,
        'k_factor': st.session_state.k_factor,
        'beta': st.session_state.beta,
        'P': st.session_state.P,
        'V_target': st.session_state.V_target,
        'V_hard': st.session_state.V_hard,
        'V_fill': st.session_state.V_fill,
        'iv_pricing': st.session_state.iv_pricing,
        'opt_price': st.session_state.opt_price,
        'delta': st.session_state.delta,
        'theta': st.session_state.theta,
        'window_days': st.session_state.window_days,
        'days_to_expiry': st.session_state.days_to_expiry,
        'k_fill': st.session_state.k_fill,
        'total_capital': st.session_state.total_capital,
        'P_anchor_global': st.session_state.P_anchor_global
    }

    return current_params