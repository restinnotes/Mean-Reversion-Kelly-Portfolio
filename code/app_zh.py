import streamlit as st
import os
import sys


# ==========================================
# 1. SETUP: Path & Imports (REVISED)
# ==========================================
def get_project_root():
    """Determines the project root (the directory containing the 'code' folder)."""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    else:
        # Assumes this file is in 'code/' relative to the project root
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up one level to the project root
        return os.path.abspath(os.path.join(current_dir, ".."))

project_root = get_project_root()

# === CRITICAL FIX: 解决 'ModuleNotFoundError: No module named 'core'' ===
# project_root 是项目根目录 (e.g., .../ProjectRoot/)，包含 'core' 文件夹。
# 必须将其添加到 sys.path 才能解析 'core.simulation' 等绝对导入。
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# code_base_dir 是当前文件所在目录 (e.g., .../ProjectRoot/code/)
# 确保它也在 sys.path 中，以支持 'ui.X' 等导入。
code_base_dir = os.path.abspath(os.path.dirname(__file__))
if code_base_dir not in sys.path:
    sys.path.insert(0, code_base_dir)

from config import DEFAULT_APP_PARAMS
from ui.plot_utils import configure_chinese_font
from ui.zh.sidebar import render_sidebar
from ui.zh.page_diagnosis import render_page_diagnosis
from ui.zh.page_solver import render_page_solver
from ui.zh.page_dashboard import render_page_dashboard, render_page_multi_asset_normalization
from ui.zh.page_glossary import render_page_glossary

# ==========================================
# 2. MAIN APP ROUTER
# ==========================================

# 1. Page Configuration
st.set_page_config(page_title="统一凯利量化工具", layout="wide", page_icon="📈")

# 2. Initialize Session State Defaults
for key, default_val in DEFAULT_APP_PARAMS.items():
    if key not in st.session_state:
        st.session_state[key] = default_val

# 3. Configure Fonts
if not configure_chinese_font():
    font_path = os.path.join(project_root, "fonts", "SimHei.ttf")
    if os.path.exists(font_path):
        st.warning(f"❌ 字体加载失败。请检查文件是否损坏。")
    else:
        st.warning(f"⚠️ 未找到字体文件：{font_path}。虽然不影响计算，但图表中文可能显示为方框。")


# 4. Render Sidebar and Get Current Parameters
current_params = render_sidebar()

# 5. Page Routing
page = current_params['page']
ticker = current_params['ticker']
lambda_val = current_params['lambda_val']
sigma_val = current_params['sigma_val']
P = current_params['P']
V_target = current_params['V_target']
V_hard = current_params['V_hard']
V_fill = current_params['V_fill']

if page == "Step 0: 市场诊断":
    render_page_diagnosis(
        ticker,
        current_params['window_days'],
        lambda_val,
        sigma_val,
        current_params['P_anchor_global']
    )

elif page == "Step 0.5: 最优期限求解":
    if V_target <= V_hard:
        st.error("错误: 目标价必须高于硬底。")
    elif lambda_val is None or sigma_val is None:
        st.error("请先在侧边栏获取 Lambda/Sigma 统计数据。")
    else:
        render_page_solver(
            P, V_target, V_hard, V_fill,
            lambda_val, sigma_val,
            current_params['iv_pricing'],
            current_params['r_f'],
            ticker,
            current_params['k_factor'],
            current_params['beta']
        )

elif page == "Step 1: 主仓位计算器":
    opt_price = current_params['opt_price']
    delta = current_params['delta']

    if lambda_val is None or sigma_val is None:
        st.error("请先在侧边栏获取 Lambda/Sigma 统计数据。")
    elif opt_price <= 0 or delta <= 0:
        st.warning("请在侧边栏输入有效的期权合约数据。")
    else:
        render_page_dashboard(
            ticker, lambda_val, sigma_val,
            current_params['r_f'], current_params['k_factor'], current_params['beta'],
            P, V_target, V_hard,
            opt_price, delta, current_params['theta'],
            V_fill, current_params['iv_pricing'], current_params['days_to_expiry'],
            current_params['k_fill'], current_params['total_capital']
        )

elif page == "Step 2: 多标的组合管理":
    max_leverage_cap = st.session_state.get('max_leverage_cap', 1.0)
    render_page_multi_asset_normalization(max_leverage_cap)

elif page == "📚 术语与原理速查":
    render_page_glossary()