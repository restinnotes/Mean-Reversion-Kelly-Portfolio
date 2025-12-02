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
# 修正路径：从 'code/' 向上走一级 '..' 到达项目根目录
project_root = os.path.abspath(os.path.join(current_dir, ".."))

# 导入路径设置
sys.path.append(os.path.join(project_root, "code", "utils"))

try:
    from utils.lambda_tools import calculate_ou_params
except ImportError:
    st.error("无法导入 utils.lambda_tools。请确认 util/ 路径和文件完整性。")
    st.stop()


PE_CSV_DIR_NAME = "pe_csv"

def get_pe_data(ticker):
    """加载 PE 数据文件"""
    # 使用正确的 project_root 构造 CSV 路径
    csv_path = os.path.join(project_root, PE_CSV_DIR_NAME, f"{ticker}_pe.csv")

    if not os.path.exists(csv_path):
        st.error(f"[错误] 找不到 PE 数据文件: {ticker}_pe.csv。")
        st.markdown(f"**预期搜索路径**: `{os.path.join(os.path.basename(project_root), PE_CSV_DIR_NAME)}/`")
        st.markdown(f"请确保您运行 Streamlit 的终端位于项目根目录，且文件结构正确。")
        return None

    try:
        df = pd.read_csv(csv_path, parse_dates=['date'])
        df = df.sort_values('date').reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"[错误] 读取 CSV 文件失败: {e}")
        return None


# Monte Carlo 模拟函数（从 rolling_analysis.py 移植）
def run_simulation(current_pe, target_pe, lambda_annual, sigma_daily, days_to_simulate=252, num_paths=10000):
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

# 概率分析函数（从 rolling_analysis.py 移植）
def analyze_probabilities(paths, target_pe, current_pe):
    days_simulated = paths.shape[0] - 1
    check_points = [21, 42, 63, 126, 189, 252]
    results = []
    is_long = target_pe > current_pe

    for day in check_points:
        if day > days_simulated: continue
        final_values = paths[day]

        if is_long: prob_end = np.mean(final_values >= target_pe)
        else: prob_end = np.mean(final_values <= target_pe)

        path_slice = paths[:day+1, :]
        if is_long: has_hit = np.any(path_slice >= target_pe, axis=0)
        else: has_hit = np.any(path_slice <= target_pe, axis=0)
        prob_touch = np.mean(has_hit)

        approx_cal_days = int(day * (365/252))

        results.append({
            "交易日": day,
            "~日历日": f"{approx_cal_days}d",
            "触摸目标概率": prob_touch,
            "结束时保持概率": prob_end,
            "预期PE": np.mean(final_values)
        })
    return pd.DataFrame(results)


def run_rolling_analysis_gui(ticker, window_days=90):
    df = get_pe_data(ticker)
    if df is None:
        return

    # --- 1. 计算滚动指标 ---
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
            # 忽略计算失败的情况
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
    status_placeholder.text("滚动指标计算完成。")


    if not lambdas_annual:
        st.warning("没有足够的历史数据点来计算滚动指标。")
        return

    # --- 2. 诊断报告 ---
    current_lambda = lambdas_annual[-1]
    current_hl = half_lives[-1]
    current_pe = pe_values[-1]
    current_mean = pe_means[-1]
    current_sigma = sigmas_daily[-1]

    st.subheader("诊断报告与 Monte Carlo 模拟")
    st.markdown("---")
    col_d1, col_d2 = st.columns(2)

    with col_d1:
        st.markdown("**PE 估值状态**")
        st.code(f"当前 PE: {current_pe:.2f}")
        st.code(f"{window_days}日均值: {current_mean:.2f}")

    with col_d2:
        st.markdown("**回归与波动率**")
        st.code(f"年化 Lambda (λ): {current_lambda:.4f}")
        st.code(f"半衰期: {current_hl:.2f} 天")
        st.code(f"日波动率 (σ): {current_sigma:.4f}")

    st.markdown("---")

    # --- 3. Monte Carlo 模拟 ---
    st.markdown("##### Monte Carlo 模拟结果")
    st.caption(f"目标: PE {current_pe:.2f} 修复到均值 PE {current_mean:.2f} | 模拟路径: 10000条")

    paths = run_simulation(current_pe, current_mean, current_lambda, current_sigma)
    df_probs = analyze_probabilities(paths, current_mean, current_pe)

    safe_days = 0
    safe_cal_days = 0
    found_safe_zone = False

    for idx, row in df_probs.iterrows():
        if row['触摸目标概率'] > 0.9:
            safe_days = int(row['交易日'])
            safe_cal_days = int(row['~日历日'].replace('d',''))
            found_safe_zone = True
            break

    df_probs['触摸目标概率'] = df_probs['触摸目标概率'].apply(lambda x: f"{x:.1%}")
    df_probs['结束时保持概率'] = df_probs['结束时保持概率'].apply(lambda x: f"{x:.1%}")
    df_probs['预期PE'] = df_probs['预期PE'].apply(lambda x: f"{x:.2f}")
    st.dataframe(df_probs, hide_index=True)

    if found_safe_zone:
        st.success(f"**[推荐行动计划]**：90% 概率触摸目标所需的最短时间为 **{safe_days} 交易日 (~{safe_cal_days} 日历日)**。")
        st.info(f"选品建议：购买到期日 **大于等于** {safe_cal_days} 日历日的 LEAPS 期权。")
    else:
        st.warning(f"**[警告]**：在 1 年内无法达到 90% 的目标触摸概率。回归缓慢/不确定。建议购买 > 1 年的 LEAPS 或保持现金。")

    st.markdown("---")

    # --- 4. 绘图 ---
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
st.set_page_config(page_title="市场诊断 - 滚动分析", layout="wide", page_icon="📈")
st.title("📈 Step 0: 市场诊断 - 滚动分析 (Rolling Analysis)")

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("参数配置")
    ticker = st.text_input("股票代码 (Ticker)", value="NVDA").upper()
    window_days = st.slider("滚动窗口 (Rolling Window, 交易日)", min_value=30, max_value=252, value=90, step=10)

    if st.button("运行分析并诊断 (Run Analysis)", type="primary"):
        st.session_state['run_analysis'] = True

# --- Main Content ---
if st.session_state.get('run_analysis', False):
    run_rolling_analysis_gui(ticker, window_days)
    st.session_state['run_analysis'] = False