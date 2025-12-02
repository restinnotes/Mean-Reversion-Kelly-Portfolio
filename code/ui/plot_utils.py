# code/ui/plot_utils.py

import os
import sys
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ==========================================
# HELPER: Path
# ==========================================
def get_resource_root():
    """Finds the project root, used to locate fonts/SimHei.ttf."""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    else:
        # Assumes this file is within 'code/ui/' relative to the project root
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up two levels: code/ui/ -> code/ -> project_root/
        return os.path.abspath(os.path.join(current_dir, "..", ".."))

project_root = get_resource_root()


# ==========================================
# Matplotlib Font Configuration (Chinese Support)
# ==========================================
def configure_chinese_font():
    """
    配置中文字体。使用项目内上传的 SimHei.ttf 文件。
    (Extracted from app_unified_zh.py)

    Returns: bool: True if font successfully configured, False otherwise.
    """
    # 1. 定义字体路径
    font_name = "SimHei.ttf"
    font_path = os.path.join(project_root, "fonts", font_name)

    if os.path.exists(font_path):
        try:
            # 2. 注册并加载字体
            fm.fontManager.addfont(font_path)
            prop = fm.FontProperties(fname=font_path)
            custom_font_name = prop.get_name()

            # 3. 应用配置
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = [custom_font_name, 'DejaVu Sans', 'Arial']
            plt.rcParams['axes.unicode_minus'] = False
            return True
        except Exception:
            return False
    else:
        return False