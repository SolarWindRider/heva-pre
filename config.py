"""
配置文件 - 使用相对路径

项目根目录: /home/ma-user/work/heva-pre/
"""

import os

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 数据集路径 (相对于项目根目录)
DATA_DIR = os.path.join(PROJECT_ROOT, "../datas/VisuRiddles")
DATA_FILE = os.path.join(DATA_DIR, "VisuRiddles.json")
IMAGE_DIR = DATA_DIR

# 模型路径
MODEL_DIR = os.path.join(PROJECT_ROOT, "../Downloads/Models/Qwen/Qwen3-VL-2B-Instruct")

# 输出目录
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")
TABLES_DIR = os.path.join(PROJECT_ROOT, "tables")

# 默认实验名称 (可自定义)
DEFAULT_EXP_NAME = "exp001"

# 创建目录
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)
