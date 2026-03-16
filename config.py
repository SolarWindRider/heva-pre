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
MODEL_DIR = os.path.join(PROJECT_ROOT, "../Downloads/Models/Qwen/Qwen3-VL-2B-Thinking")

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

# ==================== 显存优化配置 ====================
# 这些选项可以帮助在显存受限时运行 HEVA 计算

# 显存优化模式：减少 attention 层的存储
USE_LAST_LAYER_ONLY = True  # 只使用最后一层 attention (推荐开启，大幅减少显存)

# 分块计算：当序列很长时分块处理
CHUNK_SIZE = 512  # 分块大小，当序列长度超过此值时分块计算

# 关键优化：对attention矩阵的行维度分块大小（解决OOM的核心参数）
# 默认为256，如果仍然OOM可以减小到128或64
ATTENTION_CHUNK_SIZE = 256

# 自定义HEVA计算的alpha值列表 (例如 [0.1, 0.2, 0.3] 计算 heva_10, heva_20, heva_30)
ALPHA_VALUES = [0.1, 0.2, 0.3]

# 是否在计算后清理 GPU 缓存
CLEAR_CACHE_AFTER_COMPUTE = True
