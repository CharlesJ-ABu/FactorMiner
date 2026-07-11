"""
FactorMiner 主配置文件
"""

import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 数据目录
DATA_DIR = PROJECT_ROOT / "data"

# 创建必要的目录
for dir_path in [DATA_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# 日志配置
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': PROJECT_ROOT / "logs" / "factor_miner.log"
}

# 创建日志目录
LOGGING_CONFIG['file'].parent.mkdir(parents=True, exist_ok=True)