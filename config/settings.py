"""
FactorMiner 主配置文件
"""

import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 数据目录
DATA_DIR = PROJECT_ROOT / "data"

# 结果目录 - 已迁移到 factorlib/mining_history/
# RESULTS_DIR = PROJECT_ROOT / "results"  # 已弃用
# REPORTS_DIR = RESULTS_DIR / "reports"   # 已弃用
# MODELS_DIR = RESULTS_DIR / "models"     # 已弃用

# 因子库目录 - V3 扁平化结构
FACTORLIB_DIR = PROJECT_ROOT / "factorlib"
FACTOR_DEFINITIONS_DIR = FACTORLIB_DIR / "definitions"
FACTOR_EVALUATIONS_DIR = FACTORLIB_DIR / "evaluations"
FACTOR_TEMP_DIR = FACTORLIB_DIR / "temp"

# 向后兼容别名（已弃用）
EVALUATIONS_DIR = FACTOR_EVALUATIONS_DIR

# 创建必要的目录
for dir_path in [DATA_DIR, FACTORLIB_DIR, FACTOR_DEFINITIONS_DIR, 
                 FACTOR_EVALUATIONS_DIR, FACTOR_TEMP_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# 数据源配置
DATA_SOURCES = {
    'binance': {
        'base_url': 'https://api.binance.com',
        'data_dir': DATA_DIR / "binance",
        'supported_pairs': ['BTC_USDT', 'ETH_USDT', 'BNB_USDT', 'SOL_USDT'],
        'supported_timeframes': ['1m', '5m', '15m', '1h', '4h', '1d']
    }
}

# 因子配置
FACTOR_CONFIG = {
    'default_windows': [5, 10, 20, 50, 100, 200],
    'default_lags': [1, 2, 3, 5, 8, 13, 21],
    'ml_models': ['random_forest', 'gradient_boosting', 'ridge', 'lasso'],
    'evaluation_metrics': ['ic', 'ir', 'win_rate', 'effectiveness_score']
}

# 评估配置
EVALUATION_CONFIG = {
    'min_data_points': 100,
    'ic_window': 20,
    'rolling_window': 60,
    'significance_level': 0.05
}

# 日志配置
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': PROJECT_ROOT / "logs" / "factor_miner.log"
}

# 创建日志目录
LOGGING_CONFIG['file'].parent.mkdir(exist_ok=True) 