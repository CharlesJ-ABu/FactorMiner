"""
FactorMiner - 量化因子挖掘平台
专业的量化因子挖掘、评估和优化平台
"""

__version__ = "1.0.0"
__author__ = "FactorMiner Team"
__description__ = "专业的量化因子挖掘、评估和优化平台"

from .core import DataLoader, FactorBuilder, FactorEvaluator, FactorOptimizer
from .api import FactorAPI
from .api.factor_mining_api import FactorMiningAPI
from .utils import *

__all__ = [
    'DataLoader',
    'FactorBuilder', 
    'FactorEvaluator',
    'FactorOptimizer',
    'FactorAPI',
    'FactorMiningAPI',
    '__version__',
    '__author__',
    '__description__'
] 