import pandas as pd
import numpy as np
from core.miner.registry import EvaluatorRegistry

@EvaluatorRegistry.register_fitness_hook("my_bear_market_hunter")
def custom_fitness_evaluator(factor_values: pd.Series, returns: pd.Series, base_metrics: dict) -> float:
    """
    熊市猎手评价挂钩：
    直接利用引擎算好的基础指标 (如 IC) 进行定制化得分计算。
    """
    # 提取引擎自动算好的基础 IC
    ic = base_metrics.get("IC", 0.0)
            
    # 计算适应度得分 (纯 IC 放大)
    fitness_score = ic * 100
    
    return fitness_score
