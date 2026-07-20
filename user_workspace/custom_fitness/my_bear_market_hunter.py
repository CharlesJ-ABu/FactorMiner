import pandas as pd
import numpy as np
from core.miner.registry import EvaluatorRegistry

@EvaluatorRegistry.register_fitness_hook("my_bear_market_hunter")
def custom_fitness_evaluator(factor_values: pd.Series, returns: pd.Series, base_metrics: dict) -> float:
    """
    熊市猎手评价挂钩：
    使用 RankIC (Spearman) 结合数据覆盖率惩罚计算真正可靠的 Fitness 得分。
    """
    # 提取引擎自动算好的基础 RankIC (Spearman)
    rank_ic = base_metrics.get("RankIC", 0.0)
    
    # 计算有效数据覆盖率
    if hasattr(factor_values, 'dropna'):
        valid_count = int(factor_values.dropna().count().sum()) if hasattr(factor_values.dropna().count(), 'sum') else int(factor_values.dropna().count())
        total_count = int(factor_values.size)
        coverage = valid_count / total_count if total_count > 0 else 0.0
    else:
        coverage = 1.0
        
    # 覆盖率小于 20% 时进行二次方惩罚
    penalty = (coverage / 0.20) ** 2 if coverage < 0.20 else 1.0
            
    # 计算最终适应度得分
    fitness_score = abs(rank_ic) * 100.0 * penalty
    
    return float(fitness_score)
