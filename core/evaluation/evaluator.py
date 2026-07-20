import concurrent.futures
import traceback
import builtins
import logging
from typing import List, Dict, Any

from core.miner.expressions import FactorExpression
from core.miner.entities import EvaluationFeedback
from core.miner.registry import EvaluatorRegistry

logger = logging.getLogger(__name__)

class SecurityError(Exception):
    pass

class RestrictedSandbox:
    """安全的 Python 因子代码执行沙盒"""
    def __init__(self):
        # 严格限制能调用的内置函数
        self.safe_globals = {
            '__builtins__': {
                'abs': builtins.abs,
                'min': builtins.min,
                'max': builtins.max,
                'len': builtins.len,
                'range': builtins.range,
                'list': builtins.list,
                'dict': builtins.dict,
            }
        }
        # 注入标准量化计算库
        import numpy as np
        import pandas as pd
        self.safe_globals['np'] = np
        self.safe_globals['pd'] = pd

    def execute_factor_code(self, code_str: str, data: Any) -> Any:
        # 创建局部变量空间，传入行情数据
        local_vars = {'df': data}
        
        # 检查代码中是否包含恶意关键字
        forbidden = ['import', 'eval', 'exec', 'open', 'getattr', '__']
        for word in forbidden:
            if word in code_str:
                raise SecurityError(f"Detected unsafe keyword in code: {word}")
            
        # 在受限的环境中执行
        exec(code_str, self.safe_globals, local_vars)
        
        # 约定：LLM 生成的代码最终必须将结果赋予变量 `factor`
        if 'factor' not in local_vars:
            raise ValueError("The generated code must assign the result to the variable 'factor'")
            
        return local_vars['factor']

class ParallelEvaluator:
    """
    V4 多范式并行执行网关
    负责安全地执行 FactorExpression，并计算 Fitness，返回统一的 EvaluationFeedback
    """
    def __init__(self, data_client: Any, config: Dict):
        self.data_client = data_client
        self.config = config

    @staticmethod
    def _calculate_built_in_metrics(factor_values: Any, returns: Any) -> Dict:
        """计算基础评估指标: IC, RankIC, Turnover"""
        import pandas as pd
        import warnings
        
        metrics = {"IC": 0.0, "RankIC": 0.0, "Turnover": 0.0}
        
        if factor_values is None or returns is None:
            return metrics
            
        # 兼容性检查
        if not isinstance(factor_values, pd.Series):
            try:
                factor_values = pd.Series(factor_values)
            except:
                pass
        if not isinstance(returns, pd.Series):
            try:
                returns = pd.Series(returns)
            except:
                pass
                
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=Warning)
            if isinstance(factor_values, pd.DataFrame) and isinstance(returns, pd.DataFrame):
                # Cross-Asset Mode
                if not factor_values.isna().all().all() and not returns.isna().all().all():
                    # Cross-sectional IC: corr along assets (axis=1), then mean over time
                    ic = factor_values.corrwith(returns, axis=1, method='pearson').mean()
                    rank_ic = factor_values.corrwith(returns, axis=1, method='spearman').mean()
                    
                    if pd.notna(ic):
                        metrics["IC"] = ic
                    if pd.notna(rank_ic):
                        metrics["RankIC"] = rank_ic
                        
                # Turnover: mean absolute difference across time, then mean across assets
                turnover = factor_values.diff().abs().mean().mean()
                if pd.notna(turnover):
                    metrics["Turnover"] = turnover
                    
            elif isinstance(factor_values, pd.Series) and isinstance(returns, pd.Series):
                # Sequential Single Mode
                if not factor_values.isna().all() and not returns.isna().all():
                    ic = factor_values.corr(returns, method='pearson', min_periods=30)
                    rank_ic = factor_values.corr(returns, method='spearman', min_periods=30)
                    
                    if pd.notna(ic):
                        metrics["IC"] = ic
                    if pd.notna(rank_ic):
                        metrics["RankIC"] = rank_ic
                        
                turnover = factor_values.diff().abs().mean()
                if pd.notna(turnover):
                    metrics["Turnover"] = turnover
                    
        return metrics

    def evaluate(self, candidates: List[FactorExpression]) -> EvaluationFeedback:
        feedback = EvaluationFeedback()
        
        # 提取用户配置的自定义打分钩子
        fitness_hook_name = self.config.get("fitness", {}).get("hook")
        fitness_func = None
        if fitness_hook_name:
            fitness_func = EvaluatorRegistry._registry.get(fitness_hook_name)

        def safe_compute(expr: FactorExpression) -> Dict:
            try:
                data = self.data_client.get_data()
                factor_values = expr.compute(data)
                
                # 如果是深度学习网络直接输出的 tensor (带有梯度图)
                if hasattr(factor_values, 'requires_grad') and factor_values.requires_grad:
                    return {"status": "success", "raw_outputs": factor_values, "expr": expr}

                # 标量评价
                returns = self.data_client.get_returns()
                
                # 强制计算基础指标
                base_metrics = self._calculate_built_in_metrics(factor_values, returns)
                
                # 用户自定义打分或降级默认打分
                if fitness_func:
                    # 传入基础指标，返回的是用户定制的字典或标量
                    custom_score = fitness_func(factor_values, returns, base_metrics)
                    # 组合两者
                    if isinstance(custom_score, dict):
                        metrics = {**base_metrics, **custom_score}
                    else:
                        metrics = {**base_metrics, "fitness_score": float(custom_score)}
                else:
                    # 默认与 Inspector 对齐：使用 RankIC (Spearman) 乘以覆盖率惩罚项
                    rank_ic = base_metrics.get("RankIC", 0.0)
                    if hasattr(factor_values, 'dropna'):
                        valid_count = int(factor_values.dropna().count().sum()) if hasattr(factor_values.dropna().count(), 'sum') else int(factor_values.dropna().count())
                        total_count = int(factor_values.size)
                        coverage = valid_count / total_count if total_count > 0 else 0.0
                    else:
                        coverage = 1.0
                    
                    # 覆盖率小于 20% 时进行惩罚
                    penalty = (coverage / 0.20) ** 2 if coverage < 0.20 else 1.0
                    fitness_score = abs(rank_ic) * 100.0 * penalty
                    metrics = {**base_metrics, "coverage": coverage, "fitness_score": float(fitness_score)}
                
                expr.metrics = metrics
                return {"status": "success", "metrics": metrics, "expr": expr}
            except Exception as e:
                # 记录详细报错供 LLM 反思或排查
                return {"status": "error", "traceback": traceback.format_exc(), "expr": expr}

        # 简单使用线程池并行求值，生产环境可替换为 Ray 集群
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(safe_compute, candidates))
            
        for res in results:
            if res["status"] == "error":
                feedback.execution_status.append(res)
            else:
                if "raw_outputs" in res:
                    feedback.raw_outputs = res["raw_outputs"] # 粗略演示：取最后一个或者集成
                else:
                    feedback.metrics.append(res["metrics"])
                    
        return feedback