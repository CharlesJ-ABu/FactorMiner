import concurrent.futures
import traceback
import logging
from typing import List, Dict, Any

from core.miner.expressions import FactorExpression
from core.miner.entities import CandidateEvaluation, EvaluationFeedback
from core.miner.registry import EvaluatorRegistry
from core.evaluation.code_sandbox import (
    FactorOutputError,
    RestrictedSandbox,
    SandboxExecutionError,
    SandboxTimeoutError,
    SecurityError,
)

logger = logging.getLogger(__name__)

class ParallelEvaluator:
    """
    V4 多范式并行执行网关
    负责安全地执行 FactorExpression，并计算 Fitness，返回统一的 EvaluationFeedback
    """
    def __init__(self, data_client: Any, config: Dict):
        self.data_client = data_client
        self.config = config
        configured_workers = self.config.get("evaluation", {}).get("max_workers", 8)
        if isinstance(configured_workers, bool):
            raise ValueError("evaluation.max_workers must be a positive integer")
        try:
            self.max_workers = int(configured_workers)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "evaluation.max_workers must be a positive integer"
            ) from exc
        if not 1 <= self.max_workers <= 64:
            raise ValueError(
                "evaluation.max_workers must be an integer between 1 and 64"
            )

    @staticmethod
    def _calculate_built_in_metrics(factor_values: Any, returns: Any) -> Dict:
        """计算基础评估指标: IC, RankIC, Turnover"""
        import pandas as pd
        import warnings
        
        metrics = {"IC": 0.0, "RankIC": 0.0, "Turnover": 0.0}
        
        if factor_values is None or returns is None:
            return metrics
            
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

    @staticmethod
    def _validate_metric_inputs(factor_values: Any, returns: Any) -> None:
        """Reject shape/index coercion before metrics can silently realign data."""
        import numpy as np
        import pandas as pd

        if isinstance(returns, pd.Series):
            if not isinstance(factor_values, pd.Series):
                raise FactorOutputError(
                    "Factor output must be pandas.Series when returns is pandas.Series"
                )
            if not factor_values.index.equals(returns.index):
                raise FactorOutputError("Factor and returns indexes must match exactly")
            objects = [factor_values, returns]
        elif isinstance(returns, pd.DataFrame):
            if not isinstance(factor_values, pd.DataFrame):
                raise FactorOutputError(
                    "Factor output must be pandas.DataFrame when returns is pandas.DataFrame"
                )
            if (
                not factor_values.index.equals(returns.index)
                or not factor_values.columns.equals(returns.columns)
            ):
                raise FactorOutputError(
                    "Factor and returns indexes and columns must match exactly"
                )
            objects = [factor_values, returns]
        else:
            raise FactorOutputError(
                f"Returns must be pandas.Series or pandas.DataFrame, received "
                f"{type(returns).__name__}"
            )

        for value in objects:
            if value.index.has_duplicates:
                raise FactorOutputError("Metric inputs must not contain duplicate indexes")
            dtypes = [value.dtype] if isinstance(value, pd.Series) else list(value.dtypes)
            if any(not pd.api.types.is_numeric_dtype(dtype) for dtype in dtypes):
                raise FactorOutputError("Metric inputs must contain numeric values only")
            numeric = value.to_numpy(dtype=float, na_value=float("nan"))
            if bool((~np.isnan(numeric) & ~np.isfinite(numeric)).any()):
                raise FactorOutputError("Metric inputs must not contain infinite values")
        if factor_values.notna().to_numpy().sum() == 0:
            raise FactorOutputError("Factor output must contain at least one finite value")

    def evaluate(self, candidates: List[FactorExpression]) -> EvaluationFeedback:
        """Evaluate candidates on the configured mining split."""
        return self.evaluate_on(
            candidates,
            self.data_client.get_data(),
            self.data_client.get_returns(),
        )

    def evaluate_on(
        self,
        candidates: List[FactorExpression],
        data: Any,
        returns: Any,
    ) -> EvaluationFeedback:
        """Evaluate candidates on an explicit split without mutating the data client."""
        feedback = EvaluationFeedback()
        if not candidates:
            return feedback
        
        # 提取用户配置的自定义打分钩子
        fitness_hook_name = self.config.get("fitness", {}).get("hook")
        fitness_func = None
        if fitness_hook_name:
            fitness_func = EvaluatorRegistry._registry.get(fitness_hook_name)

        def safe_compute(expr: FactorExpression) -> CandidateEvaluation:
            try:
                factor_values = expr.compute(data)
                
                # 如果是深度学习网络直接输出的 tensor (带有梯度图)
                if hasattr(factor_values, 'requires_grad') and factor_values.requires_grad:
                    return CandidateEvaluation(
                        candidate=expr,
                        status="success",
                        raw_outputs=factor_values,
                    )

                # 标量评价
                self._validate_metric_inputs(factor_values, returns)
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
                return CandidateEvaluation(
                    candidate=expr,
                    status="success",
                    metrics=metrics,
                )
            except Exception as e:
                # 记录详细报错供 LLM 反思或排查
                return CandidateEvaluation(
                    candidate=expr,
                    status="error",
                    error_type=type(e).__name__,
                    error_message=str(e),
                    traceback=traceback.format_exc(),
                )

        worker_count = min(self.max_workers, len(candidates))
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="factor-evaluator",
        ) as executor:
            results = list(executor.map(safe_compute, candidates))
            
        feedback.results.extend(results)
        return feedback
