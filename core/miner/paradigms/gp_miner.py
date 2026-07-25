import logging
from typing import List, Dict, Any
import random

from core.miner.paradigms.base import BaseFactorMiner
from core.miner.expressions import FactorExpression, FactorExpressionAST
from core.miner.entities import EvaluationFeedback
from core.miner.operator_runtime import configured_operator_names, resolve_operator_specs

logger = logging.getLogger(__name__)

class GPFactorMiner(BaseFactorMiner):
    """
    遗传规划(GP)因子挖掘器
    """
    def initialize_search_space(self) -> None:
        logger.info("Initializing GP search space...")
        # 允许内置算子与动态注册的用户算子共同构成搜索空间。
        self.allowed_operators = configured_operator_names(self.config)
        self.operator_specs = resolve_operator_specs(self.allowed_operators)
        self.population_size = self.config.get("population_size", 20)
        
    def generate_candidates(self) -> List[FactorExpression]:
        logger.info("GP: Generating candidates (ASTs)...")
        candidates = []
        
        if not self.state.population:
            # 初始随机种群
            for _ in range(self.population_size):
                op = random.choice(self.allowed_operators)
                ast = {"op": op, "left": "close"}
                if self.operator_specs[op]["arity"] == 2:
                    ast["right"] = "volume"
                candidates.append(FactorExpressionAST(ast_dict=ast))
        else:
            # 交叉变异产生新一代
            # 伪代码：交叉
            for _ in range(self.population_size):
                p1 = random.choice(self.state.population)
                op = random.choice(self.allowed_operators)
                ast = {"op": op, "left": p1.get_source()}
                if self.operator_specs[op]["arity"] == 2:
                    ast["right"] = "close"
                candidates.append(FactorExpressionAST(ast_dict=ast, parent_ids=p1.get_lineage_parents()))
                
        return candidates

    def evaluate_candidates(self, candidates: List[FactorExpression]) -> EvaluationFeedback:
        if self.evaluator:
            return self.evaluator.evaluate(candidates)
        return EvaluationFeedback()
        
    def update_model(self, candidates: List[FactorExpression], feedback: EvaluationFeedback) -> None:
        logger.info("GP: Updating population based on fitness...")
        
        # 将候选因子和得分打包
        scored = []
        for expr in candidates:
            result = feedback.for_candidate(expr)
            if result and result.succeeded:
                score = result.metrics.get("fitness_score", 0)
                scored.append((score, expr))
                
        # 按得分排序，保留最优秀的 top N
        scored.sort(key=lambda x: x[0], reverse=True)
        top_n = self.config.get("survival_count", self.population_size // 2)
        
        self.state.population = [expr for score, expr in scored[:top_n]]
