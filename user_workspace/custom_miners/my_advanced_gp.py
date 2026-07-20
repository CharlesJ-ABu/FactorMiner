import logging
import random
import pandas as pd
from typing import List, Dict, Any

from core.miner.paradigms.base import BaseFactorMiner
from core.miner.registry import MinerRegistry
from core.miner.expressions import FactorExpressionAST
from core.miner.entities import EvaluationFeedback
from core.miner.operator_runtime import configured_operator_names, resolve_operator_specs

logger = logging.getLogger(__name__)

DEFAULT_ADVANCED_GP_OPERATORS = [
    "add",
    "sub",
    "mul",
    "div",
    "ts_frac_diff_05",
    "ts_hurst_60",
    "ts_sampen_20",
    "ts_mean",
    "ts_std",
    "ts_zscore_20",
    "ts_rank_20",
]

class AdvancedGPExpression(FactorExpressionAST):
    """
    可执行的 GP AST 表达式 (Advanced)。
    """
    def __init__(self, ast_dict: Dict, parent_ids: List[str] = None):
        super().__init__(ast_dict, parent_ids)
        
    def compute(self, data: pd.DataFrame) -> pd.Series:
        return super().compute(data)


@MinerRegistry.register("AdvancedSampleGP")
class AdvancedSampleGPMiner(BaseFactorMiner):
    """
    基于 MyCustomGP 修改的高级版本，专门用于挖掘包含新数学特征的算子。
    """
    def initialize_search_space(self) -> None:
        logger.info("Initializing AdvancedSampleGP search space...")
        self.operators = configured_operator_names(self.config, DEFAULT_ADVANCED_GP_OPERATORS)
        self.operator_specs = resolve_operator_specs(self.operators)
        self.terminals = self.config.get("data_feeds", {}).get("required_streams", ["close", "volume", "high", "low"])
        self.population_size = self.config.get("population_size", 10)
        self.max_depth = 4 # 允许更深一点的树来容纳复杂的数学算子
        
    def _generate_random_tree(self, current_depth: int) -> Any:
        if current_depth >= self.max_depth or random.random() < 0.25:
            return random.choice(self.terminals)
        
        op = random.choice(self.operators)
        node = {
            "op": op,
            "left": self._generate_random_tree(current_depth + 1),
        }
        if self.operator_specs[op]["arity"] == 2:
            node["right"] = self._generate_random_tree(current_depth + 1)
        return node
        
    def _mutate(self, ast: Any) -> Any:
        if random.random() < 0.2: 
            return self._generate_random_tree(current_depth=1)
            
        if isinstance(ast, dict) and "op" in ast:
            mutated = {
                "op": ast["op"],
                "left": self._mutate(ast.get("left")),
            }
            if ast.get("right") is not None:
                mutated["right"] = self._mutate(ast["right"])
            return mutated
        return ast
        
    def generate_candidates(self) -> List[AdvancedGPExpression]:
        logger.info("AdvancedSampleGP: Generating candidates...")
        candidates = []
        
        if not self.state.population:
            for _ in range(self.population_size):
                ast = self._generate_random_tree(current_depth=1)
                candidates.append(AdvancedGPExpression(ast_dict=ast))
        else:
            for p in self.state.population:
                candidates.append(p)
                mutated_ast = self._mutate(p.ast_dict)
                parent_id = p.get_source()
                candidates.append(AdvancedGPExpression(ast_dict=mutated_ast, parent_ids=[str(parent_id)]))
                
            while len(candidates) < self.population_size * 2:
                candidates.append(AdvancedGPExpression(self._generate_random_tree(current_depth=1)))
                
        return candidates[:self.population_size * 2]

    def evaluate_candidates(self, candidates: List[AdvancedGPExpression]) -> EvaluationFeedback:
        if self.evaluator:
            return self.evaluator.evaluate(candidates)
        return EvaluationFeedback()
        
    def update_model(self, candidates: List[AdvancedGPExpression], feedback: EvaluationFeedback) -> None:
        logger.info("AdvancedSampleGP: Updating population...")
        
        scored = []
        for idx, expr in enumerate(candidates):
            if idx < len(feedback.metrics):
                score = feedback.metrics[idx].get("fitness_score", 0)
                scored.append((score, expr))
                
        scored.sort(key=lambda x: x[0], reverse=True)
        self.state.population = [expr for score, expr in scored[:self.population_size]]
        
        best_ast = self.state.population[0].get_source()
        logger.info(f"🚀 [AdvancedSampleGP] Best AST this generation (Score: {scored[0][0]:.4f}): {best_ast}")
