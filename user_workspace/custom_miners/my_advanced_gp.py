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


HEAVY_WINDOW_OPERATORS = {"ts_sampen_20", "ts_hurst_60", "ts_frac_diff_05", "ts_zscore_20", "ts_rank_20"}

@MinerRegistry.register("AdvancedSampleGP")
class AdvancedSampleGPMiner(BaseFactorMiner):
    """
    基于 MyCustomGP 修改的高级版本，专门用于挖掘包含新数学特征的算子。
    已加入方案 B 防嵌套类型约束：禁止长窗口时序算子 (如 ts_sampen_20, ts_hurst_60) 自相嵌套。
    """
    def initialize_search_space(self) -> None:
        logger.info("Initializing AdvancedSampleGP search space...")
        self.operators = configured_operator_names(self.config, DEFAULT_ADVANCED_GP_OPERATORS)
        self.operator_specs = resolve_operator_specs(self.operators)
        self.terminals = self.config.get("data_feeds", {}).get("required_streams", ["close", "volume", "high", "low"])
        self.population_size = self.config.get("population_size", 10)
        self.max_depth = 4 # 允许深度为 4，但禁止长窗口算子嵌套
        # 允许选择的非重型算子集合
        self.light_operators = [op for op in self.operators if op not in HEAVY_WINDOW_OPERATORS]
        if not self.light_operators:
            self.light_operators = ["add", "sub", "mul", "div", "ts_mean"]
        
    def _generate_random_tree(self, current_depth: int, allow_heavy_op: bool = True) -> Any:
        if current_depth >= self.max_depth or random.random() < 0.25:
            return random.choice(self.terminals)
        
        # 如果子树中已存在重型算子，则禁止再次选择重型算子
        candidate_ops = self.operators if allow_heavy_op else self.light_operators
        op = random.choice(candidate_ops)
        
        # 如果当前选择了重型算子，后续子树禁止再次选择重型算子
        next_allow_heavy = False if op in HEAVY_WINDOW_OPERATORS else allow_heavy_op

        node = {
            "op": op,
            "left": self._generate_random_tree(current_depth + 1, allow_heavy_op=next_allow_heavy),
        }
        if self.operator_specs[op]["arity"] == 2:
            node["right"] = self._generate_random_tree(current_depth + 1, allow_heavy_op=next_allow_heavy)
        return node

    def _sanitize_tree(self, ast: Any, inside_heavy_op: bool = False) -> Any:
        """净化树结构：如在变异中意外产生了重型算子嵌套，自动裁剪降级为普通终结符或轻量算子"""
        if not isinstance(ast, dict) or "op" not in ast:
            return ast

        op = ast["op"]
        if op in HEAVY_WINDOW_OPERATORS:
            if inside_heavy_op:
                # 触发防嵌套约束：将重型算子降级为终结符或基础二元运算
                return random.choice(self.terminals)
            current_inside = True
        else:
            current_inside = inside_heavy_op

        sanitized_left = self._sanitize_tree(ast.get("left"), inside_heavy_op=current_inside)
        sanitized_right = self._sanitize_tree(ast.get("right"), inside_heavy_op=current_inside) if ast.get("right") is not None else None

        node = {"op": op, "left": sanitized_left}
        if sanitized_right is not None:
            node["right"] = sanitized_right
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
            return self._sanitize_tree(mutated)
        return ast
        
    def generate_candidates(self) -> List[AdvancedGPExpression]:
        logger.info("AdvancedSampleGP: Generating candidates (with anti-nesting constraints)...")
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
        for expr in candidates:
            result = feedback.for_candidate(expr)
            if result and result.succeeded:
                score = result.metrics.get("fitness_score", 0)
                scored.append((score, expr))
                
        scored.sort(key=lambda x: x[0], reverse=True)
        self.state.population = [expr for score, expr in scored[:self.population_size]]
        
        best_ast = self.state.population[0].get_source()
        logger.info(f"🚀 [AdvancedSampleGP] Best AST this generation (Score: {scored[0][0]:.4f}): {best_ast}")
