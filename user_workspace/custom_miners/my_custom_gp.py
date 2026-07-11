import logging
import random
import pandas as pd
from typing import List, Dict, Any

from core.miner.paradigms.base import BaseFactorMiner
from core.miner.registry import MinerRegistry
from core.miner.expressions import FactorExpressionAST
from core.miner.entities import EvaluationFeedback

logger = logging.getLogger(__name__)

class MyGPExpression(FactorExpressionAST):
    """
    可执行的 GP AST 表达式。
    """
    def __init__(self, ast_dict: Dict, parent_ids: List[str] = None):
        super().__init__(ast_dict, parent_ids)
        
    def compute(self, data: pd.DataFrame) -> pd.Series:
        return self._eval_node(self.ast_dict, data)
        
    def _eval_node(self, node: Any, data: Any) -> Any:
        # 叶子节点：直接是 DataFrame 的列名（比如 "close", "volume"）
        if isinstance(node, str):
            if isinstance(data, dict):
                # Cross-asset mode: data is a dict of DataFrames
                if node in data:
                    return data[node]
                else:
                    # 获取任意一个 dataframe 的 index 来构造全 0 DataFrame
                    sample_df = next(iter(data.values()))
                    return pd.DataFrame(0, index=sample_df.index, columns=sample_df.columns)
            else:
                # Sequential single mode: data is a DataFrame
                if node in data.columns:
                    return data[node]
                else:
                    return pd.Series(index=data.index, data=0)
                
        # 标量常量
        if isinstance(node, (int, float)):
            if isinstance(data, dict):
                sample_df = next(iter(data.values()))
                return pd.DataFrame(node, index=sample_df.index, columns=sample_df.columns)
            else:
                return pd.Series(index=data.index, data=node)
            
        # 操作节点
        if isinstance(node, dict) and "op" in node:
            op = node["op"]
            left = self._eval_node(node.get("left"), data)
            right = self._eval_node(node.get("right"), data)
            
            if op == "add":
                return left + right
            elif op == "sub":
                return left - right
            elif op == "mul":
                return left * right
            elif op == "div":
                # 简单防零除
                return left / (right.replace(0, 1e-9))
                
        return pd.Series(index=data.index, data=0)


@MinerRegistry.register("MyCustomGP")
class MyCustomGPMiner(BaseFactorMiner):
    """
    一个完整的演示级 GP 挖掘器。
    包含初始化随机种群、子树交叉变异以及精英保留。
    """
    def initialize_search_space(self) -> None:
        logger.info("Initializing MyCustomGP search space...")
        # 从配置读取或者硬编码可用的算子和变量
        self.operators = ["add", "sub", "mul", "div"]
        # 获取输入的数据流作为终端节点 (Terminal set)
        self.terminals = self.config.get("data_feeds", {}).get("required_streams", ["close", "volume"])
        self.population_size = self.config.get("population_size", 10)
        self.max_depth = 3
        
    def _generate_random_tree(self, current_depth: int) -> Any:
        # 如果达到最大深度，或者随机决定生成叶子节点
        if current_depth >= self.max_depth or random.random() < 0.3:
            return random.choice(self.terminals)
        
        # 否则生成操作符节点
        op = random.choice(self.operators)
        return {
            "op": op,
            "left": self._generate_random_tree(current_depth + 1),
            "right": self._generate_random_tree(current_depth + 1)
        }
        
    def _mutate(self, ast: Any) -> Any:
        # 对节点进行变异
        if random.random() < 0.2: # 20% 变异率
            return self._generate_random_tree(current_depth=1)
            
        if isinstance(ast, dict) and "op" in ast:
            # 递归变异左右子树
            return {
                "op": ast["op"],
                "left": self._mutate(ast.get("left")),
                "right": self._mutate(ast.get("right"))
            }
        return ast
        
    def generate_candidates(self) -> List[MyGPExpression]:
        logger.info("MyCustomGP: Generating candidates...")
        candidates = []
        
        if not self.state.population:
            # 第一代：纯随机生成
            logger.info("MyCustomGP: Generating initial random population.")
            for _ in range(self.population_size):
                ast = self._generate_random_tree(current_depth=1)
                candidates.append(MyGPExpression(ast_dict=ast))
        else:
            # 后续代：从精英中变异生成 (为了简化，这里使用 100% 变异策略)
            logger.info("MyCustomGP: Generating new generation via mutation...")
            for p in self.state.population:
                # 1. 保留原本的精英
                candidates.append(p)
                # 2. 从每个精英变异出子代
                mutated_ast = self._mutate(p.ast_dict)
                parent_id = p.get_source() # 在真实环境可以传 metadata ID
                candidates.append(MyGPExpression(ast_dict=mutated_ast, parent_ids=[str(parent_id)]))
                
            # 如果数量不够，补充随机个体
            while len(candidates) < self.population_size * 2:
                candidates.append(MyGPExpression(self._generate_random_tree(current_depth=1)))
                
        # 控制总数，只返回评估 self.population_size * 2 个
        return candidates[:self.population_size * 2]

    def evaluate_candidates(self, candidates: List[MyGPExpression]) -> EvaluationFeedback:
        # 直接委托给系统自带的通用评价器
        if self.evaluator:
            return self.evaluator.evaluate(candidates)
        return EvaluationFeedback()
        
    def update_model(self, candidates: List[MyGPExpression], feedback: EvaluationFeedback) -> None:
        logger.info("MyCustomGP: Updating population based on fitness (Elitism)...")
        
        scored = []
        for idx, expr in enumerate(candidates):
            if idx < len(feedback.metrics):
                score = feedback.metrics[idx].get("fitness_score", 0)
                scored.append((score, expr))
                
        # 按照 fitness_score 降序排列
        scored.sort(key=lambda x: x[0], reverse=True)
        
        # 精英保留：留下种群数量的个体
        self.state.population = [expr for score, expr in scored[:self.population_size]]
        
        best_ast = self.state.population[0].get_source()
        logger.info(f"🏆 Best AST this generation (Score: {scored[0][0]}): {best_ast}")
