import logging
import random
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple

from core.miner.paradigms.base import BaseFactorMiner
from core.miner.registry import MinerRegistry
from core.miner.expressions import FactorExpressionAST
from core.miner.entities import EvaluationFeedback
from core.miner.operator_runtime import configured_operator_names, resolve_operator_specs

logger = logging.getLogger(__name__)

class MyRLExpression(FactorExpressionAST):
    """
    可执行的 RL AST 表达式。
    携带了生成该表达式时的 '轨迹(Trajectory)'，供反向传播更新概率使用。
    """
    def __init__(self, ast_dict: Dict, trajectory: List[Tuple[str, str]]):
        super().__init__(ast_dict)
        self.trajectory = trajectory # 例如: [("op", "add"), ("term", "close")]
        
    def compute(self, data: pd.DataFrame) -> pd.Series:
        return super().compute(data)


@MinerRegistry.register("MyCustomRL")
class MyCustomRLMiner(BaseFactorMiner):
    """
    一个无依赖的 Policy Gradient 强化学习挖掘器。
    通过调整算子和数据的概率权重字典，展现 RL 的学习过程。
    """
    def initialize_search_space(self) -> None:
        logger.info("Initializing MyCustomRL search space (Policy Network)...")
        
        self.operators = configured_operator_names(self.config)
        self.operator_specs = resolve_operator_specs(self.operators)
        self.terminals = self.config.get("data_feeds", {}).get("required_streams", ["close", "volume"])
        
        self.rl_config = self.config.get("rl_config", {})
        self.learning_rate = self.rl_config.get("learning_rate", 0.5)
        self.max_depth = self.rl_config.get("max_depth", 3)
        self.batch_size = self.config.get("population_size", 20)
        self.top_k = self.config.get("top_k_factors", 5)
        
        # 初始策略网络 (Policy): 所有动作的权重均为 1.0 (等概率)
        self.op_weights = {op: 1.0 for op in self.operators}
        self.term_weights = {term: 1.0 for term in self.terminals}
        
        logger.info(f"Initial OP Weights: {self.op_weights}")
        logger.info(f"Initial Term Weights: {self.term_weights}")

    def _sample_action(self, weights_dict: Dict[str, float]) -> str:
        """根据权重字典进行 Softmax 采样"""
        actions = list(weights_dict.keys())
        # 简单归一化
        total = sum(weights_dict.values())
        if total <= 0:
            probs = [1.0 / len(actions)] * len(actions)
        else:
            probs = [weights_dict[a] / total for a in actions]
            
        return np.random.choice(actions, p=probs)

    def _generate_tree_with_trajectory(self, current_depth: int, trajectory: List[Tuple[str, str]]) -> Any:
        # 如果达到最大深度，必须选择 Terminal
        if current_depth >= self.max_depth or random.random() < 0.3:
            term = self._sample_action(self.term_weights)
            trajectory.append(("term", term))
            return term
        
        # 否则选择 Operator
        op = self._sample_action(self.op_weights)
        trajectory.append(("op", op))
        
        node = {
            "op": op,
            "left": self._generate_tree_with_trajectory(current_depth + 1, trajectory),
        }
        if self.operator_specs[op]["arity"] == 2:
            node["right"] = self._generate_tree_with_trajectory(current_depth + 1, trajectory)
        return node
        
    def generate_candidates(self) -> List[MyRLExpression]:
        candidates = []
        for _ in range(self.batch_size):
            trajectory = []
            ast = self._generate_tree_with_trajectory(current_depth=1, trajectory=trajectory)
            candidates.append(MyRLExpression(ast_dict=ast, trajectory=trajectory))
            
        return candidates

    def evaluate_candidates(self, candidates: List[MyRLExpression]) -> EvaluationFeedback:
        if self.evaluator:
            return self.evaluator.evaluate(candidates)
        return EvaluationFeedback()
        
    def update_model(self, candidates: List[MyRLExpression], feedback: EvaluationFeedback) -> None:
        """
        Policy Gradient (REINFORCE) 更新策略
        """
        rewards = [
            candidate.metrics.get("fitness_score", 0.0)
            if getattr(candidate, "metrics", None) else 0.0
            for candidate in candidates
        ]

        evaluated_candidates = [
            candidate for candidate in candidates if getattr(candidate, "metrics", None)
        ]
        if not evaluated_candidates:
            logger.warning("MyCustomRL: No candidates were evaluated successfully this generation.")
            return
                
        # 计算 Baseline (平均奖励)，用于减少方差
        baseline = np.mean(rewards) if len(rewards) > 0 else 0
        
        # 策略更新梯度上升
        for cand, reward in zip(candidates, rewards):
            advantage = reward - baseline
            
            # 如果 advantage > 0，说明这个 action 序列比平均好，增加其权重
            # 如果 advantage < 0，说明比较差，减少权重（但不低于0.01）
            for act_type, act_name in cand.trajectory:
                if act_type == "op":
                    self.op_weights[act_name] += self.learning_rate * advantage
                    self.op_weights[act_name] = max(0.01, self.op_weights[act_name])
                elif act_type == "term":
                    self.term_weights[act_name] += self.learning_rate * advantage
                    self.term_weights[act_name] = max(0.01, self.term_weights[act_name])
                    
        # 打印学习后的权重，观察 Agent 的偏好演进
        logger.info(f"Learned OP Weights: { {k: round(v, 2) for k, v in self.op_weights.items()} }")
        logger.info(f"Learned Term Weights: { {k: round(v, 2) for k, v in self.term_weights.items()} }")
        
        # 保留每轮最佳候选。BaseFactorMiner 会从 state.population 返回这些结果，
        # 供 Director 落盘、进度回调和 Web UI 使用。
        best_candidate = max(
            evaluated_candidates,
            key=lambda candidate: candidate.metrics.get("fitness_score", float("-inf")),
        )
        best_ast = best_candidate.get_source()
        logger.info(
            "🏆 Best AST this generation (Reward: %s): %s",
            best_candidate.metrics.get("fitness_score", 0.0),
            best_ast,
        )

        archived = {candidate.logic_hash: candidate for candidate in self.state.population}
        archived[best_candidate.logic_hash] = best_candidate
        self.state.population = sorted(
            archived.values(),
            key=lambda candidate: candidate.metrics.get("fitness_score", float("-inf")),
            reverse=True,
        )[:self.top_k]
