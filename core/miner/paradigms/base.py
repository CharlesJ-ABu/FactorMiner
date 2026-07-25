from abc import ABC, abstractmethod
from typing import List, Dict, Any
import logging
import hashlib
import pandas as pd

from core.miner.expressions import FactorExpression
from core.miner.entities import EvaluationFeedback, MinerState

logger = logging.getLogger(__name__)

class DiversityFilter:
    """因子正交性与多样性过滤器"""
    def __init__(self, correlation_threshold: float = 0.7, initial_hashes: set = None):
        self.threshold = correlation_threshold
        self.archive_hashes = initial_hashes or set()

    def filter_redundant(self, candidates: List[FactorExpression], data: Any, preserved_candidates: List[FactorExpression] = None) -> List[FactorExpression]:
        unique_candidates = []
        preserved_set = set(preserved_candidates) if preserved_candidates else set()
        
        for cand in candidates:
            # 1. 优先保留上一代保留下来的精英父代个体，不过滤
            if cand in preserved_set:
                unique_candidates.append(cand)
                continue
                
            # 2. 语法/语义层面的硬去重（通过 Hash 机制）
            source_str = str(cand.get_source())
            expr_hash = hashlib.md5(source_str.encode()).hexdigest()
            if expr_hash in self.archive_hashes:
                continue
                
            self.archive_hashes.add(expr_hash)
            cand.logic_hash = expr_hash
            unique_candidates.append(cand)
            
        return unique_candidates

class BaseFactorMiner(ABC):
    """
    因子挖掘的通用基类范式
    适用于 GP (遗传), RL (强化学习), LLM (大模型), NN (神经网络)
    """
    
    def __init__(self, data: Any, config: Dict):
        self.data = data           
        self.config = config       
        self.state = MinerState()
        # 预留统一的回测评价器接口（由外部 Director 或自身子类初始化时注入）
        self.evaluator = None 
        # 全局名人堂（Hall of Fame）：记录历史上出现过的全量最高分因子
        self.hall_of_fame: List[Dict[str, Any]] = []

    def _update_hall_of_fame(self, candidate: FactorExpression, score: float, max_hof_size: int = 100):
        """记录并维护全局名人堂"""
        source_str = str(candidate.get_source())
        cand_hash = hashlib.md5(source_str.encode()).hexdigest()
        
        # 避免同一因子重复进入名人堂
        for item in self.hall_of_fame:
            if item["hash"] == cand_hash:
                return
                
        self.hall_of_fame.append({
            "score": score,
            "hash": cand_hash,
            "candidate": candidate
        })
        self.hall_of_fame.sort(key=lambda x: x["score"], reverse=True)
        if len(self.hall_of_fame) > max_hof_size:
            self.hall_of_fame = self.hall_of_fame[:max_hof_size]
        
    @abstractmethod
    def initialize_search_space(self) -> None:
        """
        1. 初始化搜索空间 / 环境
        """
        pass

    @abstractmethod
    def generate_candidates(self) -> List[FactorExpression]:
        """
        2. 生成候选因子 (Propose)
        @return: 严格返回 FactorExpression 对象的列表
        """
        pass

    @abstractmethod
    def evaluate_candidates(self, candidates: List[FactorExpression]) -> EvaluationFeedback:
        """
        3. 评价与打分 (Evaluate / Reward)
        - GP/LLM/RL: 返回传统的指标（IC、夏普等）。
        - NN: 携带计算图 Tensor 原样返回，供 update_model 梯度计算。
        """
        pass

    @abstractmethod
    def update_model(self, candidates: List[FactorExpression], feedback: EvaluationFeedback) -> None:
        """
        4. 反馈与模型更新 (Feedback & Learn)
        - GP: 更新种群 (self.state.population = new_trees)
        - LLM: 更新反思记忆 (self.state.failed_reflections.append(...))
        - NN: 执行反向传播 (loss.backward(); optimizer.step())
        """
        pass

    def mine(self, n_iterations: int, progress_callback=None) -> List[FactorExpression]:
        """
        5. 标准的主循环引擎 (Main Loop) - 集成去重拦截器
        """
        logger.info(f"[{self.__class__.__name__}] Starting mining loop for {n_iterations} iterations.")
        self.initialize_search_space()
        
        initial_hashes = set()
        if hasattr(self, 'storage_client') and self.storage_client:
            try:
                initial_hashes = self.storage_client.get_all_logic_hashes()
                logger.info(f"Loaded {len(initial_hashes)} global deduplication hashes from storage.")
            except Exception as e:
                logger.warning(f"Failed to load global hashes: {e}")
                
        div_filter = DiversityFilter(self.config.get('max_corr', 0.7), initial_hashes=initial_hashes)

        
        for epoch in range(n_iterations):
            self.current_epoch = epoch
            # 1. 提案
            raw_candidates = self.generate_candidates()
            
            # 1.5 拦截过滤：不过滤上一代保留的种群精英，只去重全新生成的表达式
            preserved = getattr(self.state, "population", None)
            candidates = div_filter.filter_redundant(raw_candidates, self.data, preserved_candidates=preserved)
            if not candidates:
                logger.warning(f"Epoch {epoch}: All candidates filtered out as redundant.")
                if progress_callback:
                    progress_callback(epoch + 1, n_iterations, None)
                continue
            
            # 2. 判卷
            feedback = self.evaluate_candidates(candidates)
            
            # 2.5 搜集本轮高分因子放入名人堂 (Hall of Fame)
            if feedback:
                for cand in candidates:
                    result = feedback.for_candidate(cand)
                    if result and result.succeeded and result.metrics:
                        metrics = result.metrics
                        score = metrics.get("fitness_score", 0.0)
                        cand.metrics = metrics
                        self._update_hall_of_fame(cand, score)
            
            # 3. 学习：update_model 负责更新内部的 self.state 或模型参数
            self.update_model(candidates, feedback)
            
            self._log_epoch(epoch, candidates, feedback.metrics)
            
            if progress_callback:
                best_factor = self._get_best_factors()[0] if self._get_best_factors() else None
                progress_callback(epoch + 1, n_iterations, best_factor)
            
        return self._get_best_factors()
        
    def _log_epoch(self, epoch: int, candidates: List[FactorExpression], metrics: List[Dict[str, float]]):
        logger.info(f"Epoch {epoch} completed. Processed {len(candidates)} unique candidates.")
        
    def _get_best_factors(self) -> List[FactorExpression]:
        # 优先从全局名人堂中获取全期最高分的因子列表
        if self.hall_of_fame:
            return [item["candidate"] for item in self.hall_of_fame]
        if hasattr(self.state, "population") and self.state.population:
            return self.state.population
        if hasattr(self.state, "replay_buffer") and self.state.replay_buffer:
            return self.state.replay_buffer
        return []
