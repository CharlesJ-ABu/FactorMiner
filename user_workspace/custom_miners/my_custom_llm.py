import logging
import re
import random
import pandas as pd
from typing import List

from core.miner.paradigms.base import BaseFactorMiner
from core.miner.registry import MinerRegistry
from core.miner.expressions import FactorExpressionCode
from core.miner.entities import EvaluationFeedback

# 尝试导入 LLMAPIManager
try:
    from core.miner.paradigms.llm_api_manager import LLMAPIManager
except ImportError:
    LLMAPIManager = None

logger = logging.getLogger(__name__)


@MinerRegistry.register("MyCustomLLM")
class MyCustomLLMMiner(BaseFactorMiner):
    """
    基于大语言模型 (LLM) 和反思机制 (Reflection) 的因子挖掘器。
    它生成 Python 代码片段，并通过 FactorExpressionCode 执行。
    """
    def initialize_search_space(self) -> None:
        logger.info("Initializing MyCustomLLM Search Space (Reflection Memory)...")
        
        # 初始化记忆
        self.reflection_history = "Initial State: No prior knowledge."
        self.population_size = self.config.get("population_size", 3)
        self.terminals = self.config.get("data_feeds", {}).get("required_streams", ["close", "volume"])
        
        # 初始化 API Manager
        api_config = self.config.get("llm_api_config")
        if not api_config or not LLMAPIManager:
            logger.warning("LLMAPIManager is not available or api_config is missing. Will use fallback random strings.")
            self.api_manager = None
        else:
            self.api_manager = LLMAPIManager(api_config)
            
    def _extract_code(self, llm_response: str) -> str:
        """从 Markdown 中提取纯代码块"""
        if not llm_response:
            return ""
        
        match = re.search(r'```python\n(.*?)\n```', llm_response, re.DOTALL)
        if match:
            return match.group(1).strip()
            
        # 如果没有用代码块包裹，直接返回原文本尝试执行
        return llm_response.strip()
        
    def _get_fallback_code(self) -> str:
        """当网络断开或 API Key 错误时使用的回退伪随机代码"""
        ops = ['+', '-', '*', '/']
        op = random.choice(ops)
        t1 = random.choice(self.terminals)
        t2 = random.choice(self.terminals)
        if op == '/':
            return f"factor = df['{t1}'] / (df['{t2}'].replace(0, 1e-9))"
        return f"factor = df['{t1}'] {op} df['{t2}']"

    def generate_candidates(self) -> List[FactorExpressionCode]:
        logger.info(f"MyCustomLLM: Generating {self.population_size} candidates based on reflection...")
        
        prompt = f"""
You are an expert quantitative researcher. Please write a simple Python calculation for a financial alpha factor.
You have access to a pandas DataFrame named `df` with the following columns: {self.terminals}.
Please calculate a factor and assign it to a pandas Series named `factor`.
Your code should be wrapped in a Markdown ```python``` block.

Reflection History from previous generations (learn from it):
{self.reflection_history}
"""
        logger.debug(f"Prompt sent to LLM:\n{prompt}")

        candidates = []
        if self.api_manager:
            prompts = [prompt] * self.population_size
            logger.info("Sending batch requests to LLM API...")
            responses = self.api_manager.batch_generate(prompts)
            
            for resp in responses:
                if resp:
                    code = self._extract_code(resp)
                    candidates.append(FactorExpressionCode(code_str=code))
                else:
                    # API 失败时使用 Fallback
                    logger.warning("LLM API returned None (e.g. invalid key). Using Fallback code.")
                    candidates.append(FactorExpressionCode(code_str=self._get_fallback_code()))
        else:
            for _ in range(self.population_size):
                candidates.append(FactorExpressionCode(code_str=self._get_fallback_code()))
                
        return candidates

    def evaluate_candidates(self, candidates: List[FactorExpressionCode]) -> EvaluationFeedback:
        if self.evaluator:
            return self.evaluator.evaluate(candidates)
        return EvaluationFeedback()

    def update_model(self, candidates: List[FactorExpressionCode], feedback: EvaluationFeedback) -> None:
        """
        更新反思记忆 (Reflection History)
        """
        logger.info("MyCustomLLM: Reflecting on evaluations...")
        
        scored = []
        for idx, expr in enumerate(candidates):
            if idx < len(feedback.metrics):
                score = feedback.metrics[idx].get("fitness_score", 0)
                scored.append((score, expr))
                
        # 按分数排序
        scored.sort(key=lambda x: x[0], reverse=True)
        
        if not scored:
            return
            
        best_score, best_expr = scored[0]
        best_code = best_expr.get_source()
        
        logger.info(f"🏆 Best Code this generation (Score: {best_score}): {best_code}")
        
        # 将本次最好的公式更新进自我反思记忆中
        reflection_note = (
            f"\n[Epoch Note] The best performing code snippet scored {best_score}. "
            f"The code was:\n```python\n{best_code}\n```\n"
            f"Please try to improve upon this logic or combine it with other signals."
        )
        self.reflection_history += reflection_note
