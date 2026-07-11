import logging
import re
from typing import List, Dict, Any

from core.miner.paradigms.base import BaseFactorMiner
from core.miner.expressions import FactorExpression, FactorExpressionCode
from core.miner.entities import EvaluationFeedback
from core.evaluation.evaluator import RestrictedSandbox
from core.miner.paradigms.llm_api_manager import LLMAPIManager

logger = logging.getLogger(__name__)

class LLMFactorMiner(BaseFactorMiner):
    """
    大模型(LLM)因子挖掘器
    """
    def initialize_search_space(self) -> None:
        logger.info("Initializing LLM search space (prompts)...")
        self.sandbox = RestrictedSandbox()
        self.system_prompt = "You are a top quantitative researcher. Generate predictive factor formulas using pandas on DataFrame 'df' and assign the final pandas Series to a variable named 'factor'. ONLY output Python code, without any markdown formatting."
        
        # 从用户的 config 里提取 LLM 相关的详细配置
        api_config = self.config.get("llm_api_config", {})
        if not api_config:
            logger.warning("No 'llm_api_config' found in user configuration. LLM Miner might fail if no mock is provided.")
            
        try:
            self.api_manager = LLMAPIManager(api_config)
        except Exception as e:
            logger.error(f"Failed to initialize LLMAPIManager: {e}")
            self.api_manager = None
            
        self.batch_size = self.config.get("batch_size", 5)
        
    def generate_candidates(self) -> List[FactorExpression]:
        logger.info("LLM: Generating candidates via LLMAPIManager...")
        candidates = []
        
        # 组装 Prompt
        prompt = self.system_prompt + "\n" + self.state.get_llm_context_prompt()
        
        # 构建批次请求
        prompts = [prompt] * self.batch_size
        
        if self.api_manager:
            # 真实请求大模型
            raw_responses = self.api_manager.batch_generate(prompts)
        else:
            # Mock 模式（如果未配置 aiohttp 或 keys）
            logger.warning("Using Mock LLM responses because LLMAPIManager is not initialized.")
            raw_responses = [
                "factor = df['close'].pct_change(1)",
                "factor = df['volume'] * df['close']"
            ] * max(1, self.batch_size // 2)

        for resp in raw_responses:
            if not resp:
                continue
                
            # 简单清洗一下 markdown 代码块
            code_str = re.sub(r'```python|```', '', resp).strip()
            
            expr = FactorExpressionCode(
                code_str=code_str, 
                sandbox=self.sandbox, 
                parent_ids=["llm_batch_generate"]
            )
            candidates.append(expr)
            
        return candidates

    def evaluate_candidates(self, candidates: List[FactorExpression]) -> EvaluationFeedback:
        if self.evaluator:
            return self.evaluator.evaluate(candidates)
        return EvaluationFeedback()
        
    def update_model(self, candidates: List[FactorExpression], feedback: EvaluationFeedback) -> None:
        logger.info("LLM: Updating reflection history in MinerState...")
        
        for idx, expr in enumerate(candidates):
            is_error = False
            for err in feedback.execution_status:
                if err["expr"] == expr:
                    self.state.failed_reflections.append({
                        "code": expr.get_source(),
                        "error": err["traceback"]
                    })
                    is_error = True
                    break
                    
            if not is_error and idx < len(feedback.metrics):
                metric = feedback.metrics[idx]
                if metric.get("fitness_score", 0) > 0:
                    self.state.successful_reflections.append({
                        "code": expr.get_source(),
                        "metrics": metric
                    })
