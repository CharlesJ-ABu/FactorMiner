import logging
import re
import hashlib
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
        sandbox_config = self.config.get("llm_sandbox", {})
        self.sandbox = RestrictedSandbox(
            timeout_seconds=sandbox_config.get("timeout_seconds", 5.0),
            cpu_seconds=sandbox_config.get("cpu_seconds", 3),
            memory_mb=sandbox_config.get("memory_mb", 1024),
        )
        self.is_cross_asset = (
            self.config.get("data_feeds", {}).get("mining_mode")
            == "cross_asset"
        )
        if self.is_cross_asset:
            input_contract = (
                "The variable 'df' is a mapping from feature names to aligned "
                "pandas DataFrames (time index × asset columns). Assign an "
                "aligned pandas DataFrame to 'factor'."
            )
        else:
            input_contract = (
                "The variable 'df' is a pandas DataFrame. Assign an aligned "
                "pandas Series to 'factor'."
            )
        self.system_prompt = (
            "You are a top quantitative researcher. Generate a predictive "
            f"factor using pandas. {input_contract} ONLY output Python code, "
            "without markdown formatting."
        )
        
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
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        
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
                parent_ids=["llm_batch_generate"],
                provenance={
                    "generator": self.__class__.__name__,
                    "model": self.config.get("llm_api_config", {}).get("model", ""),
                    "prompt_sha256": prompt_hash,
                    "iteration": getattr(self, "current_epoch", 0),
                },
            )
            candidates.append(expr)
            
        return candidates

    def evaluate_candidates(self, candidates: List[FactorExpression]) -> EvaluationFeedback:
        if self.evaluator:
            return self.evaluator.evaluate(candidates)
        return EvaluationFeedback()
        
    def update_model(self, candidates: List[FactorExpression], feedback: EvaluationFeedback) -> None:
        logger.info("LLM: Updating reflection history in MinerState...")
        
        successful = []
        for expr in candidates:
            result = feedback.for_candidate(expr)
            if not result:
                continue
            if not result.succeeded:
                self.state.failed_reflections.append({
                    "code": expr.get_source(),
                    "error_type": result.error_type,
                    "error": result.error_message,
                })
                expr.update_provenance(
                    evaluation_status="error",
                    error_type=result.error_type,
                    error_message=result.error_message[:500],
                )
            else:
                metric = result.metrics
                self.state.successful_reflections.append({
                    "code": expr.get_source(),
                    "metrics": metric,
                })
                expr.update_provenance(
                    evaluation_status="success",
                    metrics=dict(metric),
                )
                successful.append(expr)
        self.state.successful_reflections = self.state.successful_reflections[-100:]
        self.state.failed_reflections = self.state.failed_reflections[-100:]
        reflection_snapshot = self.state.get_llm_context_prompt()
        for expr in candidates:
            expr.set_reflection_history(reflection_snapshot)
        self.state.population = successful
