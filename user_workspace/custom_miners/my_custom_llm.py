import logging
import re
import random
import hashlib
import os
from typing import Any, Dict, List

from core.miner.paradigms.base import BaseFactorMiner
from core.miner.registry import MinerRegistry
from core.miner.expressions import FactorExpressionCode
from core.miner.entities import EvaluationFeedback
from core.evaluation.evaluator import RestrictedSandbox
from user_workspace.experiment_tools import LLMExperimentRecorder

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
        self.is_cross_asset = (
            self.config.get("data_feeds", {}).get("mining_mode")
            == "cross_asset"
        )
        sandbox_config = self.config.get("llm_sandbox", {})
        self.sandbox = RestrictedSandbox(
            timeout_seconds=sandbox_config.get("timeout_seconds", 5.0),
            cpu_seconds=sandbox_config.get("cpu_seconds", 3),
            memory_mb=sandbox_config.get("memory_mb", 1024),
        )
        api_config = self._resolved_api_config()
        self.recorder = LLMExperimentRecorder(
            config=self.config,
            population_size=self.population_size,
            resolved_api_config=api_config,
            paradigm_name=self.__class__.__name__,
        )

        # 初始化 API Manager
        require_live_api = bool(
            self.config.get("experiment", {}).get("require_live_api", False)
        )
        if not api_config or not LLMAPIManager:
            if require_live_api:
                self.recorder.record(
                    "run_blocked",
                    reason="A live LLM API is required but no usable API key is configured.",
                )
                raise RuntimeError(
                    "This experiment requires a live LLM API. Set one of the configured "
                    "llm_api_config.keys_env environment variables."
                )
            logger.warning("LLMAPIManager is not available or api_config is missing. Will use fallback random strings.")
            self.api_manager = None
        else:
            self.api_manager = LLMAPIManager(api_config)
        self.recorder.record(
            "miner_initialized",
            api_mode="live" if self.api_manager else "fallback",
            model=(api_config or {}).get("model", ""),
        )

    @staticmethod
    def _looks_like_placeholder(value: str) -> bool:
        lowered = value.lower()
        markers = ("dummy", "testing", "your_api", "your-api", "changeme", "xxx")
        return not value or any(marker in lowered for marker in markers)

    def _resolved_api_config(self) -> Dict[str, Any] | None:
        raw_config = self.config.get("llm_api_config")
        if not raw_config:
            return None
        api_config = dict(raw_config)
        keys = [
            str(value)
            for value in api_config.get("keys", [])
            if not self._looks_like_placeholder(str(value))
        ]
        for variable_name in api_config.get("keys_env", []):
            value = os.environ.get(str(variable_name), "")
            if value and not self._looks_like_placeholder(value):
                keys.append(value)
        model_env = api_config.get("model_env")
        if model_env and os.environ.get(str(model_env)):
            api_config["model"] = os.environ[str(model_env)]
        base_url_env = api_config.get("base_url_env")
        if base_url_env and os.environ.get(str(base_url_env)):
            api_config["base_url"] = os.environ[str(base_url_env)]
        base_url = str(api_config.get("base_url", "")).rstrip("/")
        if base_url and not base_url.endswith("/chat/completions"):
            api_config["base_url"] = f"{base_url}/chat/completions"
        api_config["keys"] = list(dict.fromkeys(keys))
        api_config.pop("keys_env", None)
        api_config.pop("model_env", None)
        api_config.pop("base_url_env", None)
        if not api_config["keys"]:
            return None
        return api_config
            
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

        if self.is_cross_asset:
            data_contract = (
                "The variable `df` maps each feature name to an aligned pandas "
                "DataFrame whose rows are timestamps and columns are assets. "
                "Assign an aligned pandas DataFrame to `factor`."
            )
        else:
            data_contract = (
                "The variable `df` is a pandas DataFrame. Assign an aligned "
                "pandas Series to `factor`."
            )
        prompt = f"""
You are an expert quantitative researcher. Please write a simple Python calculation for a financial alpha factor.
Available features: {self.terminals}.
{data_contract}
Your code should be wrapped in a Markdown ```python``` block.

Reflection History from previous generations (learn from it):
{self.state.get_llm_context_prompt()}
"""
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        logger.debug(f"Prompt sent to LLM:\n{prompt}")
        epoch = getattr(self, "current_epoch", 0)
        self.recorder.generation_started(
            epoch,
            prompt_hash,
            prompt,
            self.state.get_llm_context_prompt(),
        )

        candidates = []
        if self.api_manager:
            prompts = [prompt] * self.population_size
            logger.info("Sending batch requests to LLM API...")
            responses = self.api_manager.batch_generate(prompts)
            
            for candidate_index, resp in enumerate(responses):
                if resp:
                    code = self._extract_code(resp)
                    expression = self._build_expression(
                        code, prompt_hash, candidate_index, "live_api"
                    )
                    candidates.append(expression)
                    self.recorder.candidate_generated(
                        expression,
                        epoch,
                        candidate_index,
                        resp,
                        "live_api",
                    )
                else:
                    allow_fallback = bool(
                        self.config.get("experiment", {}).get(
                            "allow_fallback", True
                        )
                    )
                    if not allow_fallback:
                        self.recorder.generation_failed(
                            epoch,
                            candidate_index,
                            "LLM API returned no response and fallback is disabled.",
                        )
                        continue
                    logger.warning("LLM API returned None (e.g. invalid key). Using Fallback code.")
                    code = self._get_fallback_code()
                    expression = self._build_expression(
                        code, prompt_hash, candidate_index, "fallback"
                    )
                    candidates.append(expression)
                    self.recorder.candidate_generated(
                        expression,
                        epoch,
                        candidate_index,
                        None,
                        "fallback",
                    )
        else:
            for candidate_index in range(self.population_size):
                code = self._get_fallback_code()
                expression = self._build_expression(
                    code, prompt_hash, candidate_index, "fallback"
                )
                candidates.append(expression)
                self.recorder.candidate_generated(
                    expression,
                    epoch,
                    candidate_index,
                    None,
                    "fallback",
                )
        self.recorder.generation_completed(epoch, candidates)
        if not candidates:
            raise RuntimeError(
                "LLM API returned no usable responses; this experiment forbids fallback "
                "candidate generation."
            )
        return candidates

    def _build_expression(
        self,
        code: str,
        prompt_hash: str,
        candidate_index: int = 0,
        generation_source: str = "unknown",
    ) -> FactorExpressionCode:
        epoch = getattr(self, "current_epoch", 0)
        candidate_id = hashlib.sha256(
            f"{epoch}:{candidate_index}:{code}".encode("utf-8")
        ).hexdigest()[:16]
        return FactorExpressionCode(
            code_str=code,
            sandbox=self.sandbox,
            parent_ids=["custom_llm_batch_generate"],
            provenance={
                "generator": self.__class__.__name__,
                "model": self.config.get("llm_api_config", {}).get("model", ""),
                "prompt_sha256": prompt_hash,
                "iteration": epoch,
                "candidate_index": candidate_index,
                "candidate_id": candidate_id,
                "generation_source": generation_source,
            },
        )

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
        for expr in candidates:
            result = feedback.for_candidate(expr)
            if result and result.succeeded:
                score = result.metrics.get("fitness_score", 0)
                self.state.successful_reflections.append({
                    "code": expr.get_source(),
                    "metrics": result.metrics,
                })
                expr.update_provenance(
                    evaluation_status="success",
                    metrics=dict(result.metrics),
                )
                self.recorder.candidate_succeeded(
                    expr,
                    getattr(self, "current_epoch", 0),
                    result.metrics,
                    self.data,
                )
                scored.append((score, expr))
            elif result:
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
                self.recorder.candidate_failed(
                    expr,
                    getattr(self, "current_epoch", 0),
                    result.error_type,
                    result.error_message,
                )

        self.state.successful_reflections = self.state.successful_reflections[-100:]
        self.state.failed_reflections = self.state.failed_reflections[-100:]
        reflection_snapshot = self.state.get_llm_context_prompt()
        for expr in candidates:
            expr.set_reflection_history(reflection_snapshot)
        self.recorder.reflection_updated(
            getattr(self, "current_epoch", 0),
            len(self.state.successful_reflections),
            len(self.state.failed_reflections),
            reflection_snapshot,
        )
                
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

        # Reflection is a training artifact; the scored code expressions are
        # the research artifacts that Director can persist and Inspector can
        # review. Retain a bounded Top-K archive across generations.
        top_k = self.config.get("top_k_factors", self.population_size)
        archived = {candidate.logic_hash: candidate for candidate in self.state.population}
        for _, candidate in scored:
            archived[candidate.logic_hash] = candidate
        self.state.population = sorted(
            archived.values(),
            key=lambda candidate: candidate.metrics.get("fitness_score", float("-inf")),
            reverse=True,
        )[:top_k]
        self.recorder.archive_updated(
            getattr(self, "current_epoch", 0),
            self.state.population,
        )

    def _log_epoch(
        self,
        epoch: int,
        candidates: List[FactorExpressionCode],
        metrics: List[Dict[str, float]],
    ) -> None:
        super()._log_epoch(epoch, candidates, metrics)
        self.recorder.epoch_completed(epoch, candidates)

    def mine(self, n_iterations: int, progress_callback=None) -> List[FactorExpressionCode]:
        candidates = super().mine(
            n_iterations,
            progress_callback=progress_callback,
        )
        self.recorder.run_completed(n_iterations, candidates)
        return candidates
