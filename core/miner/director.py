import logging
from typing import Dict, Any, List

from core.miner.entities import FactorMetadata, EvaluationFeedback
from core.storage.factor_storage import get_global_storage
from core.evaluation.evaluator import ParallelEvaluator
from core.miner.paradigms.base import BaseFactorMiner

logger = logging.getLogger(__name__)

class FactorMinerDirector:
    """
    因子挖掘的全局网关 Facade
    负责组装数据源、配置、评价器，并驱动底层具体流派的 Miner 跑完闭环。
    """
    def __init__(self, config: Dict, data_client: Any):
        self.config = config
        self.data_client = data_client
        self.storage_client = get_global_storage()
        
        # 1. 初始化统一并行评价器
        self.evaluator = ParallelEvaluator(self.data_client, self.config)
        
        self.universe = config.get("universe", "DEFAULT_UNIVERSE")
        self.data_feeds = config.get("data_feeds", {})
        
        # 2. 实例化流派 Miner
        paradigm = config.get("paradigm", "GP")
        self.miner = self._instantiate_miner(paradigm, config, data_client)
        # 将统一的 evaluator 和 storage_client 注入到 miner 中
        self.miner.evaluator = self.evaluator
        self.miner.storage_client = self.storage_client
        
        self.latest_factor_ids: List[str] = []

    def _instantiate_miner(self, paradigm: str, config: Dict, data: Any) -> BaseFactorMiner:
        from core.miner.registry import MinerRegistry
        
        # 优先查询用户自定义注册的 Miner
        if paradigm in MinerRegistry._registry:
            miner_cls = MinerRegistry._registry[paradigm]
            return miner_cls(data, config)
            
        # Fallback 到系统原生的四大流派
        if paradigm == "GP":
            from .paradigms.gp_miner import GPFactorMiner
            return GPFactorMiner(data, config)
        elif paradigm == "LLM":
            from .paradigms.llm_miner import LLMFactorMiner
            return LLMFactorMiner(data, config)
        elif paradigm == "DL":
            from .paradigms.dl_miner import DLFactorMiner
            return DLFactorMiner(data, config)
        elif paradigm == "RL":
            from .paradigms.rl_miner import RLFactorMiner
            return RLFactorMiner(data, config)
        else:
            raise ValueError(f"Unknown paradigm: {paradigm}. Did you forget to register it?")

    def run(self, max_iterations: int = 10, progress_callback=None) -> List[Any]:
        logger.info(f"Starting FactorMiner loop for paradigm: {self.config.get('paradigm')}")
        
        # 委托给底层的通用范式引擎去跑 (它自带去重和生命周期)
        best_candidates = self.miner.mine(max_iterations, progress_callback=progress_callback)
        
        self._save_excellent_factors(best_candidates)
        self._print_results_table(best_candidates)
        
        logger.info("FactorMiner loop completed.")
        return best_candidates

    def _save_excellent_factors(self, candidates):
        import uuid
        from datetime import datetime
        
        self.latest_factor_ids = []
        for cand in candidates:
            factor_id = f"fac_{uuid.uuid4().hex[:8]}"
            self.latest_factor_ids.append(factor_id)
            
            metadata = FactorMetadata(
                factor_id=factor_id,
                miner_type=self.config.get("paradigm"),
                user_id="CharlesJ-ABu",
                metrics=cand.metrics,
                logic_hash=getattr(cand, 'logic_hash', ""),
                created_at=datetime.now().isoformat()
            )
            
            # 存储分流
            from core.miner.expressions import FactorExpressionAST, FactorExpressionCode, FactorExpressionTensor, FactorExpressionAction
            if isinstance(cand, FactorExpressionAST):
                self.storage_client.save_gp_factor(cand.get_source(), metadata)
            elif isinstance(cand, FactorExpressionCode):
                self.storage_client.save_llm_factor(cand.get_source(), cand.get_reflection_history(), metadata)
            elif isinstance(cand, FactorExpressionAction):
                self.storage_client.save_rl_factor(cand.get_source(), b"", metadata)
            elif isinstance(cand, FactorExpressionTensor):
                src = cand.get_source()
                # Persist the model once alongside the channel metadata.  A
                # channel ID alone cannot reproduce a trained neural factor.
                weights = getattr(getattr(cand, "model_instance", None), "W", None)
                if weights is not None:
                    try:
                        self.storage_client.save_model_weights(
                            src["model_version"], weights.tobytes()
                        )
                    except Exception as exc:
                        logger.warning(
                            "Failed to persist weights for model %s: %s",
                            src["model_version"], exc,
                        )
                self.storage_client.save_dl_factor_channel(src["model_version"], src["channel"], metadata)

    def _print_results_table(self, candidates):
        try:
            from rich.console import Console
            from rich.table import Table
            
            console = Console()
            mining_mode = self.config.get("data_feeds", {}).get("mining_mode", "sequential_single")
            universe = self.config.get("universe", "DEFAULT_UNIVERSE")
            
            table = Table(
                title=f"🎯 Mining Completed | Mode: {mining_mode} | Universe: {universe}",
                show_header=True, 
                header_style="bold magenta"
            )
            table.add_column("Rank", style="dim", width=6)
            table.add_column("Factor ID", width=12)
            
            if mining_mode == "sequential_single":
                table.add_column("Target Pair")
                
            table.add_column("Paradigm")
            table.add_column("Logic / Expression", overflow="fold")
            table.add_column("Fitness Score", justify="right")
            table.add_column("IC", justify="right")
            table.add_column("RankIC", justify="right")
            table.add_column("Complexity")
            
            for idx, (factor_id, cand) in enumerate(zip(self.latest_factor_ids, candidates)):
                metrics = cand.metrics
                fitness = f"{metrics.get('fitness_score', 0):.4f}"
                ic = f"{metrics.get('IC', 0):.4f}"
                rank_ic = f"{metrics.get('RankIC', 0):.4f}"
                paradigm = self.config.get("paradigm")
                logic = cand.to_display_string(max_length=None)
                complexity = cand.get_complexity()
                
                if mining_mode == "sequential_single":
                    target_pair = self.config.get("data_feeds", {}).get("pairs", [universe])[0]
                    table.add_row(str(idx + 1), factor_id, target_pair, paradigm, logic, fitness, ic, rank_ic, complexity)
                else:
                    table.add_row(str(idx + 1), factor_id, paradigm, logic, fitness, ic, rank_ic, complexity)
                    
            console.print(table)
        except ImportError:
            logger.info("Rich library not installed. Skipping fancy table.")

    def get_latest_factor_ids(self) -> List[str]:
        return self.latest_factor_ids
