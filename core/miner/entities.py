from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

@dataclass
class FactorMetadata:
    """因子的元数据（灵魂）"""
    factor_id: str                   # 因子唯一全球标识 (例如: alp_llm_20260710_001)
    miner_type: str                  # 流派标签: 'GP', 'RL', 'LLM', 'NN'
    user_id: str                     # 提交/运行该任务的研究员ID
    
    # 因子生命周期状态机
    # 状态流转: DISCOVERED (刚挖出) -> INSPECTED (已人工审查) -> PAPER_TRADING (模拟盘) -> LIVE (实盘) -> RETIRED (衰减退役)
    lifecycle_status: str = field(default="DISCOVERED") 
    
    # 谱系追踪 (Lineage): 支撑因子的达尔文演进或反思链条
    parent_ids: List[str] = field(default_factory=list) # 遗传变异的父代ID，或LLM上一版报错的代码ID
    generation_config: Dict[str, Any] = field(default_factory=dict) # 产生该因子时的超参数快照
    
    # 灵魂索引 (Soul References): 不同流派指向其逻辑核心的钥匙
    logic_reference: Dict[str, Any] = field(default_factory=dict) 
    logic_hash: str = ""             # 用于全局硬去重的代码逻辑哈希值
    
    # 评测指标 (肉体成绩单)
    metrics: Dict[str, float] = field(default_factory=dict) # 留存当时的评估指标 (IC, IR, Sharpe等)
    data_lineage: Dict[str, Any] = field(default_factory=dict) # 快照来源、数据范围与收益定义
    created_at: str = ""

@dataclass
class CandidateEvaluation:
    """One immutable association between a candidate and its evaluation."""
    candidate: Any
    status: str
    metrics: Dict[str, float] = field(default_factory=dict)
    raw_outputs: Any = None
    error_type: str = ""
    error_message: str = ""
    traceback: str = ""

    @property
    def succeeded(self) -> bool:
        return self.status == "success"

    def as_legacy_error(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "expr": self.candidate,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "traceback": self.traceback,
        }


@dataclass
class EvaluationFeedback:
    """Candidate-aligned feedback, with compatibility views for older miners."""
    results: List[CandidateEvaluation] = field(default_factory=list)
    raw_targets: Any = None  # 用于 DL 计算 Loss 的配置化未来收益标签

    def for_candidate(self, candidate: Any) -> Optional[CandidateEvaluation]:
        return next(
            (result for result in self.results if result.candidate is candidate),
            None,
        )

    @property
    def metrics(self) -> List[Dict[str, float]]:
        return [
            result.metrics
            for result in self.results
            if result.succeeded and result.metrics
        ]

    @property
    def execution_status(self) -> List[Dict[str, Any]]:
        return [
            result.as_legacy_error()
            for result in self.results
            if not result.succeeded
        ]

    @property
    def raw_outputs(self) -> Any:
        outputs = [
            result.raw_outputs
            for result in self.results
            if result.succeeded and result.raw_outputs is not None
        ]
        return outputs[-1] if outputs else None

class MinerState:
    """
    流派状态管理器：统一承载不同算法的记忆和演进实体
    """
    def __init__(self):
        # 1. 适用于 GP
        self.population: List[Any] = []
        
        # 2. 适用于 LLM (成功和失败的经验池)
        self.successful_reflections: List[Dict] = []
        self.failed_reflections: List[Dict] = []
        
        # 3. 适用于 RL / NN
        self.model_weights_bytes: Optional[bytes] = None
        self.replay_buffer: List[Any] = []

    def get_llm_context_prompt(self) -> str:
        """给 LLM 流派快速组装 Few-shot 提示词的方法"""
        prompt = "Here are past successful examples:\n"
        for item in self.successful_reflections[-5:]: # 取最近5个
            prompt += f"Code: {item['code']}\nPerformance: {item['metrics']}\n"
        prompt += "Here are recent failed examples to avoid:\n"
        for item in self.failed_reflections[-5:]:
            error_type = item.get("error_type", "ExecutionError")
            error = str(item.get("error", ""))[:500]
            prompt += f"Code: {item['code']}\nFailure: {error_type}: {error}\n"
        return prompt
