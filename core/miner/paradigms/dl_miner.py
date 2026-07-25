from typing import List, Dict, Any

from core.miner.paradigms.base import BaseFactorMiner
from core.miner.expressions import FactorExpression
from core.miner.entities import EvaluationFeedback

class DLFactorMiner(BaseFactorMiner):
    """Deprecated compatibility symbol; use the registered NN miner."""

    def __init__(self, data: Any, config: Dict):
        raise RuntimeError(
            "DLFactorMiner is deprecated and cannot run. "
            "Use paradigm 'NN' (or 'MyCustomNN') instead."
        )

    def initialize_search_space(self) -> None:
        raise RuntimeError("DLFactorMiner is deprecated; use NN.")
        
    def generate_candidates(self) -> List[FactorExpression]:
        raise RuntimeError("DLFactorMiner is deprecated; use NN.")

    def evaluate_candidates(self, candidates: List[FactorExpression]) -> EvaluationFeedback:
        raise RuntimeError("DLFactorMiner is deprecated; use NN.")
        
    def update_model(self, candidates: List[FactorExpression], feedback: EvaluationFeedback) -> None:
        raise RuntimeError("DLFactorMiner is deprecated; use NN.")
