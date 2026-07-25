from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Dict, Any, Union, Optional

class FactorExpression(ABC):
    """
    统一的因子中间体表达式
    不论生成器是 GP、RL、LLM 还是 NN，最终吐出的都是该类的子类实例。
    """
    
    def __init__(self):
        self.metrics: Dict[str, float] = {}
        self.logic_hash: str = ""
    @abstractmethod
    def compute(self, data: pd.DataFrame) -> Union[pd.Series, pd.DataFrame, Any]:
        """
        统一计算接口。
        - AST派生类: 在这里遍历解析树并计算
        - Code派生类: 在这里使用 exec/eval 计算
        - DL派生类: 在这里执行 model.forward(data)
        """
        pass
        
    @abstractmethod
    def get_source(self) -> Any:
        """
        获取底层真正的逻辑表达（比如 ast dict，code 字符串，或 nn.Module）
        """
        pass
        
    @abstractmethod
    def get_complexity(self) -> str:
        """
        获取因子的复杂度标签，如 "Depth: 5", "LoC: 10", "Params: 100"
        """
        pass
        
    @abstractmethod
    def to_display_string(self, max_length: int = 50) -> str:
        """
        格式化为用于终端或 UI 表格中展示的精简字符串
        """
        pass

class LineageTrackableMixIn:
    """可追溯谱系的能力标签"""
    def __init__(self, parent_ids: List[str] = None):
        self._parent_ids = parent_ids or []
        
    def get_lineage_parents(self) -> List[str]:
        return self._parent_ids

class LLMReflectableMixIn:
    """具有大模型反思记忆的能力标签"""
    def __init__(self, reflection_history: str = ""):
        self._reflection_history = reflection_history
        
    def get_reflection_history(self) -> str:
        return self._reflection_history

class FactorExpressionAST(FactorExpression, LineageTrackableMixIn):
    """供遗传规划 (GP) 专用的 AST 表达式"""
    def __init__(self, ast_dict: Dict, parent_ids: List[str] = None):
        FactorExpression.__init__(self)
        LineageTrackableMixIn.__init__(self, parent_ids)
        self.ast_dict = ast_dict
        
    def compute(self, data: pd.DataFrame):
        from core.miner.operator_runtime import evaluate_ast

        return evaluate_ast(self.ast_dict, data)
        
    def get_source(self) -> Dict:
        return self.ast_dict
        
    def get_complexity(self) -> str:
        # 简单计算深度
        def get_depth(node):
            if not isinstance(node, dict) or 'op' not in node:
                return 1
            left_depth = get_depth(node.get('left', None))
            right_depth = get_depth(node.get('right', None))
            return 1 + max(left_depth, right_depth)
        return f"Depth: {get_depth(self.ast_dict)}"
        
    def to_display_string(self, max_length: int = None) -> str:
        def to_str(node):
            if not isinstance(node, dict) or 'op' not in node:
                return str(node)
            left = to_str(node.get('left'))
            if node.get('right') is None:
                return f"{node['op']}({left})"
            return f"{node['op']}({left}, {to_str(node.get('right'))})"
        s = to_str(self.ast_dict)
        if max_length is not None and len(s) > max_length:
            return s[:max_length-3] + "..."
        return s

class FactorExpressionCode(FactorExpression, LineageTrackableMixIn, LLMReflectableMixIn):
    """供大语言模型 (LLM) 生成纯源码使用的表达式"""
    def __init__(
        self,
        code_str: str,
        sandbox: Any = None,
        parent_ids: List[str] = None,
        reflection_history: str = "",
        provenance: Optional[Dict[str, Any]] = None,
    ):
        FactorExpression.__init__(self)
        LineageTrackableMixIn.__init__(self, parent_ids)
        LLMReflectableMixIn.__init__(self, reflection_history)
        self.code_str = code_str
        self.sandbox = sandbox
        self.provenance = provenance or {}
        
    def compute(self, data: pd.DataFrame):
        if not self.sandbox:
            raise RuntimeError(
                "FactorExpressionCode requires an explicit sandbox executor; "
                "unsafe exec fallback is disabled"
            )
        return self.sandbox.execute_factor_code(self.code_str, data)

    def set_reflection_history(self, reflection_history: str) -> None:
        self._reflection_history = reflection_history

    def get_provenance(self) -> Dict[str, Any]:
        return dict(self.provenance)

    def update_provenance(self, **values: Any) -> None:
        self.provenance.update(values)
        
    def get_source(self) -> str:
        return self.code_str
        
    def get_complexity(self) -> str:
        loc = len([line for line in self.code_str.split('\n') if line.strip()])
        return f"LoC: {loc}"
        
    def to_display_string(self, max_length: int = None) -> str:
        s = self.code_str.replace('\n', ' ').strip()
        if max_length is not None and len(s) > max_length:
            return s[:max_length-3] + "..."
        return s

class FactorExpressionTensor(FactorExpression):
    """供 NN 范式使用的网络计算表达式。"""
    def __init__(self, model_version_id: str, channel_idx: int, model_instance: Any = None):
        FactorExpression.__init__(self)
        self.model_version_id = model_version_id
        self.channel_idx = channel_idx
        self.model_instance = model_instance # PyTorch 模型实例
        
    def compute(self, data: pd.DataFrame):
        # NN 训练阶段不在这里截断计算，需要保留梯度图
        if self.model_instance:
            # 假设 data 转为 tensor 传入
            # return self.model_instance(data)[:, self.channel_idx]
            pass
        return None
        
    def get_source(self) -> Dict:
        return {"model_version": self.model_version_id, "channel": self.channel_idx}

    def export_model_artifact(self):
        """Return a portable artifact when the custom model supports the NN contract."""
        exporter = getattr(self.model_instance, "export_artifact", None)
        return exporter() if callable(exporter) else None
        
    def get_complexity(self) -> str:
        # 简单统计模型参数
        params = 0
        if self.model_instance and hasattr(self.model_instance, 'get_parameter_count'):
            params = int(self.model_instance.get_parameter_count())
        elif self.model_instance and hasattr(self.model_instance, 'parameters'):
            params = sum(p.numel() for p in self.model_instance.parameters() if p.requires_grad)
        elif self.model_instance and hasattr(self.model_instance, 'W'):
            # Supports the NumPy-based reference miner as well as torch-like
            # models above.
            params = getattr(self.model_instance.W, 'size', 0)
        return f"Params: {params}"
        
    def to_display_string(self, max_length: int = None) -> str:
        s = f"NNModel(v={self.model_version_id}) [Ch: {self.channel_idx}]"
        if max_length is not None and len(s) > max_length:
            return s[:max_length-3] + "..."
        return s

class FactorExpressionAction(FactorExpression):
    """供强化学习 (RL) 使用的动作轨迹序列表达式"""
    def __init__(self, action_sequence: List[str]):
        FactorExpression.__init__(self)
        self.action_sequence = action_sequence
        
    def compute(self, data: pd.DataFrame):
        # 根据动作序列累加计算
        return pd.Series(index=data.index, data=0)
        
    def get_source(self) -> List[str]:
        return self.action_sequence
        
    def get_complexity(self) -> str:
        return f"Depth: {len(self.action_sequence)}"
        
    def to_display_string(self, max_length: int = None) -> str:
        s = " -> ".join(self.action_sequence)
        if max_length is not None and len(s) > max_length:
            return s[:max_length-3] + "..."
        return s
