import logging
import numpy as np
import pandas as pd
from typing import List, Any

from core.miner.paradigms.base import BaseFactorMiner
from core.miner.registry import MinerRegistry
from core.miner.expressions import FactorExpressionTensor
from core.miner.entities import EvaluationFeedback

logger = logging.getLogger(__name__)

class MockTensor:
    """模拟 PyTorch 的 Tensor 对象，能够携带梯度标志并暂存前向传播的输入特征"""
    def __init__(self, data_array: np.ndarray, inputs: np.ndarray = None, requires_grad: bool = True):
        self.data = data_array
        self.inputs = inputs
        self.requires_grad = requires_grad

class MockLinearLayer:
    """纯 NumPy 实现的单层感知机 (Mock NN)"""
    def __init__(self, features: List[str], out_channels: int):
        self.features = features
        self.in_channels = len(features)
        self.out_channels = out_channels
        # 初始化权重 W (in_channels x out_channels)
        self.W = np.random.randn(self.in_channels, self.out_channels) * 0.1

    def __call__(self, df: pd.DataFrame) -> MockTensor:
        # 1. 提取特征矩阵
        X = df[self.features].fillna(0).values
        # 2. 简单的 Z-score 归一化 (防爆栈)
        X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-9)
        # 3. 前向传播
        out = np.dot(X, self.W)
        # 返回 MockTensor，将输入 X 存下来用于反向传播
        return MockTensor(data_array=out, inputs=X, requires_grad=True)

class MyNNExpression(FactorExpressionTensor):
    """
    继承自底层的 FactorExpressionTensor
    因为底层模板中将 forward pass 隐去了，我们在这里重写 compute 以便它被 Evaluator 调用。
    """
    def compute(self, data: pd.DataFrame):
        if self.model_instance:
            return self.model_instance(data)
        return None

@MinerRegistry.register("MyCustomNN")
class MyCustomNNMiner(BaseFactorMiner):
    """
    轻量级的深度学习模拟挖掘器。
    它生成携带 NN 模型引用的 FactorExpression，然后通过 Evaluator 返回的
    MockTensor 来进行手动的梯度下降 (Gradient Descent)。
    """
    def initialize_search_space(self) -> None:
        logger.info("Initializing MyCustomNN Search Space (Mock Linear Model)...")
        self.terminals = self.config.get("data_feeds", {}).get("required_streams", ["close", "volume"])
        self.hidden_dim = self.config.get("hidden_dim", 8)
        self.lr = self.config.get("learning_rate", 0.01)
        
        # 实例化我们的微型神经网络
        self.model = MockLinearLayer(self.terminals, self.hidden_dim)
        self.epoch = 0

    def generate_candidates(self) -> List[FactorExpressionTensor]:
        logger.info(f"MyCustomNN: Generating candidates via NN Forward Pass (Channels: {self.hidden_dim})...")
        # 深度学习中，通常把整个模型包装为一个因子组
        expr = MyNNExpression(
            model_version_id=f"nn_v{self.epoch}", 
            channel_idx=-1, # -1 意味着整个模型的所有通道
            model_instance=self.model
        )
        return [expr]

    def evaluate_candidates(self, candidates: List[FactorExpressionTensor]) -> EvaluationFeedback:
        if self.evaluator:
            return self.evaluator.evaluate(candidates)
        return EvaluationFeedback()

    def update_model(self, candidates: List[FactorExpressionTensor], feedback: EvaluationFeedback) -> None:
        """
        核心反向传播逻辑 (Backpropagation)
        我们通过 MSE (均方误差) 损失函数计算梯度，然后更新 self.model.W
        """
        logger.info("MyCustomNN: Updating Model Weights via Pseudo-Gradient Descent...")
        
        raw = getattr(feedback, "raw_outputs", None)
        if raw is None:
            logger.warning("No raw_outputs tensor returned from Evaluator. Cannot perform backprop.")
            return

        # 1. 拿到前向传播时的预测值 (N x hidden_dim) 和输入特征 (N x in_features)
        y_pred = raw.data
        X = raw.inputs
        
        # 2. 从 Evaluator 获取真实的 Label (这里简单使用 收益率 returns 作为靶向 Label)
        returns_df = self.evaluator.data_client.get_returns()
        if returns_df is None or returns_df.empty:
            logger.warning("No returns data available for labeling.")
            return
            
        y_true = returns_df.fillna(0).values.reshape(-1, 1) # (N, 1)
        
        # 由于我们有 hidden_dim 个通道，我们将标签 Broadcast 成 (N, hidden_dim) 
        # 只是为了演示：所有的因子头都在尝试预测同一种收益率
        y_true_broadcast = np.tile(y_true, (1, self.hidden_dim))
        
        # 3. 计算 MSE 损失: Loss = mean( (y_pred - y_true)^2 )
        loss = np.mean((y_pred - y_true_broadcast) ** 2)
        logger.info(f"Epoch {self.epoch} - Current MSE Loss: {loss:.6f}")
        
        # 4. 反向传播 (链式法则)
        # dLoss/dy_pred = 2 * (y_pred - y_true) / N
        grad_y = 2 * (y_pred - y_true_broadcast) / y_pred.shape[0]
        
        # dLoss/dW = X.T * dLoss/dy_pred
        dW = np.dot(X.T, grad_y) # 形状: (in_features, hidden_dim)
        
        # 5. 权重更新 (Gradient Descent)
        self.model.W -= self.lr * dW
        logger.debug(f"Weight updates (dW sum: {np.sum(np.abs(dW)):.6f}) applied.")
        
        self.epoch += 1
