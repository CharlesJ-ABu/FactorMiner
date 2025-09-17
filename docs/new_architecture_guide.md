# 简化因子架构指南

## 概述

新的因子架构完全简化了因子存储和计算流程，移除了 `functions/` 文件夹，直接调用 `user_algo` 中的函数进行因子计算。

## 核心设计理念

1. **极简架构**：只需要 `definitions/` 和 `models/` 两个文件夹
2. **用户完全控制**：用户编写完整的因子计算逻辑
3. **高性能**：直接调用用户函数，无中间层
4. **灵活性**：用户可以使用任何ML框架和特征工程方法

## 目录结构

```
factorlib/
├── definitions/     # 因子定义（元数据）
├── models/          # ML模型文件（.pkl, .joblib等）
└── temp/           # 临时缓存
```

## 用户代码规范

### 1. 算法文件结构

每个算法文件必须包含：

```python
# 算法元信息
ALGORITHM_INFO = {
    'name': '算法名称',
    'description': '算法描述',
    'category': '算法类别',
    'version': '1.0.0',
    'author': '作者'
}

def calculate_factors(data: pd.DataFrame) -> Dict[str, pd.Series]:
    """挖掘阶段：训练模型并生成所有因子"""
    pass

def calculate_single_factor(data: pd.DataFrame, factor_name: str) -> pd.Series:
    """计算阶段：计算单个因子"""
    pass
```

### 2. 核心函数说明

#### `calculate_factors(data)`
- **用途**：挖掘阶段，训练模型并生成所有因子
- **输入**：OHLCV数据
- **输出**：因子名称到因子值的映射
- **职责**：
  - 特征工程
  - 模型训练
  - 因子生成
  - 模型保存

#### `calculate_single_factor(data, factor_name)`
- **用途**：计算阶段，计算单个因子
- **输入**：OHLCV数据，因子名称
- **输出**：因子值序列
- **职责**：
  - 加载预训练模型
  - 特征工程
  - 因子计算

## 系统调用流程

### 1. 挖掘阶段
```python
# factor_builder.py
def _execute_algorithm(self, algo_id, data, **kwargs):
    algorithm_module = self._load_algorithm_module(algo_id)
    return algorithm_module.calculate_factors(data, **kwargs)
```

### 2. 实时计算阶段
```python
# factor_engine.py
def compute_single_factor(self, factor_id, data, **kwargs):
    factor_def = self.storage.load_factor_definition(factor_id)
    algorithm_name = factor_def.computation_data['algorithm_name']
    algorithm_module = self._load_algorithm_module(algorithm_name)
    return algorithm_module.calculate_single_factor(data, factor_name)
```

## 因子定义格式

```json
{
  "factor_id": "algorithm_factor_name",
  "name": "factor_name",
  "description": "因子描述",
  "category": "ml",
  "computation_type": "ml_model",
  "computation_data": {
    "algorithm_name": "algorithm_id",
    "model_file": "model_file.pkl",
    "performance_metrics": {}
  },
  "parameters": {},
  "dependencies": [],
  "output_type": "series",
  "metadata": {
    "checksum": "hash",
    "created_at": "timestamp"
  }
}
```

## 示例算法

### ML动量因子示例

```python
# user_algo/ml_momentum_factor.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
from pathlib import Path

ALGORITHM_INFO = {
    'name': 'ML动量因子',
    'description': '基于机器学习的动量因子挖掘算法',
    'category': 'ml_momentum',
    'version': '1.0.0'
}

def calculate_factors(data: pd.DataFrame) -> Dict[str, pd.Series]:
    """挖掘阶段：训练模型并生成所有因子"""
    # 1. 特征工程
    features = create_features(data)
    
    # 2. 训练不同时间窗口的模型
    factors = {}
    for window in ['5d', '10d', '20d']:
        # 训练模型
        model = train_model(features, data, window)
        
        # 生成因子
        factor_name = f'ml_momentum_{window}'
        factor_values = predict_factor(features, model)
        factors[factor_name] = pd.Series(factor_values, index=data.index)
    
    return factors

def calculate_single_factor(data: pd.DataFrame, factor_name: str) -> pd.Series:
    """计算阶段：计算单个因子"""
    # 从因子名称中提取时间窗口
    window = factor_name.split('_')[-1]
    
    # 加载预训练模型
    model_path = Path("factorlib/minactors/models") / f"ml_momentum_{window}.pkl"
    model = joblib.load(model_path)
    
    # 特征工程和预测
    features = create_features(data)
    predictions = model.predict(features.fillna(0))
    
    return pd.Series(predictions, index=data.index)
```

## 优势

1. **极简架构**：只需要 `definitions/` 和 `models/` 两个文件夹
2. **用户完全控制**：用户编写完整的因子计算逻辑
3. **高性能**：直接调用用户函数，无中间层
4. **灵活性**：用户可以使用任何ML框架和特征工程方法
5. **易维护**：代码结构清晰，逻辑集中

## 迁移指南

### 从旧架构迁移

1. **移除 `functions/` 文件夹**：不再需要生成函数文件
2. **更新算法文件**：添加 `calculate_single_factor` 函数
3. **更新因子定义**：使用新的 `computation_data` 格式
4. **测试新架构**：确保所有功能正常工作

### 兼容性

- 保持原有API接口不变
- 支持现有的因子定义格式
- 向后兼容旧的因子计算方式

## 最佳实践

1. **算法设计**：
   - 将特征工程和模型训练分离
   - 使用标准化的特征工程函数
   - 保存模型和预处理器

2. **错误处理**：
   - 添加适当的异常处理
   - 提供有意义的错误信息
   - 处理数据不足的情况

3. **性能优化**：
   - 缓存模型加载
   - 使用向量化操作
   - 避免重复计算

4. **代码组织**：
   - 将特征工程函数独立出来
   - 使用配置文件管理参数
   - 添加详细的文档和注释

## 总结

新的简化架构大大降低了系统复杂度，提高了灵活性和性能。用户现在有完全的控制权来编写自己的因子计算逻辑，系统只负责调度和存储。这种设计更加符合现代机器学习工作流的需求。
