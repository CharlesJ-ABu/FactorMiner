# 因子存储API使用指南

## 概述

因子存储系统提供了简洁的API接口，让用户只需要提供必要的值就能直接存储到对应的文件夹。新的API设计更加直观和易用。

## 核心API接口

### 1. 保存技术指标因子

```python
storage.save_technical_factor(
    factor_id: str,           # 因子唯一标识
    name: str,               # 因子名称
    function_code: str,      # Python函数代码
    description: str = "",   # 因子描述
    category: str = "technical",  # 因子分类
    entry_point: str = "calculate",  # 入口函数名
    imports: List[str] = None  # 导入语句列表
) -> bool
```

**功能**: 保存技术指标因子到 `technicals/` 目录
**存储位置**: 
- 函数代码 → `technicals/functions/{factor_id}.py`
- 因子定义 → `technicals/definitions/{factor_id}.json`

**示例**:
```python
from factor_miner.core.factor_storage import get_global_storage

storage = get_global_storage()

function_code = '''
def calculate(data: pd.DataFrame, **kwargs) -> pd.Series:
    """计算简单移动平均因子"""
    return data['close'].rolling(20).mean()
'''

success = storage.save_technical_factor(
    factor_id="sma_20",
    name="20日简单移动平均",
    function_code=function_code,
    description="计算20日简单移动平均线",
    category="technical",
    imports=["import pandas as pd"]
)
```

### 2. 保存挖掘因子

```python
storage.save_minactor_factor(
    factor_id: str,           # 因子唯一标识
    name: str,               # 因子名称
    algorithm_name: str,      # 算法名称
    model_file: str = "",     # 模型文件名
    description: str = "",    # 因子描述
    category: str = "ml",     # 因子分类
    performance_metrics: Dict = None  # 性能指标
) -> bool
```

**功能**: 保存挖掘因子到 `minactors/` 目录
**存储位置**: 因子定义 → `minactors/definitions/{factor_id}.json`

**示例**:
```python
success = storage.save_minactor_factor(
    factor_id="ml_momentum_5d",
    name="ML动量因子5日",
    algorithm_name="ml_momentum_factor",
    model_file="ml_momentum_5d.pkl",
    description="基于机器学习的5日动量因子",
    category="ml",
    performance_metrics={
        "ic": 0.15,
        "sharpe": 1.2,
        "win_rate": 0.65
    }
)
```

### 3. 保存模型文件

```python
storage.save_model(
    factor_id: str,           # 因子标识
    model_data: bytes,        # 模型数据（字节）
    model_type: str = "pkl"   # 模型类型（pkl, joblib等）
) -> bool
```

**功能**: 保存模型文件到 `minactors/models/` 目录
**存储位置**: 模型文件 → `minactors/models/{factor_id}.{model_type}`

**示例**:
```python
import pickle

# 准备模型数据
model_data = pickle.dumps({"model": "dummy_model", "version": "1.0"})

success = storage.save_model(
    factor_id="ml_momentum_5d",
    model_data=model_data,
    model_type="pkl"
)
```

### 4. 保存评估结果

```python
storage.save_evaluation(
    factor_id: str,           # 因子标识
    evaluation_data: Dict,    # 评估数据
    source: str = "minactors" # 来源（technicals 或 minactors）
) -> bool
```

**功能**: 保存评估结果
**存储位置**: 评估结果 → `{source}/evaluations/{factor_id}.json`

**示例**:
```python
evaluation_data = {
    "ic_pearson": 0.15,
    "ic_spearman": 0.12,
    "sharpe_ratio": 1.2,
    "win_rate": 0.65,
    "long_short_return": 0.08
}

success = storage.save_evaluation(
    factor_id="ml_momentum_5d",
    evaluation_data=evaluation_data,
    source="minactors"
)
```

### 5. 保存挖掘历史

```python
storage.save_mining_history(
    session_id: str,          # 会话ID
    session_data: Dict        # 会话数据
) -> bool
```

**功能**: 保存挖掘历史到 `minactors/mining_history/` 目录
**存储位置**: 
- 会话汇总 → `minactors/mining_history/mining_sessions.json`
- 详细结果 → `minactors/mining_history/mining_results_{session_id}.json`

**示例**:
```python
session_data = {
    "session_id": "test_session_001",
    "config": {
        "symbols": ["BTCUSDT"],
        "timeframes": ["1h"],
        "algorithms": ["ml_momentum_factor"]
    },
    "results": {
        "total_factors": 3,
        "algorithms_used": ["ml_momentum_factor"],
        "factors": {
            "ml_momentum_5d": "factor_data_here",
            "ml_momentum_10d": "factor_data_here",
            "ml_momentum_20d": "factor_data_here"
        }
    },
    "status": "completed",
    "completed_time": "2024-01-01T12:00:00"
}

success = storage.save_mining_history(
    session_id="test_session_001",
    session_data=session_data
)
```

## 文件夹结构

```
factorlib/
├── technicals/               # 传统技术指标
│   ├── definitions/          # 因子定义文件
│   ├── functions/           # 函数代码文件
│   └── evaluations/         # 评估结果文件
├── minactors/               # 挖掘因子
│   ├── definitions/         # 因子定义文件
│   ├── models/              # ML模型文件(.pkl)
│   ├── evaluations/         # 评估结果文件
│   └── mining_history/      # 挖掘历史记录
└── temp/                    # 临时文件
```

## 使用流程

### 1. 保存技术指标因子
```python
# 1. 准备函数代码
function_code = "def calculate(data): return data['close'].rolling(20).mean()"

# 2. 调用API保存
storage.save_technical_factor(
    factor_id="sma_20",
    name="20日简单移动平均",
    function_code=function_code
)
```

### 2. 保存挖掘因子
```python
# 1. 保存因子定义
storage.save_minactor_factor(
    factor_id="ml_momentum_5d",
    name="ML动量因子5日",
    algorithm_name="ml_momentum_factor"
)

# 2. 保存模型文件
model_data = pickle.dumps(trained_model)
storage.save_model("ml_momentum_5d", model_data)

# 3. 保存评估结果
storage.save_evaluation("ml_momentum_5d", evaluation_results)
```

### 3. 保存挖掘历史
```python
# 保存完整的挖掘会话
storage.save_mining_history(session_id, session_data)
```

## 优势

1. **简洁易用**: 只需要提供必要的值，系统自动处理文件路径和结构
2. **类型安全**: 根据因子类型自动选择正确的存储位置
3. **统一接口**: 所有存储操作都通过统一的API接口
4. **自动管理**: 系统自动创建目录结构和文件格式
5. **向后兼容**: 保持与现有系统的兼容性

## 注意事项

1. **因子ID唯一性**: 确保 `factor_id` 在系统中唯一
2. **函数代码格式**: 技术指标的函数代码必须是完整的Python函数
3. **模型数据格式**: 模型数据必须是字节格式
4. **评估数据格式**: 评估数据必须是字典格式
5. **错误处理**: 所有API都返回布尔值表示成功/失败状态

## 完整示例

参考 `examples/storage_api_usage.py` 文件查看完整的使用示例。
