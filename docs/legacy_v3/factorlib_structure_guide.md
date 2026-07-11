# FactorLib 文件夹结构指南

## 概述

FactorLib 已重新组织为两个主要文件夹：`technicals/`（传统技术指标）和 `minactors/`（挖掘因子），使文件管理更加清晰和有序。

## 新的文件夹结构

```
factorlib/
├── technicals/                    # 传统技术指标
│   ├── definitions/              # 因子定义（JSON格式）
│   ├── functions/                # 因子函数代码（Python文件）
│   └── evaluations/              # 因子评估结果
├── minactors/                    # 挖掘因子
│   ├── definitions/              # 因子定义（JSON格式）
│   ├── models/                   # ML模型文件（.pkl文件）
│   ├── evaluations/              # 因子评估结果
│   └── mining_history/           # 挖掘历史记录
└── temp/                         # 临时缓存
```

## 文件夹说明

### technicals/ - 传统技术指标

包含所有传统的技术分析指标，如移动平均线、RSI、MACD等。

- **definitions/**: 存储因子定义文件（JSON格式）
  - 包含因子的元数据、参数、计算逻辑等
  - 文件命名格式：`{factor_id}.json`
  
- **functions/**: 存储因子计算函数（Python文件）
  - 包含因子的具体计算逻辑
  - 文件命名格式：`{factor_id}.py`
  
- **evaluations/**: 存储因子评估结果
  - 包含因子的性能指标、回测结果等
  - 文件命名格式：`{factor_id}_evaluation.json`

### minactors/ - 挖掘因子

包含通过算法挖掘生成的因子，主要是机器学习相关的因子。

- **definitions/**: 存储因子定义文件（JSON格式）
  - 包含算法信息、模型文件路径等
  - 文件命名格式：`{algorithm_id}_{factor_name}.json`
  
- **models/**: 存储机器学习模型文件
  - 包含训练好的模型文件（.pkl格式）
  - 文件命名格式：`{algorithm_id}_{factor_name}.pkl`
  
- **evaluations/**: 存储因子评估结果
  - 包含挖掘因子的性能评估
  - 文件命名格式：`{factor_id}_evaluation.json`
  
- **mining_history/**: 存储挖掘历史记录
  - 包含挖掘会话记录和详细结果
  - 文件命名格式：`mining_results_{session_id}.json`

## 文件分类规则

### 传统技术指标 (technicals/)

以下类型的因子会被归类到 `technicals/` 文件夹：

- **计算类型**：`formula` 或 `function`
- **类别**：`trend`、`momentum`、`volatility`、`volume`、`oscillator` 等
- **示例**：
  - `sma.json` - 简单移动平均线
  - `rsi.json` - 相对强弱指数
  - `macd.json` - MACD指标
  - `bb_upper.json` - 布林带上轨

### 挖掘因子 (minactors/)

以下类型的因子会被归类到 `minactors/` 文件夹：

- **计算类型**：`ml_model`
- **类别**：`ml` 或包含 `mining`、`ml_`、`statistical_`、`advanced_`、`technical_` 关键词
- **示例**：
  - `technical_mining_rsi_20.json` - 技术挖掘算法生成的RSI因子
  - `ml_momentum_5d.json` - ML动量因子
  - `adaptive_ml_factor.json` - 自适应ML因子

## 存储系统更新

### TransparentFactorStorage 类

已更新 `TransparentFactorStorage` 类以支持新的文件夹结构：

```python
class TransparentFactorStorage:
    def __init__(self, storage_dir: str = None):
        # 新的目录结构
        self.technicals_definitions_dir = self.storage_dir / "technicals" / "definitions"
        self.technicals_functions_dir = self.storage_dir / "technicals" / "functions"
        self.technicals_evaluations_dir = self.storage_dir / "technicals" / "evaluations"
        
        self.minactors_definitions_dir = self.storage_dir / "minactors" / "definitions"
        self.minactors_models_dir = self.storage_dir / "minactors" / "models"
        self.minactors_evaluations_dir = self.storage_dir / "minactors" / "evaluations"
        self.minactors_mining_history_dir = self.storage_dir / "minactors" / "mining_history"
```

### 自动分类逻辑

系统会根据以下规则自动选择存储位置：

1. **ML模型因子**：`computation_type == "ml_model"` 或 `category == "ml"`
   - 存储到 `minactors/definitions/`

2. **传统技术指标**：其他所有因子
   - 存储到 `technicals/definitions/`

### 加载逻辑

`load_factor_definition()` 方法会按以下顺序查找因子定义：

1. 先在 `minactors/definitions/` 中查找
2. 再在 `technicals/definitions/` 中查找

## 迁移统计

### 文件迁移结果

- **technicals/definitions/**: 306 个文件
- **technicals/functions/**: 162 个文件  
- **technicals/evaluations/**: 268 个文件
- **minactors/definitions/**: 30 个文件
- **minactors/models/**: 20 个文件
- **minactors/mining_history/**: 9 个文件

### 总计

- **传统技术指标**: 736 个文件
- **挖掘因子**: 59 个文件
- **总计**: 795 个文件

## 优势

1. **清晰分离**：传统技术指标和挖掘因子完全分离
2. **易于管理**：相关文件集中在一起，便于维护
3. **扩展性好**：新类型的因子可以轻松添加到相应文件夹
4. **向后兼容**：保持原有API接口不变
5. **性能优化**：减少文件查找时间

## 使用示例

### 保存因子定义

```python
from factor_miner.core.factor_storage import TransparentFactorStorage

storage = TransparentFactorStorage()

# 保存传统技术指标
storage.save_function_factor(
    factor_id="sma_20",
    name="简单移动平均线",
    category="trend",  # 会保存到 technicals/
    # ...
)

# 保存挖掘因子
storage.save_ml_factor(
    factor_id="ml_momentum_5d",
    name="ML动量因子",
    category="ml",  # 会保存到 minactors/
    # ...
)
```

### 加载因子定义

```python
# 系统会自动在正确的目录中查找
factor_def = storage.load_factor_definition("sma_20")
factor_def = storage.load_factor_definition("ml_momentum_5d")
```

## 总结

新的文件夹结构使 FactorLib 更加组织化和易于管理。传统技术指标和挖掘因子被清晰地分离，同时保持了系统的向后兼容性和高性能。这种结构为未来的扩展和维护提供了良好的基础。
