# 📚 FactorMiner 因子库（V3架构）

## 🎯 设计目标
- 因子=算法（不存储因子数值）
- 计算逻辑完全透明、可读、可审计
- 统一目录、统一引擎、统一调用方式

## 📁 目录结构
```
factorlib/
└── factors/
    ├── definitions/   # 因子定义（JSON）
    ├── formulas/      # 公式（.txt，可读）
    ├── functions/     # 函数（.py，可读）
    ├── pipelines/     # ML流水线（.json，可读）
    ├── evaluations/   # 评估结果（JSON，可选）
    └── temp/          # 临时缓存（可清理）
```

## 🧠 因子定义（definitions/*.json）
示例：
```json
{
  "factor_id": "sma_v3",
  "name": "透明SMA",
  "category": "trend",
  "computation_type": "formula",
  "computation_data": {
    "formula_file": "formulas/sma_v3.txt",
    "formula": "close.rolling(window=period).mean()"
  },
  "parameters": {"period": 20}
}
```

## 🔢 公式类（formulas/*.txt）
```
# 计算简单移动平均线
# 参数: period
close.rolling(window=period).mean()
```

## 🧩 函数类（functions/*.py）
```python
def calculate(data, fast_period=12, slow_period=26, signal_period=9):
    fast = data['close'].ewm(span=fast_period).mean()
    slow = data['close'].ewm(span=slow_period).mean()
    return fast - slow
```

## 🤖 流水线类（pipelines/*.json）
- 特征工程代码（可读）
- 模型配置（算法与参数）
- 后处理代码（可读）

## ⚙️ 调用方式
```python
from factor_miner.core.factor_engine import get_global_engine
engine = get_global_engine()
value_series = engine.compute_single_factor('sma_v3', data)
```

## ✅ 原则
- 不需要导入任何"注册器"或"实时引擎"；一切来自 JSON
- 因子逻辑以文本/源码保存，严禁二进制黑盒
- 统一使用 `factor_miner.core.factor_engine` 与 `factor_miner.core.factor_storage`

---

*最后更新: 2025年9月*
