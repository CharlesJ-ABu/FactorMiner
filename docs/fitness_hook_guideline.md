# 📘 FactorMiner 因子评估 Fitness Hook 编写与对齐指南

为了防止遗传编程（GP）或大语言模型（LLM）算出来的因子产生 **“虚高分数（如 Fitness Score 30+，但真实覆盖率 5% 且 RankIC 为负）”** 的过拟合假象，本指南将介绍如何将 GP 挖矿引擎与 Factor Inspector 审查逻辑进行精准对齐，并提供编写 **自定义 Fitness Hook (自定义打分钩子)** 的标准范例。

---

## 核心对齐改进 (Engine Alignments)

针对之前挖矿评估与 Inspector 审查不对齐的问题，底层 `ParallelEvaluator` 已经完成了如下对齐：

1. **评估指标转为 RankIC (Spearman 秩相关)**：
   - 默认放弃易受离群点干扰的 Pearson IC 线性相关，全面改用 **Spearman RankIC (秩相关)** 作为基础相关性评估指标。
2. **加入数据覆盖率惩罚项 (Data Coverage Penalty)**：
   - 若因子的非 NaN 有效数据覆盖率低于 20%，系统会自动按二次方惩罚项 `(coverage / 0.20)^2` 进行衰减扣分，彻底杜绝数据稀疏因子的虚高评分。

---

## 🛠️ 如何编写自定义 Fitness Hook

你可以随时在 `user_workspace/` 目录下创建任意 Python 文件（例如 `user_workspace/my_fitness.py`），使用 `@register_evaluator("hook_name")` 装饰器注册你专属的 Fitness 打分逻辑。

### 1. 标准 Fitness Hook 签名与输入参数

```python
from core.miner.registry import register_evaluator

@register_evaluator("custom_sharpe_rankic_hook")
def custom_sharpe_rankic_hook(factor_values, returns, base_metrics):
    """
    输入参数说明：
    - factor_values: pd.Series 或 pd.DataFrame，因子算出的原始数值序列
    - returns: pd.Series 或 pd.DataFrame，未来一期的收益率序列
    - base_metrics: dict，系统自动预先计算的基础指标：
        * base_metrics["IC"]: Pearson 线性相关
        * base_metrics["RankIC"]: Spearman 秩相关
        * base_metrics["Turnover"]: 单期因子换手率

    返回值说明：
    - 可以返回一个浮点数（即 Fitness Score）
    - 或返回一个 dict，包含 "fitness_score" 以及你想记录在元数据中的其他自定义指标
    """
    # 提取基础指标
    rank_ic = base_metrics.get("RankIC", 0.0)
    turnover = base_metrics.get("Turnover", 0.0)
    
    # 1. 计算数据覆盖率 (Coverage)
    total_count = factor_values.size
    valid_count = factor_values.dropna().count()
    if hasattr(valid_count, 'sum'):
        valid_count = valid_count.sum()
    coverage = valid_count / total_count if total_count > 0 else 0.0

    # 2. 覆盖率惩罚（覆盖率低于 20% 时惩罚）
    coverage_penalty = (coverage / 0.20) ** 2 if coverage < 0.20 else 1.0

    # 3. 换手率惩罚（换手率高于 80% 时惩罚）
    turnover_penalty = 1.0 / (1.0 + max(0.0, turnover - 0.8))

    # 4. 最终复合 Fitness 得分
    fitness_score = abs(rank_ic) * 100.0 * coverage_penalty * turnover_penalty

    return {
        "fitness_score": float(fitness_score),
        "coverage": float(coverage),
        "coverage_penalty": float(coverage_penalty),
        "turnover_penalty": float(turnover_penalty)
    }
```

---

## ⚙️ 如何在 `config.json` 中配置并激活你的 Fitness Hook

在你的挖矿配置文件 `config.json` 中，通过 `"fitness": { "hook": "your_hook_name" }` 即可激活：

```json
{
  "miner": "AdvancedSampleGP",
  "fitness": {
    "hook": "custom_sharpe_rankic_hook"
  },
  "data_feeds": {
    "exchange": "binance",
    "instrument_type": "futures",
    "timeframe": "5m",
    "pairs": ["BTC/USDT:USDT"]
  }
}
```

启动挖矿时：
```bash
factorminer mine --miner AdvancedSampleGP --config config.json
```
系统会在动态加载 `user_workspace/` 时自动识别该装饰器并用其替换默认打分逻辑！
