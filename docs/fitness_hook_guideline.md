# FactorMiner Fitness Hook 编写与指标口径

先按[产品指南](FactorMiner_Product_Report_ZH.md)安装源码环境并准备本地行情。本指南说明当前评分接口；自定义 Hook 替换默认评分，不会自动继承覆盖率惩罚。

## 默认评分与 Coverage

统一评估器同时保留 Pearson IC、Spearman RankIC 和 Turnover。默认 fitness 为 `abs(RankIC) × 100`，输出覆盖低于 20% 时乘 `(coverage / 0.20) ** 2`。负 RankIC 也可得到高分，必须检查信号方向；惩罚不能证明没有过拟合。

以下指标不要混用：

| 入口 | 当前定义 |
| --- | --- |
| 默认评估器 Coverage | Series 为非 NaN 输出数 / 总长度；DataFrame 先丢弃含缺失值的整行，再统计剩余单元 / 原矩阵单元数；不额外与标签求共同有效掩码 |
| CLI Inspector Coverage | 因子与标签共同有效观测 / 总观测；跨资产按矩阵单元统计 |
| Data Center 覆盖表 | 文件起止时间与行数，不是因子覆盖或逐根完整性证明 |
| 统一评估器 Turnover | 因子沿时间的平均绝对差分，截面再对资产平均；不是账户成交百分比 |
| CLI 单标的换手诊断 | 共同有效因子序列的平均绝对差分 / 平均绝对值；与上行原始单位不同 |

CLI Inspector 的 IR 固定乘 `sqrt(252)`，未按 K 线周期/加密资产日历校准。正 IC 比例不是交易胜率。缺失、零值、负分与未计算应分别解释。

## 注册 Hook

将下例保存为 `user_workspace/custom_fitness/coverage_rankic.py`。加载器只扫描工作区的 `custom_miners`、`custom_operators`、`custom_fitness` 子目录，不扫描工作区根目录的任意文件。

```python
from core.miner.registry import EvaluatorRegistry

@EvaluatorRegistry.register_fitness_hook("coverage_rankic")
def coverage_rankic(factor_values, returns, base_metrics):
    # factor_values / returns 是索引和资产列对齐的 Series 或 DataFrame。
    # returns 的 horizon/入场/收益类型由 target 配置决定。
    # 此例明确沿用默认评分的输出 Coverage，未改成共同有效口径。
    valid_count = factor_values.dropna().count()
    if hasattr(valid_count, "sum"):
        valid_count = valid_count.sum()
    total_count = factor_values.size
    coverage = float(valid_count / total_count) if total_count else 0.0
    penalty = (coverage / 0.20) ** 2 if coverage < 0.20 else 1.0
    score = abs(float(base_metrics["RankIC"])) * 100.0 * penalty
    return {
        "fitness_score": score,
        "coverage": coverage,
        "coverage_penalty": penalty,
    }
```

Hook 必须接收 `factor_values, returns, base_metrics`，可返回有限数值或包含有限 `fitness_score` 的字典。零分、负分有效；缺失/NaN/Infinity 主分数使当前候选失败，其他候选继续。额外诊断原值保留在档案，HTTP/WS 传输的非有限值转成 `null`，表示不可用。

`base_metrics` 含 `IC`、`RankIC`、`Turnover`。若要加入因子变化惩罚，应考虑因子的数值单位和尺度，不能把 `Turnover=0.8` 自动解释为账户换手 80%。

## 完整示例配置

先按产品指南准备 BTC 永续 1h 数据，再将下列 JSON 保存为新文件 `user_workspace/configs/fitness_example.json`，保留已有实验配置：

```json
{
  "population_size": 12,
  "max_iterations": 2,
  "data_feeds": {
    "exchange": "binance",
    "instrument_type": "futures",
    "timeframe": "1h",
    "pairs": ["BTC/USDT:USDT"],
    "required_streams": ["close", "volume"],
    "mine_period": [["2024-01-01 00:00:00", "2024-01-07 23:00:00"]],
    "test_period": [["2024-01-08 00:00:00", "2024-01-10 23:00:00"]],
    "mining_mode": "sequential_single"
  },
  "search_space": {
    "allowed_operators": ["add", "sub", "mul", "div", "ts_mean", "ts_std"]
  },
  "fitness": {"hook": "coverage_rankic"},
  "evaluation": {"max_workers": 2}
}
```

从源码根目录执行：

```bash
factorminer mine --miner MyCustomGP \
  --config user_workspace/configs/fitness_example.json \
  --user-dir user_workspace --iterations 1
```

命令行的 `--miner` 决定算法，不使用 `miner` 配置字段。Hook 注册名未知或任一扩展导入失败时，启动在读行情之前拒绝；先解决依赖/命名错误，不以默认评分替代。全工作区加载要求所有扩展可导入。

GP 的上述评分使用挖掘区间；填写 test_period 不会自动为它完成独立测试。后续用 Inspector 复评另一段数据，并保留未参与挑选的最终测试。标签、覆盖和评分公式改变时应作为不同实验记录。
