# FactorMiner 增强模式适配协议

## 目录

1. 环境检测
2. 读取顺序
3. 扩展点路由
4. 当前 V4 契约
5. 配置与执行
6. Inspector
7. 证据和修改边界

## 1. 环境检测

只有同时发现多数以下标识时，才把目录视为 FactorMiner 仓库：

- `README.md` 描述 FactorMiner；
- `core/miner/registry.py`；
- `core/miner/paradigms/base.py`；
- `core/miner/operator_runtime.py`；
- `core/commands/mine.py` 或可用的 `factorminer mine`；
- `user_workspace/`。

如果只发现同名文件夹或不完整副本，继续使用独立模式，并报告缺失项。

## 2. 读取顺序

按任务需要读取，不一次加载整个仓库：

1. `README.md`：版本、CLI、目录和结果契约；
2. `core/miner/registry.py`：注册装饰器真实签名；
3. `core/miner/operator_runtime.py`：内置算子和 AST 约束；
4. `core/miner/paradigms/base.py`：Miner 生命周期；
5. `core/startup_validation.py`：启动前验证；
6. 最接近需求的 `user_workspace` 示例；
7. 相关配置和 Inspector 实现。

仓库代码优先于文档；发现文档和代码不一致时明确指出。

## 3. 扩展点路由

### 只生成表达式或配置

当现有数据流和算子足以表达研究假设时选择。不要为了展示能力创建新 Python 文件。

### Custom Operator

仅用于缺失的数据变换。创建前确认：

- 是时序还是截面语义；
- 输入 arity；
- 返回类型、索引和形状；
- rolling 窗口和 `min_periods`；
- NaN、Inf、除零和常数序列；
- 是否在 GP/RL 搜索空间中显式启用。

### Fitness Hook

当评价偏好变化时选择，例如低换手、覆盖率、跨期稳定、复杂度惩罚。搜索机制不变。

### Custom Miner

只有候选生成、搜索状态、反馈学习或产物保存机制需要改变时选择。优先复制同范式示例。

### Inspector

当已有因子只需要跨标的、跨时期、衰减、分组收益和换手审查时选择，不创建新 Miner。

## 4. 当前 V4 契约

以下信息基于 2026-07-27 的本地 FactorMiner V4；使用前必须与当前仓库复核。

### Operator

```python
from core.miner.registry import OperatorRegistry

@OperatorRegistry.register(arity=1)
def my_operator(series):
    return series
```

- arity 只接受 `1` 或 `2`；
- 函数名是注册名；
- 返回 `pandas.Series` 或 `pandas.DataFrame`；
- 内置算子包括 `add`、`sub`、`mul`、`div`、`ts_mean`、`ts_std`；
- 通过 `search_space.allowed_operators` 启用。

### Fitness Hook

```python
from core.miner.registry import EvaluatorRegistry

@EvaluatorRegistry.register_fitness_hook("my_hook")
def my_hook(factor_values, returns, base_metrics):
    return {"fitness_score": 0.0}
```

- 必须接受 `factor_values, returns, base_metrics`；
- `base_metrics` 的核心字段为 `IC`、`RankIC`、`Turnover`；
- 返回数值，或包含 `fitness_score` 的字典；
- 通过 `fitness.hook` 引用。

不要复制旧文档中的 `register_evaluator` 写法，除非当前代码确实提供该别名。

### Custom Miner

Custom Miner 继承 `BaseFactorMiner` 并实现：

1. `initialize_search_space()`；
2. `generate_candidates()`；
3. `evaluate_candidates()`；
4. `update_model()`。

候选必须是 `FactorExpression` 子类。最终产物应进入可持久化结果路径；当前基类优先从 `hall_of_fame` 取结果，其次读取 `state.population` 或 `state.replay_buffer`。不要只更新模型而不保留候选。

### 结果指标

当前档案核心字段使用：

- `metrics.IC`
- `metrics.RankIC`
- `metrics.Turnover`
- `metrics.fitness_score`

保持大小写，不自行生成小写别名。

## 5. 配置与执行

典型配置包含：

```json
{
  "population_size": 50,
  "max_iterations": 100,
  "data_feeds": {
    "required_streams": ["close", "volume"],
    "exchange": "binance",
    "instrument_type": "futures",
    "timeframe": "5m",
    "pairs": ["BTC/USDT:USDT"],
    "mine_period": [["2024-01-01", "2024-06-01"]],
    "test_period": [["2024-06-02", "2024-12-31"]],
    "mining_mode": "sequential_single"
  },
  "search_space": {
    "allowed_operators": ["add", "sub", "mul", "div", "ts_mean", "ts_std"]
  },
  "fitness": {
    "hook": "hook_name"
  },
  "storage": {
    "db_root": "factor_db"
  }
}
```

不要机械复制日期、币种和 hook；按研究任务生成。FactorMiner 当前主要使用 `mine_period` 和 `test_period`，但研究设计仍应区分训练、候选选择用验证集和最终留出集。若配置接口不能原生表达三段切分，使用独立配置、实验工具或分阶段运行，并准确命名各时期角色。

执行入口通常为：

```bash
factorminer mine --miner GP --config user_workspace/configs/example.json
factorminer mine --miner MyCustomGP --config user_workspace/configs/example.json --user-dir user_workspace
```

若 CLI 未安装，可用：

```bash
python -m core.cli mine --miner MyCustomGP --config user_workspace/configs/example.json --user-dir user_workspace
```

运行前：

- 验证依赖和命令是否可用；
- 检查数据覆盖区间；
- 检查配置、注册名和导入；
- 估计实验成本；
- 不输出或记录 API 密钥；
- 先做小规模 smoke test，再做完整实验。

## 6. Inspector

Inspector 可从 factor ID、AST 或代码解析候选。常见入口：

```bash
factorminer inspect --factor "fac_id" --pairs "BTC/USDT:USDT"
factorminer inspect --ast "{...}" --pairs "BTC/USDT:USDT,ETH/USDT:USDT" \
  --start 2025-08-01 --end 2025-12-31 --timeframe 5m
```

审查时明确：

- 使用的标的、时期、频率和交易类型；
- 该时期是验证集还是最终留出集；
- Coverage、IC、RankIC、IR、Decay、Quantiles 和 Turnover；
- 挖掘评分与 Inspector 复评是否一致；
- 缺少快照或数据时不得补造图表或指标。

## 7. 证据和修改边界

默认允许建议或修改：

```text
user_workspace/
├── configs/
├── custom_operators/
├── custom_fitness/
├── custom_miners/
└── experiment_tools/
```

未经用户明确要求，不主动修改：

```text
core/
api/
web/
```

每次增强模式交付记录：

- 检测到的仓库版本或证据；
- 读取的契约文件；
- 修改文件；
- 执行入口和退出状态；
- 真实产物路径；
- 实际指标；
- 失败、警告和未验证项。
