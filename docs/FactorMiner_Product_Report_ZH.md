# FactorMiner V4 产品说明与安装配置指南

版本：4.0.0｜更新日期：2026 年 11 月 25 日｜适用对象：量化研究员、算法工程师及项目维护人员

FactorMiner 是一个在本地运行的量化因子研究工作台。它把历史行情、因子生成、统一评分、结果存档和人工审查连接起来，让研究人员能够用配置文件发起实验，并追溯候选因子的逻辑、数据来源和评估结果。

当前主要数据链路面向加密资产 OHLCV 行情，提供 Python CLI、FastAPI 后端和 React Web 界面。发布范围是从源码 checkout 运行的本机单用户研究环境。本文描述已实现能力；需求文档中的规划项单独标注。

## 阅读顺序

- 了解产品：阅读“产品功能”和“挖掘方法”。
- 第一次使用：依次完成“安装”“启动”“首次研究流程”。
- 定制研究：阅读“配置说明”“LLM 配置”和“用户扩展”。
- 排查问题或交接项目：阅读“文件管理”“常见问题”“验证与边界”。

## 产品功能

研究流程如下：

```mermaid
flowchart LR
    A[下载或准备本地行情] --> B[配置资产与标签]
    B --> C[选择挖掘方法]
    C --> D[生成和评估候选]
    D --> E[保存因子与数据快照]
    E --> F[Inspector 审查与比较]
    F --> G[调整研究配置]
    G --> C
```

| 功能 | 当前可以做什么 | 使用入口 |
| --- | --- | --- |
| 历史行情管理 | 下载、分页、合并、补缺、健康检查和 Feather 保存；检查本地数据覆盖范围 | Data Center、`factorminer download` |
| 多标的研究 | 逐个标的独立挖掘，或将多个标的对齐成截面矩阵共同计算 | `data_feeds.mining_mode` |
| 多种候选生成方法 | 用公式搜索、概率策略更新、神经网络或 LLM 生成候选 | Launchpad、`factorminer mine` |
| 统一标签 | 配置未来收益的入场价格、持有根数及简单或对数收益 | 顶层 `target` |
| 指标与自定义评分 | 计算 IC、RankIC、因子值变化指标；可接入 Fitness Hook | 统一评估器 |
| 任务监控 | 显示任务状态、日志、进度和评估散点；连接恢复后重放内存中的任务状态 | Launchpad、WebSocket |
| 研究概览 | 查看因子库存、Miner 分布、生命周期标记、近期任务和高分候选 | Dashboard |
| 因子审查 | 查看逻辑、指标、来源、数据血缘、滚动 IC、分组收益和换手诊断；比较 2–5 个有快照的因子 | Web Inspector |
| 命令行复评 | 按因子 ID、AST 或 Python 因子代码，在指定标的和区间重新计算指标 | `factorminer inspect` |
| 本地存档 | 保存元数据、公式或源码、因子值与未来收益快照，以及可恢复的 NN 工件 | `factor_db/` |
| 用户扩展 | 注册 Miner、单目或双目算子、Fitness Hook 和模型加载器 | `user_workspace/` |
| 研究辅助 Skill | 将研究想法整理为假设、特征、标签、数据切分和评价方案 | [FactorMiner Skills](../skills/README.md) |

### 四个 Web 页面

| 页面 | 路径 | 主要操作 |
| --- | --- | --- |
| Dashboard | `/` | 查看研究概况，进入因子详情或任务列表，再打开任务详情 |
| Mining Launchpad | `/launchpad` | 选择已注册 Miner 和 JSON 配置，启动挖掘，查看状态、日志和结果 |
| Factor Inspector | `/inspector` | 查看因子档案、分析快照、比较候选，单个或批量更新生命周期 |
| Data Center | `/data` | 选择交易所、市场、交易对、周期和日期，检查覆盖并下载 |

当前 Data Center 提供 Binance 和 OKX 选项，CLI 可指定 CCXT 支持的交易所名称；具体标的、市场和周期仍取决于交易所接口及下载器支持范围。优先使用已覆盖的固定时长周期，例如 1m、5m、15m、1h、4h、1d。

Web 的交易对和周期可多选，市场类型单选；一批下载组合在后台顺序执行，不同任务可以并行。覆盖表显示本地文件的起止时间和记录数，不能单凭这个范围认定区间内没有缺口。Launchpad 的散点记录每轮最佳候选的 IC 和可取得的实际复杂度，颜色表示复杂度，不表示谱系或交易收益。

界面提供中文、英文、德文切换，已覆盖全局导航、Dashboard 与 Inspector；部分页面和日志仍保留原语言。研究工件中的因子 ID、表达式、交易对与指标名保持原样。

## 挖掘方法及适用场景

| 方法 | 建议入口 | 当前实现与适用场景 |
| --- | --- | --- |
| GP 公式搜索 | `MyCustomGP` | 随机公式树、变异与精英保留，适合搜索可读公式；内置 `GP` 提供较简单的公式树搜索 |
| RL 参考方法 | `MyCustomRL` | 根据候选评分更新算子和字段的采样权重，适合研究搜索偏好如何变化 |
| NN 神经网络 | `NN` 或 `MyCustomNN` | 默认采用 NumPy MLP，训练模型并将输出通道保存为独立因子，保存权重及标准化参数 |
| LLM 代码生成 | `MyCustomLLM` | 调用兼容 Chat Completions 的接口生成因子代码，根据反馈继续搜索，并可记录完整实验过程 |

需要区分几个名称：

- 内置 `RL` 目前仍是随机轨迹和策略更新占位骨架；实际参考实验使用 `MyCustomRL`，不能将它描述为完整 Actor-Critic 系统。
- `NN` 和 `MyCustomNN` 注册到同一个参考实现。旧 `DL` 名称已弃用，新配置使用 `NN`。
- `MyCustomGP` 当前主要使用变异和精英保留，不应据此宣称已经实现完整的多种交叉策略。
- `AdvancedSampleGP`、`MyTemporalNN` 及工作区中的其他研究模型属于扩展入口，应按各自实现配置参数；本指南的入门流程使用 `MyCustomGP`。

### 如何理解评分

`IC` 是因子值与未来收益的 Pearson 相关；`RankIC` 是 Spearman 秩相关。顺序单标的计算时间序列相关，跨资产模式计算每期截面相关后取平均。

`Turnover` 在当前统一评估器中是因子值沿时间的平均绝对变化，不等同于真实交易账户的成交换手率。默认 `fitness_score` 使用 `abs(RankIC) × 100`，并在因子覆盖率低于 20% 时施加二次惩罚；负 RankIC 也可能得到较高分数，应同时检查因子方向。

Fitness Hook 可以替换默认评分。有限零分和负分有效；主评分为 NaN、Infinity 或缺失时，该候选失败。额外诊断值在档案中保留原值，HTTP/WS 传输中的非有限值表示为 `null`，页面显示不可用。

CLI Inspector 的 IR 当前使用固定 `sqrt(252)` 缩放，未按 K 线周期或加密资产交易日历校准；t-stat 和正 IC 比例是相关性诊断，不是交易胜率或独立显著性结论。默认衰减分析使用 Lag 1、2、3、5、10。分组平均未来收益也不等于含手续费和滑点的策略净值。

### 训练区间与测试区间

`mine_period` 定义挖掘或训练数据，`test_period` 提供另一段数据，但仅填写后者不等于所有 Miner 都自动完成独立测试：

- GP、参考 RL 和普通 LLM 的统一评估默认使用挖掘区间；应另用 Inspector 或相应研究工具复评后续区间。
- `MyCustomNN` 在挖掘区间训练，有测试数据时用测试区间筛选通道；没有测试数据时使用训练区间。因为测试区间参与筛选，它在研究设计上承担验证集作用，最终结论仍需另外保留未参与选择的数据。
- 标签按每个请求区间独立构造；末端没有足够未来价格的样本保留为缺失，不借用下一段区间的价格。

## 安装

### 环境准备

| 项目 | 本指南采用的配置 |
| --- | --- |
| Python | 3.10 及以上；已实测 3.12，以下命令采用 3.12 |
| Node.js | 22.12 及以上，使用随 Node 提供的 npm |
| 运行系统 | 以下命令适用于 macOS 或 Linux 的 shell；现有验证来自 macOS |
| GPU | 入门 GP、参考 RL 和默认 NumPy NN 不要求 GPU |
| 网络 | 安装依赖、下载行情及真实 LLM 调用需要对应服务可达；已有完整本地数据的普通因子计算可离线运行 |

内存和磁盘需求由资产数、K 线数量、模型及并发决定。首次使用先选一个资产、1h 周期、小种群和少量迭代，再扩大规模。

### 安装后端与命令行入口

先获取源码并进入项目根目录；已有 checkout 时直接进入其目录：

```bash
git clone https://github.com/CharlesJ-ABu/FactorMiner.git
cd FactorMiner
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
factorminer --help
```

`.[test]` 安装核心运行与测试依赖；只使用产品时可执行 `python -m pip install -e .`。`requirements.txt` 保留较完整的扩展/研究依赖集合，需要这些工具时也可采用下列方式，再按需安装测试额外依赖：

```bash
python -m pip install -r requirements.txt
python -m pip install -e '.[test]'
```

wheel 提供 Python 引擎、CLI 和 API 模块；工作区、配置示例及前端来自源码 checkout，不随 wheel 提供完整运行站点。当前操作流程始终从源码根目录启动。Python 依赖为兼容范围约束，精确环境可能随解析结果变化；迁移研究时保留所用源码版本、依赖版本与输入数据。

### 安装前端

在项目根目录执行：

```bash
cd web
npm ci
```

仓库包含 `package-lock.json`，使用 `npm ci` 按锁文件安装。仅使用 CLI 时可以不安装或启动前端。

### 扩展依赖

默认 NumPy NN 不需要 PyTorch。若自行加入的模型直接导入 `torch`，需要在同一 Python 环境额外安装：

```bash
python -m pip install torch
```

动态加载会扫描工作区的 `custom_miners`、`custom_operators`、`custom_fitness` 下的 Python 文件；其中任一导入失败会阻止挖掘启动。新增扩展应自行声明额外依赖。首次 CLI 流程下方使用只含 GP 的独立扩展目录；Web 使用完整 `user_workspace`，应在后端启动后确认 `/api/miners` 返回 `load_errors: []`。没有选择某个 Miner，也仍需满足被扫描模块的导入依赖。

## 启动服务

### 终端一启动后端

在源码根目录执行：

```bash
source .venv/bin/activate
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

浏览器打开 `http://localhost:8000/docs` 可查看 REST API。健康检查：

```bash
curl http://localhost:8000/api/health
```

正常响应为：

```json
{"status":"ok","engine":"FactorMinerDirector"}
```

该响应只表明 API 能响应，不代表行情、Miner、LLM 凭证均已就绪。开发时可以加 `--reload`；代码变动引起的服务重启会中断当前内存任务。

### 终端二启动前端

新开终端并进入源码根目录，然后执行：

```bash
cd web
npm run dev -- --host 127.0.0.1 --port 5173 --strictPort
```

打开 `http://localhost:5173`。当前前端使用固定的 `http://localhost:8000` 和 `ws://localhost:8000/ws/monitor`，请先保持这些端口。远程访问需要一并处理前端 API/WS 地址、代理和访问控制；当前安装流程面向本机使用。

前端构建命令为 `npm run build`，产物位于 `web/dist/`；`npm run preview` 用于本地检查构建产物。构建成功本身不等于完成了远程服务部署。

## 首次研究流程

下面以 BTC 永续、1h K 线、两轮 GP 为例。它用于熟悉操作和产物，不预设任何因子会取得有效收益。

### 下载行情

回到项目根目录并激活 Python 环境：

```bash
factorminer download \
  --exchange binance \
  --symbols 'BTC/USDT:USDT' \
  --timeframes 1h \
  --type futures \
  --start 2024-01-01 \
  --end 2024-01-11 \
  --mode full
```

下载范围为 UTC 的 `[start, end)`，本例不包含 1 月 11 日。完整区间应有 240 根 1h K 线，文件为：

```text
data/binance/futures/BTC_USDT_USDT-1h-futures.feather
```

现货使用 `BTC/USDT` 和 `--type spot`。名称中的 `/` 和 `:` 均转为 `_`；目录同时区分交易所与市场类型。

CLI 的 `full` 和 `incremental` 当前都进入合并保存流程，应明确指定所需日期，不把 `incremental` 理解为自动定位最后一根并续传。`overwrite` 会替换旧文件区间，仅在确实需要重建时使用。Web 另外提供补缺模式。

### 保存入门配置

将下列内容保存为 `user_workspace/configs/product_quickstart.json`。该文件名是本指南示例，需要自行创建；不要覆盖已有实验配置。

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
  "target": {
    "type": "forward_return",
    "entry_price": "current_close",
    "exit_price": "close",
    "horizon_bars": 1,
    "return_type": "simple"
  },
  "search_space": {
    "allowed_operators": ["add", "sub", "mul", "div", "ts_mean", "ts_std"]
  },
  "evaluation": {
    "max_workers": 2
  }
}
```

挖掘区间包含首尾 K 线，因此这里显式写出末日的 `23:00:00`。只写 `2024-01-07` 表示当天 00:00，不表示整天。示例省略 `fitness.hook`，使用默认评分。

### 启动挖掘

首次使用 CLI，先在项目根目录准备只含参考 GP 的扩展目录。示例配置已明确使用内置算子，因此不需要复制其他算子或评分文件：

```bash
mkdir -p user_workspace/quickstart_extensions/custom_miners
cp user_workspace/custom_miners/my_custom_gp.py user_workspace/quickstart_extensions/custom_miners/
```

然后执行：

```bash
factorminer mine \
  --miner MyCustomGP \
  --config user_workspace/configs/product_quickstart.json \
  --user-dir user_workspace/quickstart_extensions
```

Web：确认完整工作区的 `load_errors` 为空后，刷新 Launchpad，选择 `MyCustomGP` 和 `product_quickstart.json`，点击启动，进入任务查看日志和结果。Web 固定读取 `user_workspace/configs/` 并加载完整工作区，不使用上述 CLI 扩展目录；它是配置选择入口，修改参数仍需编辑 JSON 文件。

CLI 的 `--miner` 或 Web 中选择的 Miner 决定实际算法，并覆盖配置中的 `paradigm`；CLI 的 `--iterations` 可以临时覆盖 `max_iterations`。

### 查看并复评结果

成功结果会给出 `fac_...` 因子 ID。进入 Inspector 查看逻辑、原始指标和快照。任务显示完成并不等于发现了有效因子；`completed_empty` 表示执行结束但没有可归档结果。

将下面的 `fac_REPLACE_WITH_REAL_ID` 替换为本次实际 ID，再复评后续区间：

```bash
factorminer inspect \
  --factor fac_REPLACE_WITH_REAL_ID \
  --config user_workspace/configs/product_quickstart.json \
  --pairs 'BTC/USDT:USDT' \
  --timeframe 1h \
  --start '2024-01-08 00:00:00' \
  --end '2024-01-10 23:00:00' \
  --user-dir user_workspace/quickstart_extensions
```

同时提供 `--start` 和 `--end` 才覆盖配置区间。CLI 也支持 `--ast` 和 `--code` 输入；同一次使用选择一种因子来源。跨资产 NN 的 CLI 复评当前会明确拒绝，可查看 Web 已存快照，或通过 Python Resolver 提供模型需要的完整矩阵。

## 配置说明

### 通用参数

| 字段 | 含义与注意事项 |
| --- | --- |
| `population_size` | 候选生成规模，具体含义由 Miner 定义；NN 不应按“一个模型等于一个因子”理解 |
| `max_iterations` | 外层搜索或训练轮数；CLI 未提供该字段及覆盖参数时使用 5 |
| `data_feeds.exchange` | 行情来源交易所，必须与本地数据目录一致 |
| `instrument_type` | 市场类型，入门使用 `spot` 或 `futures` |
| `timeframe` | K 线周期，必须与文件名一致 |
| `pairs` | 真实研究资产列表，永续使用带结算币种的 CCXT symbol |
| `required_streams` | 参考 Miner 的候选终端、模型特征或 LLM 提示字段，例如 close、volume；它不是全局数据访问白名单，读取器仍可保留其他行情列 |
| `mine_period` / `test_period` | 二维区间列表，时间按 UTC 理解，切片含末端 K 线 |
| `mining_mode` | 只允许 `sequential_single` 或 `cross_asset`；省略时默认前者，错误拼写会失败 |
| `search_space.allowed_operators` | 内置或已注册算子名称；未知名称在启动时拒绝 |
| `fitness.hook` | 已注册的评分函数名；省略使用默认评分，拼错不会静默回退 |
| `evaluation.max_workers` | 候选评估线程数，默认 8，可设置 1–64；首次运行可设 2 |

`sequential_single` 会逐个标的运行，适合单资产时序因子；`cross_asset` 会将资产对齐后共同运行，要求算子、评分函数和模型都能处理时间 × 资产的数据。切换模式会改变研究对象和相关性的含义，不只是提高并行度。

### 未来收益标签

| 字段 | 当前支持 |
| --- | --- |
| `type` | `forward_return` |
| `entry_price` | `current_close`、`next_open` |
| `exit_price` | `close` |
| `horizon_bars` | 正整数，单位为 K 线根数 |
| `return_type` | `simple`、`log` |

默认标签为 `close[t+1] / close[t] - 1`。若用 `next_open`、`horizon_bars=3`、简单收益，公式为 `close[t+3] / open[t+1] - 1`；行情必须有 `open` 列，但不要求把它加入模型输入特征。标签定义随档案保存，Inspector 可沿用或显式覆盖；不同标签的结果不要混为同一实验。

### NN 配置

可以从 `user_workspace/configs/configNN.json` 复制配置并调整数据区间。主要参数有 `hidden_dim`、`learning_rate`、`random_seed`、`nn_training_epochs`、`nn_min_samples`、`nn_l2`、`nn_diversity_penalty`、`nn_gradient_clip`、`nn_early_stopping_patience`、`top_k_factors` 与 `max_corr`。

`nn_training_epochs` 是每次外层迭代的最多训练轮数，可依据训练损失提前停止，和 `max_iterations` 不是同一预算；`hidden_dim` 决定可物化通道的规模。默认模型保存在 `factor_db/models/`，通道元数据保存在 `factor_db/metadata/`，恢复时需要完整工件及相同输入定义。

确认配置中的数据已准备后，可运行：

```bash
factorminer mine --miner NN --config user_workspace/configs/configNN.json --user-dir user_workspace
```

### LLM 配置

先完成前面的核心依赖安装；接口客户端需要 `aiohttp`。

参考配置为 `user_workspace/configs/configLLM_experiment.json`。它从环境读取 `AI_API_KEY` 或 `OPENAI_API_KEY`，允许 `AI_MODEL`、`AI_API_BASE` 覆盖模型和服务地址。

在启动 CLI 或后端的同一终端注入环境变量。以下模型和地址必须替换为实际服务配置；`read` 会等待输入密钥且不回显：

```bash
read -r -s AI_API_KEY
export AI_API_KEY
export AI_MODEL='替换为已开通的模型标识'
export AI_API_BASE='https://替换为服务地址/v1'
```

当前实现从进程环境读取这些值；仅创建 `.env` 文件不会自动加载。使用 Web 时，在设置环境后重新启动后端。

真实实验建议保留 `experiment.require_live_api=true` 和 `experiment.allow_fallback=false`：凭证缺失时拒绝启动，失败请求不生成回退候选，整批没有可用回答时任务报错；同批其他有效回答仍可继续处理。`max_concurrency`、`max_retries`、`timeout_seconds` 控制请求行为；模型费用由实际服务商和调用量决定。

确认模板的数据区间、模型和调用预算后，启动命令为：

```bash
factorminer mine --miner MyCustomLLM --config user_workspace/configs/configLLM_experiment.json --user-dir user_workspace
```

`llm_sandbox` 默认墙钟超时 5 秒、累计 CPU 预算 3 秒、内存配置 1024 MiB；资源限制的实际支持依赖操作系统。生成代码受到语法/API 白名单和独立进程执行限制。自定义扩展文件在主进程导入，仍应只加载可信代码。

启用 `experiment.record_dir` 后，可保存请求、原始回答、候选结果和复评记录。详细使用见[自定义 LLM 与实验留痕指南](guides/custom_llm_experiments.md)。Prompt 和回答可能包含研究内容，应按自己的研究数据管理要求保存。

## 用户扩展

扩展放在专门目录，不需要修改核心调度流程：

| 扩展 | 目录 | 当前注册方式 |
| --- | --- | --- |
| Miner | `user_workspace/custom_miners/` | 继承 `BaseFactorMiner`，使用 `@MinerRegistry.register("名称")` |
| 算子 | `user_workspace/custom_operators/` | 使用 `@OperatorRegistry.register(arity=1)`，双目算子设为 `2` |
| 评分 | `user_workspace/custom_fitness/` | 使用 `@EvaluatorRegistry.register_fitness_hook("名称")` |
| 可恢复 NN 模型 | 自定义模型模块 | 使用 `register_nn_model_loader(model_format, loader)` |

Fitness Hook 签名为 `hook(factor_values, returns, base_metrics)`，返回有限数值或含有限 `fitness_score` 的字典。它替换默认评分；需要覆盖率或其他惩罚时应明确写入自己的函数。

算子和因子输出必须保持输入的索引、资产列与维度，不应偷偷排序、填零或改变标签。普通代码因子返回 `pandas.Series` 或 `pandas.DataFrame`。模型扩展需要同时考虑训练、预测通道和工件恢复，只有保存权重文件不代表能在另一环境复现。

## 文件管理与运行方式

| 路径 | 保存内容 |
| --- | --- |
| `data/<exchange>/<market>/` | 原始/合并后的 Feather 行情 |
| `user_workspace/configs/` | JSON 研究配置 |
| `config/user_config/config.json` | 首次实际保存时创建的交易所账户配置，可能包含凭证 |
| `factor_db/metadata/` | 因子 ID、逻辑引用、指标、生命周期和数据血缘 |
| `factor_db/values/` | 对齐后的因子值及未来收益 parquet 快照 |
| `factor_db/sources/` | 代码类因子的源码及相关记录 |
| `factor_db/models/` | 可恢复的 NN 模型工件 |
| `factor_db/weights/` | 旧版或特定实现的权重文件 |
| `user_workspace/experiments/` | 启用 Recorder 后的额外研究过程文件 |

后端和 CLI 均应从项目根目录启动，因为数据和存档路径相对当前工作目录解析。当前公共入口的 `get_global_storage()` 使用本地 `factor_db/`；部分配置虽然包含 `storage.db_root`，该字段尚未贯通公共入口，不能把修改它当成已实现的换库功能。

元数据保留 Miner 声明的父代引用及 LLM 模型来源。父代引用不保证都是可解析的已入库 Factor ID；这些记录也不等于完整搜索状态或 RL 策略 checkpoint。恢复 NN 依赖完整模型工件、加载器和相同输入定义。

交易所账户配置始终写入运行工作目录，wheel 安装也不会写入 site-packages；采用临时私有文件和原子替换，POSIX 新文件权限为 0600。含凭证的配置应单独保护，不纳入公开源码。

备份时一并保留研究配置、扩展代码、行情及整个 `factor_db/`。只拷贝元数据可能失去源码、模型或快照。服务重启会丢失内存任务列表和下载状态，但不会删除已落盘的因子档案。

当前适合单后端进程运行。同一进程的下载线程按文件路径协调合并；多个进程或多台机器同时写同一数据目录不在该保护范围内。

某候选快照失败时保留可诊断的元数据，`pending` / `failed` 档案不可分析；同批其他候选继续，已完成的档案保留。CLI 以失败退出，本机日志列出成功和失败 ID；Web 报失败，公开错误做脱敏。旧档案按已有快照读取。

生命周期值包括 `DISCOVERED`、`INSPECTED`、`PAPER_TRADING`、`LIVE`、`RETIRED`。这些是人工管理标记，修改标签不会启动模拟账户或真实交易。

## 常见问题

| 现象 | 优先检查 |
| --- | --- |
| `factorminer: command not found` | 激活正确环境并执行 `python -m pip install -e .`；也可用 `python -m core.cli` |
| Web 页面能开但没有数据 | 后端 8000 是否可达；前端固定地址是否仍匹配；查看浏览器请求错误 |
| Miner 列表缺项或启动导入失败 | 查看 `/api/miners` 的 `load_errors`；检查扩展目录和额外依赖 |
| 找不到行情或触发自动下载 | 检查工作目录、交易所、市场、CCXT symbol、周期和文件命名是否一致 |
| 数据看似存在但区间为空 | 核对 UTC、日期末端、请求范围以及文件实际覆盖时间 |
| 下载 100% 但任务失败 | 按 completed / failed / partial_failed 终态和每项错误判断，不以百分比判断成功 |
| 重启后任务记录消失 | 任务只存内存；已归档因子到 Inspector 中查看 |
| Inspector 没有分析图 | 确认因子有 values 快照；旧档案不会自动补造历史曲线 |
| NN 旧模型恢复失败 | 核对工件、model_format 注册和特征定义；旧裸权重不能猜测恢复 |
| LLM 不发请求或报凭证错误 | 检查同一进程环境、占位密钥、aiohttp、模型权限、服务地址及实验模式 |
| 大量评分为零 | 核对输入是否为常数、有效样本量、标签、Hook 与后端版本，不直接把零解释为正常研究结果 |

## 验证与当前边界

2026-11-25 在 macOS 的全新 Python 3.12 环境验证了源码安装与公开测试集合。Python 有 322 项通过，另有 10 个子测试通过；Node 22.23.2 下 28 项 Web 测试通过，按锁文件安装和生产构建均成功。浏览器实测打开了首页、发射台、数据页与因子审查页，并渲染合成行情产生的因子图表。交易所元数据使用测试夹具。下列命令可在后续版本重复执行。

全新源码 checkout 安装 `.[test]` 后，可在根目录运行完整产品测试：

```bash
python -m pytest -q tests
cd web
npm ci
npm test
npm run build
```

如果工作区另有私人研究测试，可在根目录使用以下 15 个文件组成的固定核心集合：

```bash
python -m pytest -q \
  tests/test_cli_smoke.py tests/test_data_gap_repair.py tests/test_data_naming.py \
  tests/test_download_integrity_round2.py tests/test_download_status_round2.py \
  tests/test_factor_analytics.py tests/test_inspector_review_round2.py \
  tests/test_llm_api_manager.py tests/test_llm_guardrails.py \
  tests/test_missing_target_semantics.py tests/test_nn_miner.py \
  tests/test_operator_runtime.py tests/test_source_review_core.py \
  tests/test_targets.py tests/test_task_execution.py
```

核心测试使用合成数据、临时存档与外部接口替身，覆盖下载完整性和并发合并、模式校验、指标/模型恢复、失败传播和 Web 状态等路径。Web 测试验证组件与 hook 的受控事件，构建验证 TypeScript 和静态产物；它们都不代替真实网络、交易或新机器安装验收。

使用范围还应注意：

- 当前核心提供因子研究和管理，不包含完成验收的实盘订单、账户风控、手续费/滑点/资金费率回测闭环；因子相关性不代表交易盈利。
- 通用去重主要基于源码哈希；参考 NN 另有候选输出相关性筛选，不能将其扩展为所有 Miner 的统一语义去重。
- 评估使用本地线程池，尚未接入完整 Ray/Celery 分布式执行。
- 当前 API 没有完整的用户认证与多租户权限体系，安装流程采用本机监听。
- 代码执行限制不替代前视偏差、过拟合和独立样本研究审查。Data Center 显示文件时间范围；评估器 Coverage 基于输出有效量，CLI Inspector Coverage 基于因子与标签共同有效量。跨资产与单标的的分母不同，不能直接混用；详见[评分指南](fitness_hook_guideline.md)。
- Web Inspector 使用已存快照，CLI Inspector 逐标的复评。底层跨资产指标入口的分组收益尚未实现，缺失分组显示不可用；不要将缺失结果视为零收益。
- 现有 Web 测试是受控事件验证；真实浏览器网络断线、多进程写入及服务重启恢复不属于上述测试通过结论。

维护人员可在已有环境中复查文档所用接口：

```bash
python -m core.cli --help
python -m core.cli mine --help
python -m core.cli download --help
python -m core.cli inspect --help
```

测试不会自动证明因子有经济价值，也不覆盖所有自行添加的扩展。执行私人研究工具前，应单独确认其输入、运行预算和输出范围。

## 维护参考

- [项目 README](../README.md)：项目入口与扩展示例。
- [架构文档](architecture/v4_architecture_design.md)：模块关系及已实现契约。
- [产品需求文档](FactorMiner_PRD.md)：产品需求和演进方向。
- [CLI 定义](../core/cli.py)、[API 定义](../api/main.py)：参数与服务接口。
- [启动校验](../core/startup_validation.py)、[标签实现](../core/evaluation/targets.py)、[存档实现](../core/storage/factor_storage.py)：配置与产物的实际解释。
