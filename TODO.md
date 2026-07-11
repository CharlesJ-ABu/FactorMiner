# FactorMiner V4 待办事项

> 🚀 **项目状态**: V4 架构底层引擎已搭建完毕，成功跑通 GP、RL、LLM、DL 四大异构挖掘范式！评估引擎 (`ParallelEvaluator` + `custom_fitness` 钩子) 也已通过沙盒与真实数据回测的闭环验证。目前正处于补齐“持久化存储”并向 Web UI 界面对接的阶段。

## 🔄 当前核心任务 (进行中)

### 1. **Web UI 界面联调与可视化配置** (完成)
**优先级**: 最高  
**说明**: 将 V4 强大的后台引擎接入 React 前端，抛弃纯 CLI 运行。
- [x] **可视化 Config 生成**: 在前端增加配置面板，支持用户点选 Universe、Mine Period、Test Period 以及所需的行情特征列 (Features)。(通过直接解析工作区 Config JSON 实现)
- [x] **范式动态切换**: 允许用户在界面上选择 `MyCustomGP` 或 `MyCustomRL`，并自动呈现对应的参数表单。
- [x] **启停控制与后台任务池**: 将前端表单序列化下发至后台 `/api/launch`，后端以异步线程启动 `FactorMinerDirector`，并通过 WebSocket 将挖掘进度实时广播至前端 Drawer。

### 2. **算子与计算引擎扩展** (进行中)
**优先级**: 中  
**说明**: 扩展现有的单品种串行计算能力。
- [x] **Cross-Asset 截面计算**: 当 `mining_mode` 设置为 `cross_asset` 时，重构底层数据对其逻辑，支持横截面算子 (如 `cs_rank`, `cs_zscore`) 的计算。
- [ ] **更多原生算子支持**: 在 `OperatorRegistry` 中预置更多金融界常用的基础算子库 (如 `ts_decay`, `ts_corr`)。

---

## ✅ 已完成核心里程碑 (V4 架构)

- [x] **V4 架构底座奠基**: 抽象出统一的 `BaseFactorMiner` 和 `FactorMinerDirector` 流程，支持异构流派。
- [x] **统一评判标准**: 抽象出 `FactorExpressionAST` 语法树节点执行规范，允许因子转换为可执行的 Pandas 计算图。
- [x] **真实数据切片器**: 完成 `RealDataClient` 的开发，支持读取高频 `.feather` 并在时间片上（`mine_period`, `test_period`）无缝拼接。
- [x] **工业级 Config-Driven 模式**: 告别硬编码，全盘迁移至类似 Freqtrade 的 `config.json` 及 `factorminer` 驱动。
- [x] **GP 范式落地**: 编写 `MyCustomGPMiner`，完成达尔文式的变异、交叉及精英保留闭环验证。
- [x] **RL 范式落地**: 编写 `MyCustomRLMiner`，彻底解耦 PyTorch 依赖，通过 Policy Gradient 权重字典完成概率采样与反馈闭环验证。
- [x] **LLM 范式落地**: 编写 `MyCustomLLMMiner`，实现大语言模型的自然反思机制 (Reflection) 及 API 容灾降级容错，直接生成 Python 源代码并通过安全沙盒评估。
- [x] **DL 范式落地**: 编写 `MyCustomNNMiner`，使用纯 NumPy 实现含有向后传播 (Backpropagation) 及梯度截断 (`requires_grad=True`) 能力的微型张量机制，验证了 V4 引擎对端到端深度学习的原生兼容性。
- [x] **评估与沙盒闭环验证**: 成功剥离出 `user_workspace/custom_fitness/` 并在真实执行流中验证了 `EvaluatorRegistry` 钩子注入机制（如 `my_bear_market_hunter`）；跑通了防御性沙盒 `RestrictedSandbox` 及针对 DL 的张量短路评估机制。
- [x] **因子落盘与持久化存储**: 补全了 `LocalFactorStorage`，实现了每个 Epoch 结束时将优质因子、元数据及评价指标落盘至 `factor_db/` 数据库。
- [x] **多品种序列及横截面挖掘引擎**: 实现了 `sequential_single` 以及基于矩阵计算的 `cross_asset` 并行截面 IC 计算，并在 CLI 终端完美输出跨资产综合战报。
- [x] **数据自动拉取与补全**: 强化 `RealDataClient`，支持当本地缺失配置的行情 `.feather` 时，自动调用 `DataDownloader` 尝试后台无感下载并对齐目标数据段。
- [x] **健壮的 IC 评估机制 (Anti-Bloat)**: 针对 GP 范式由于代码膨胀 (Bloat) 生成的常数无效因子，在 Pandas 矩阵 `corrwith` 时引入告警拦截与降级，保证挖掘控制台输出清爽。
- [x] **全局逻辑硬去重 (Global Hard Deduplication)**: 打通 `FactorStorage` 与 `DiversityFilter` 的通讯，在引擎启动时自动将全库因子历史 Hash 注入拦截网，阻止重复因子的无效计算与污染。
- [x] **高级批量数据下载器集成 (Advanced Batch Downloader)**: 将 CCXT 动态元数据获取、网络降级熔断机制、以及基于笛卡尔积排列组合的批量下载和覆盖率分析无缝集成到统一后台。
- [x] **Web UI 下载控制台集成**: 在前端实现下载日志的实时 Console 打印，提供对运行中下载任务的透明度和进度监控。
- [x] **市场元数据优化与联动**: 在后端实现基于市值/流动性（Quote Volume）的智能排序，并优化前端 `Exchange -> TradeType -> Symbol` 的级联过滤逻辑，防止跨市场误选。
- [x] **底层存储命名规范闭环**: 修复全链路对 `ccxt` 返回带冒号合约标的（如 `1000CAT/USDT:USDT`）的文件名解析一致性，确保 `Batch Downloader`、`Main API`、`Real Client` 之间对历史存储命名 (`replace('/', '_').replace(':', '_')`) 的完全兼容。
---

**最后更新**: 2026年7月11日  
**维护者**: @CharlesJ-ABu  

> 💡 **提示**: 此文件记录了 V4 重构后的核心骨架与待落地的待办事项。
