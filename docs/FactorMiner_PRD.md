# FactorMiner 产品需求文档 (PRD)

## 1. 产品概述
**产品名称**: FactorMiner
**当前版本**: V4.0.0 (底层架构与全栈UI重构阶段)
**产品定位**: 专业的量化因子挖掘、评估和优化平台。基于全新 V4 “领域驱动”设计架构，实现多挖掘范式的统一整合，并提供基于 React/Vite 的极度沉浸式可视化大盘，帮助量化交易者和研究人员高效构建高质量的量化因子。
**目标用户**: 量化交易者、金融数据分析师、量化策略研究员。

## 2. 核心业务流程
1. **数据获取与管理**: 从交易所下载高质量K线数据，支持高级动态元数据联动获取（按流动性排序）、智能批量下载与前端控制台实时监控、覆盖率检查和断层修复。
2. **因子挖掘**: 支持四大异构流派 (GP、RL、NN、LLM) 统一调度与生成。
3. **因子存储**: 标准化存储与谱系追踪，分离灵魂（逻辑代码/模型权重）与肉体（Parquet截面得分矩阵）。
4. **因子评估**: 防御性并行计算图，支持自动计算 IC、IR、胜率及动态防重复过滤（Anti-Bloat 拦截机制）。
5. **因子优化与组合**: 使用算法对单因子或多因子组合进行优化。

## 3. 功能需求详细说明

### 3.1 数据管理模块
- **数据源获取**: 深度集成 CCXT 框架，动态读取各交易所元数据与流动性排序。支持多时间框架 (1m, 5m, 15m, 1h, 4h, 1d)。
- **批量与智能下载**: 支持基于“Exchange -> TradeType -> Symbol”级联过滤的多维度笛卡尔积批量下载，彻底解决错乱文件命名的问题 (`1000CAT_USDT_USDT-1m-futures`)。
- **数据质量控制**:
  - Web 客户端直观覆盖率（Coverage）多维宽表透视透视。
  - 自动断层检测与智能填充 (Gap Filling) 以及增量（Merge）下载更新功能。

### 3.2 因子挖掘与计算模块 (V4)
- **GP (遗传规划)**: 基于达尔文式演化，自动生成因子 AST 树并杂交变异。
- **RL (强化学习)**: 将因子生成视作 MDP 过程，基于奖励使用策略梯度 (Policy Gradient) 自适应演化公式。
- **LLM (大语言模型)**: 基于 In-context 动态生成与执行沙盒反馈（Reflection），驱动安全迭代。
- **NN (神经网络)**: 张量级全特征矩阵计算，保留完整反向传播梯度图 `requires_grad=True`。

> 命名约定：历史名称 `DL` 已弃用，对外产品、配置和界面统一使用 `NN`；旧档案仅做兼容读取。

### 3.3 因子评估体系
- **并发与防重复**: `ParallelEvaluator` 引擎结合 `DiversityFilter` 硬去重，通过 MD5 缓存免疫机制极速拦截计算冗余。
- **防御性沙箱**: `RestrictedSandbox` 对 AI 生成的逻辑进行白名单限制执行，确保生产环境底层物理安全。

### 3.4 因子优化与组合
- **优化算法**: 贪婪算法、遗传算法、Lasso回归，控制过拟合并最大化IC。
- **组合方式**: 等权重、IC加权、机器学习加权。

### 3.5 谱系感知因子存储架构 (V4)
- **元数据绑定**: 每条因子携带独一无二的 `FactorMetadata`（存储父代 ID、超参 Config、创建时间等）。
- **异构统一**: GP (存 JSON AST)、RL (存 Action List + Model Weights)、LLM (存 Py 脚本 + Reflection Log)、NN (存 Model Checkpoint)，底层全打平存储。

### 3.6 用户界面与交互接口 (Web UI & CLI)
- **指挥中心 Dashboard**: React/Vite 构建。提供一键任务发射、动态配置加载、以及全局挖掘指标看板。
- **监控大厅与控制台**: 通过 WebSocket 推送微秒级进度。居中模态弹窗提供运行时 LLM Reflection / 终端控制台 Execution Console 以及实时 IC 适应度演化图。
- **命令行界面 (CLI)**: 纯无头模式运行，支持 `factorminer mine`（配置驱动挖掘）、`factorminer download`（行情批量下载）和 `factorminer inspect`（因子多维审查与归因）。
- **API接口**: FastAPI 高性能异步后台支持，完全解耦。

### 3.7 因子归因与审查引擎 (Factor Inspector)
- **多维度归因分析**: 支持 Pearson IC、Spearman RankIC、RankIC IR (年化)、RankIC t-stat、Positive IC Ratio 占比。
- **数据有效覆盖率检查 (Coverage)**: 自动统计有效 Bars 占总 K 线数的百分比，及时暴露因子逻辑因嵌套极值/复杂算子导致严重缺少有效数据（如只剩几十根有效 K 线）的虚假高分过拟合问题。
- **Lag IC 衰减分析**: 计算因子在 Lag 1..10 的延迟相关性，检验因子的衰减速度。
- **分组多空收益**: 5-Quantile 分组收益计算与 Long-Short (Q5-Q1) 换手率/多空利差计算。
- **无缝输入支持**: 兼容 Factor ID（解析 DB）、AST 表达式字典字符串（解析任意日志中的表达式）以及纯 Python 代码。

## 4. 系统架构设计 (V4)
- **展现层**: React.js, WebSockets, ECharts 可视化 (独立运行在 5173 端口)。
- **控制层**: FastAPI (后台 8000 端口)，依赖 `BackgroundTasks` 分配常驻线程。
- **抽象层**: 引入 `BaseFactorMiner`, `FactorExpression` 统一范式。
- **审查与归因层**: 引入 `core/inspector` (`FactorResolver`, `InspectorMetricEngine`, `InspectorReporter`, `FactorInspectorEngine`)。
- **底层引擎**: 异构存储引擎、并发评价引擎池、自动数据补全探针。

## 5. 当前阶段与重点迭代方向 (V4 落地与扩展)
- **完善多资产跨截面挖掘**: 强化 `cross_asset` (横截面运算) 原生能力。
- **MCP协议改造 (Model Context Protocol)** (规划中):
  - **目标**: 将核心服务转化为MCP标准服务，赋能AI大模型与Agent直接调用。
  - **接口规划**: `get_market_data`, `compute_factor`, `evaluate_factor`, `mine_factors`, `inspect_factor`等。

## 6. 非功能性需求
- **环境要求**: Python 3.8+, 建议8GB内存与50GB+存储。
- **可用性**: 提供良好的错误处理、日志记录与Web端实时进度反馈。
- **扩展性**: 系统需保持模块化设计，以支持未来轻松接入更多数据源、因子类型与评估算法。
