# FactorMiner V4 官方文档体系

[English](README.en.md)

当前发布范围是从源码 checkout 运行的本机单用户因子研究工具。下面区分使用指南、功能需求和历史归档；需求文档中的规划项不代表已交付功能。

首次了解或安装当前版本，请先阅读 **[FactorMiner V4 产品说明与安装配置指南](FactorMiner_Product_Report_ZH.md)**：包含实际功能、环境准备、启动命令、完整入门配置、使用流程、文件管理和已知限制。

## 🏛️ 1. 核心架构红皮书 (Architecture)

这里存放着指导整个 V4 系统设计的顶层文档。如果您是核心开发者或希望深度定制引擎的高级研究员，请务必阅读本区内容。

*   **[FactorMiner V4 架构设计](architecture/v4_architecture_design.md)**
    *   **核心内容**：详述了系统基于“控制反转 (IoC)”的设计哲学。
    *   **关键模块**：彻底解耦的数据供养层 (`data_feed`)、异构因子存储层 (`storage`)、评价器 (`evaluation`) 以及极速落地编译层 (`execution`)。

## 🗺️ 2. 用户操作与实战指南 (Guides)

本区存放面向一线量化研究员的实操教程，帮助您快速上手，无需关注底层的复杂实现。

*   **[FactorMiner 产品需求文档 (PRD)](FactorMiner_PRD.md)**
    *   项目整体演进路线、功能需求和四大挖掘流派（GP/RL/NN/LLM）的产品愿景。
*   **[Web UI/UX 设计方案与前后端交互规范 (WEB_UI_PRD)](../web/WEB_UI_PRD.md)**
    *   详述了指挥中心、任务发射台、执行控制台以及批量数据下载器沉浸式体验的前端产品定义。
*   **[网络与环境配置指南](guides/vpn_setup.md)**
    *   本机服务、行情连通性与可选代理设置。
*   **[自定义 LLM 与实验留痕指南](guides/custom_llm_experiments.md)**
    *   介绍 `MyCustomLLM` 的用户扩展点、环境变量凭证、独立 Recorder、无头运行和研究报告生成。
*   **[Fitness Hook 编写指南](fitness_hook_guideline.md)**
    *   评分签名、覆盖率口径、可运行配置与失败行为。
*   **[用户扩展与算子示例](../README.md#-自定义-miner算子与-fitness-hook)**
    *   注册 Miner、算子和评分函数的当前接口。

## 🕰️ 3. 历史遗留归档 (Legacy V3)

V3 时代的各种文档（包含不再兼容的 `factorlib` 结构设计、旧版类继承算法规范、旧版 API）已全部移至归档区。
**注意**：这些文档只用于历史与兼容性对照，不能作为当前安装或接口说明。

*   [V3 API 接口废弃列表](legacy_v3/api.md)
*   [V3 旧版因子存储格式说明](legacy_v3/factorlib_structure_guide.md)
*   [V3 用户算法继承范例](legacy_v3/user_algorithm_paradigm.md)
*   [V3 旧版架构变迁文档](legacy_v3/new_architecture_guide.md)

---
安装与测试统一参照产品指南；历史归档不构成当前功能验收。
