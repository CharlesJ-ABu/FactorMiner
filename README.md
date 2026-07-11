# FactorMiner - 量化因子挖掘平台 (V4)

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/Version-4.0.0-orange.svg)]()
[![Maintenance](https://img.shields.io/badge/Maintenance-Actively%20Maintained-green.svg)](https://github.com/CharlesJ-ABu/FactorMiner)

> 🚀 **项目状态**: **V4 架构全面重构完成**（涵盖 React + FastAPI 前后端分离、以及四大挖掘范式）！
> 👨‍💻 **维护者**: [@CharlesJ-ABu](https://github.com/CharlesJ-ABu)  
> 📅 **最后更新**: 2026年7月  

FactorMiner 是一款极客级别的专业量化因子挖掘与管理平台。在全新的 **V4 架构** 下，系统完成了从“传统代码堆砌”向“基于声明式配置 (Config-Driven)”和“控制反转 (IoC)”的彻底蜕变。我们不仅提供了底层极其硬核的异构计算引擎，更配备了充满未来科技感的 React + FastAPI Web 工作台，帮助量化研究人员以顶级的视觉密度洞察数据。

---

## ✨ V4 核心特性

- 🧬 **四大异构挖掘范式**: 系统原生支持 **大语言模型 (LLM)**、**遗传规划 (GP)**、**深度学习隐空间 (DL)** 以及 **强化学习策略梯度 (RL)** 四大维度的因子挖掘，并将它们的底层表达统一映射为标准的 `FactorExpression` 体系。
- 👁️ **沉浸式控制台体验**: 抛弃繁杂的纯 CLI 黑盒，基于 WebSocket 实现微秒级的任务状态穿透。提供“动态赛马图”、“大模型反思滚动日志”、“Data Downloader 暗黑终端”等丰富的实时前端反馈。
- 🧩 **全透明的因子生命周期 (FactorStorage)**: 引入 `FactorMetadata` 对因子实现“灵魂与肉体分离”。GP 存为跨语言 AST 树，LLM 存为带反思历史的纯 Python 脚本，DL 存为网络张量通道；而产出的时序打分全部统一为最高性能的 Parquet 矩阵，供下游随时组合。
- 🛡️ **反膨胀与硬去重免疫系统**: 结合 MD5 哈希校验池 (`DiversityFilter`)，引擎在启动前即拦截 99% 的同质化废弃代码或重叠逻辑，大幅节省并行回测集群（如 Ray / Celery）的计算开销。
- 📡 **数据基建级联联动**: 深度集成 CCXT 元数据获取引擎，支持按市场流动性自动排序，并通过最严格的文件命名规范 (`symbol.replace('/', '_').replace(':', '_')`) 彻底消除现货、期货、永续合约的数据跨界污染。

---

## 🏗️ V4 架构目录结构

```text
FactorMiner/
├── README.md                    # 本文档
├── requirements.txt             # Python 依赖包
├── config/                      # 系统级配置文件目录
│   └── settings.py              
├── api/                         # FastAPI 后端模块 (V4)
│   ├── main.py                  # API 主入口、HTTP 路由与 WebSocket 日志劫持
│   └── ws_manager.py            # WebSocket 广播管理器
├── web/                         # React + Vite 前端工作台 (V4)
│   ├── src/                     # 组件、页面与 Hook (包含 Launchpad, Downloader)
│   ├── package.json             
│   └── vite.config.ts           
├── core/                        # V4 核心挖掘引擎
│   ├── data_feed/               # 高频行情切片、CCXT 批量爬虫与无缝拼接模块
│   ├── evaluation/              # 统一并行评价器与代码执行沙盒 (RestrictedSandbox)
│   ├── miner/                   # 因子挖掘引擎逻辑
│   │   ├── paradigms/           # 异构挖掘范式基类 (llm_miner, gp_miner 等)
│   │   └── director.py          # 任务调度与演化生命周期总管 (FactorMinerDirector)
│   └── storage/                 # 持久化存储规范接口层
├── user_workspace/              # ⭐️ 你的实验室：用户自定义实验区
│   ├── configs/                 # Config-Driven 驱动文件 (configRL.json 等)
│   ├── custom_miners/           # 用户扩展的具体挖掘流派实现
│   ├── custom_operators/        # 用户手写的数学或技术特征衍生算子
│   └── custom_fitness/          # 自定义评价挂钩函数 (Fitness Hooks)
├── factor_db/                   # 挖掘因子的落地存储区 (Parquet 矩阵与灵魂元数据)
├── data/                        # Binance 市场历史行情存储库 (.feather)
└── docs/                        # 系统架构与使用指南文档区
```

---

## 🚀 快速开始

### 1. 环境准备
```bash
# 克隆项目
git clone https://github.com/CharlesJ-ABu/FactorMiner.git
cd FactorMiner

# 建议使用 uv 或 conda 创建纯净的 Python 3.10+ 环境
python -m venv venv
source venv/bin/activate  # Mac/Linux

# 安装后端依赖
pip install -r requirements.txt
```

### 2. 启动服务 (前后端分离)

**启动后端引擎 (FastAPI)**
```bash
# 后端运行于 8000 端口
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

**启动极客工作台 (React Web)**
```bash
# 新开一个终端窗口
cd web
npm install  # 仅首次需要
npm run dev
```
启动成功后，浏览器访问 `http://localhost:5173` 即可进入 FactorMiner 极客工作台。

### 3. 无头模式 (CLI 命令行挖掘与下载)

如果你希望在服务器后端挂机，或者不启动 Web 界面直接运行，FactorMiner 也提供了原生纯命令行的工业级入口：

**1. 命令行批量下载历史行情**
我们内置了 `factorminer download` 命令，可直接走高速通道批量拉取数据：
```bash
# 下载现货行情
factorminer download --exchange binance --symbols BTC/USDT,ETH/USDT --timeframes 1d,1h --type spot --start 2023-01-01 --end 2024-01-01

# 下载 U 本位永续合约行情
factorminer download --exchange binance --symbols BTC/USDT:USDT --timeframes 1m --type futures --start 2024-05-01 --end 2024-05-10
```

**2. 命令行执行因子挖掘**
使用 `factorminer mine` 工业级总控入口，通过 Config 驱动任务运行：
```bash
# 运行原生的 GP (遗传规划) 挖掘 (代数和参数均由 config 文件控制)
factorminer mine --miner GP --config user_workspace/configs/demo_config.json

# 运行你在 user_workspace 中自己写的自定义挖掘器 (例如 MyCustomGP)
factorminer mine --miner MyCustomGP --config user_workspace/configs/config.json --user-dir user_workspace
```
挖掘完成后，终端会直接打印全局大表 (Final Mining Summary)，记录所有存活的因子及其 IC 表现。

---

## 📚 官方文档体系

欲了解 V4 架构的深度技术细节与组件交互原理，请前往 `docs/` 目录查阅官方文档：

1. 🏛️ **[FactorMiner V4 架构设计红皮书](docs/architecture/v4_architecture_design.md)**：包含四大流派的设计哲学、持久化追踪机制与沙盒拦截原理。
2. 🗺️ **[产品需求与功能规格 (FactorMiner_PRD)](docs/FactorMiner_PRD.md)**：项目整体功能列表。
3. 🖥️ **[前端体验规范 (WEB_UI_PRD)](web/WEB_UI_PRD.md)**：包含沉浸式大盘、Data Downloader 日志终端等 UI 设计理念。
4. ⚙️ **[网络与环境配置指南](docs/guides/vpn_setup.md)**：代理环境调优指南。

> *(注意：V3 时代的函数式 API、过期的 Streamlit 界面和 `factorlib` 结构设计均已放入 `docs/legacy_v3/` 作归档处理。)*

---

## 🤝 贡献与反馈

欢迎各位同好提交 PR！在使用中遇到任何 Bug 或有新的因子评估建议，请随时在 [Issues](https://github.com/CharlesJ-ABu/FactorMiner/issues) 提交。
量化之路漫漫，愿 **FactorMiner** 助你挖掘出最强的 Alpha。

## 📄 许可证

本项目基于 MIT 许可证。详见 [LICENSE](LICENSE) 文件。
