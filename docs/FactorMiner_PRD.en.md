# FactorMiner product requirements

[简体中文](FactorMiner_PRD.md) · [English documentation](README.en.md)

## 1. Product overview

**Product:** FactorMiner, V4.0.0. Updated November 25, 2026.

**Scope:** A local, single-user factor research workbench run from a source checkout. Researchers generate candidates with several mining methods, evaluate them, archive the results and review them in a React/Vite interface. Trading accounts, order execution and a complete portfolio backtesting system are outside the current release.

**Audience:** Quantitative traders, financial data analysts and quantitative strategy researchers. The sections below identify implemented features and planned work. For installation and first use, follow the [product guide](FactorMiner_Product_Report_EN.md).

## 2. Research workflow

1. **Acquire and manage market data.** Download OHLCV bars, retrieve exchange metadata ranked by liquidity, submit batch downloads, monitor progress in the web console, inspect local coverage and repair gaps.
2. **Mine factors.** The engine schedules candidate generation across four paradigms: genetic programming (GP), reinforcement learning (RL), neural networks (NN) and large language models (LLM).
3. **Store factors.** Track provenance and separate the logic or model weights from computed values, such as a Parquet matrix of cross-sectional scores.
4. **Evaluate factors (implemented).** Compute IC, RankIC, changes in factor values and custom fitness in parallel. The default deduplication rule hashes source code. Inspector reports correlation diagnostics. Trade win rates require account-level execution data.
5. **Optimize and combine factors (planned).** Single-factor selection is available. Joint optimization across multiple factors has no product entry point yet.

## 3. Functional requirements

### 3.1 Market data

- **Sources.** CCXT retrieves exchange metadata and supports liquidity rankings. Timeframes include 1m, 5m, 15m, 1h, 4h and 1d.
- **Batch downloads.** Filter by exchange, then market type, then symbol. The downloader processes combinations of selected dimensions using consistent filenames, such as `1000CAT_USDT_USDT-1m-futures` before the extension.
- **Quality.** The web coverage table shows file existence, first and last timestamps, and row counts. Checking every expected bar for gaps and calculating factor coverage are separate operations. The downloader can detect and fill gaps and merge incremental updates.

### 3.2 Mining and computation

- **GP (reference search implemented).** Generates abstract syntax trees (ASTs), mutates them and retains elite candidates. A full crossover strategy remains an extension.
- **RL (partial).** `MyCustomRL` updates operator and field sampling weights from scores. The built-in `RL` illustrates random actions. Full actor–critic, PPO and DQN implementations remain outside this release.
- **LLM.** Generates code using prompt context and uses sandbox feedback to revise candidates through reflection.
- **NN (reference model implemented).** A NumPy MLP fits the model and scaling parameters on the training period, then selects and saves output channels. The training branch’s `requires_grad` flag does not guarantee a general PyTorch autograd graph. Users can extend the model protocol. When `test_period` helps select channels, it serves as validation. Reserve separate data for final testing.

The public product, configuration and UI use `NN`. The historical name `DL` is supported when reading older archives.

### 3.3 Evaluation

- **Concurrency and deduplication (implemented).** The local evaluator defaults to 8 threads, configurable from 1 to 64. `DiversityFilter` hashes source with MD5 and can miss different implementations of the same formula. General expression-bloat control is separate. The reference NN also filters by output correlation.
- **Execution safeguards (implemented within limits).** AST and API allowlists, separate processes and resource budgets restrict code execution. Resource limits depend on OS support. Researchers must still check causality, look-ahead bias and study validity.
- **Distributed evaluation (planned).** Ray and Celery backends have not been integrated.

### 3.4 Factor and portfolio optimization (planned)

The proposed optimization methods are greedy search, genetic algorithms and Lasso regression, with the goals of controlling overfitting and maximizing IC. Proposed combinations include equal weights, IC-based weights and weights learned by an ML model. There is no unified interface for this workflow yet.

### 3.5 Storage and lineage

- **Metadata (implemented).** `FactorMetadata` stores logic, metrics, data lineage, creation time, selected configuration fields and parent references declared by the miner. A parent reference may not resolve to an archived factor ID. This record alone cannot reconstruct an entire search state or evolutionary tree.
- **Paradigm-specific storage (main paths implemented).** GP and reference RL store evaluable ASTs. LLM stores source, reflection and model provenance. NN stores a portable model and channel. Old action-only RL archives cannot reconstruct an executable factor. A full policy-network checkpoint workflow is still pending.

### 3.6 Web UI, CLI and API

- **Dashboard.** The React/Vite workbench offers task launch, dynamic configuration discovery and aggregate mining metrics.
- **Monitoring (implemented).** WebSocket delivers task status, logs and each epoch’s best candidate IC and measured complexity. A modal shows the results. Reflection appears in the common logs or archive. A dedicated live drawer is planned. There is no microsecond latency guarantee. The current API process replays its latest task state after a reconnect. Restarting the process clears that state.
- **CLI.** `factorminer mine` runs a configuration-driven experiment, `factorminer download` fetches historical bars in batches and `factorminer inspect` evaluates and diagnoses a factor without a browser.
- **API (implemented).** FastAPI runs separately from the computation threads. The default service binds to loopback. Full authentication and multi-tenant access control are outside the release.

### 3.7 Factor Inspector

- **Correlation (implemented).** Computes Pearson IC, Spearman RankIC, IR, a t-statistic and the fraction of positive IC observations. IR uses a fixed `sqrt(252)` multiplier regardless of bar frequency or market calendar. These measures do not establish annualized trading returns or a trade win rate.
- **Coverage (implemented).** CLI Inspector counts observations with both a valid factor and a valid label. In cross-asset mode the denominator counts matrix cells. The mining evaluator measures output coverage, while Data Center reports a file’s time range. See the [fitness guide](fitness_hook_guideline.en.md).
- **Lag decay (implemented).** Default lags are 1, 2, 3, 5 and 10. Each requested segment builds its own labels.
- **Group diagnostics (partial).** For a single asset, report mean forward return in five quantiles and the Q5−Q1 spread. Missing groups display as unavailable. Cross-asset CLI group returns are pending. Web Inspector computes its own groups from saved snapshots. Changes in the factor are measured separately from account trading volume.
- **Inputs and failures (implemented).** Accepts a factor ID, a valid AST or Python factor source. The CLI reports failures by asset and exits nonzero if any asset fails. It rejects cross-asset NN re-evaluation. The Python resolver can work with the complete matrix.
- **Web snapshot review (implemented).** Shows the AST, source and reflections, model and channel metadata, rolling IC, group returns and turnover. Users can compare 2–5 factors and manually change lifecycle labels. 3D NN views, feature importance, equity and drawdown charts, and a trading deployment interface remain planned.

## 4. System architecture

- **Presentation:** React, WebSockets and ECharts. The development server runs on port 5173.
- **Control:** FastAPI on local port 8000. BackgroundTasks uses `asyncio.to_thread` for synchronous downloads and mining. Task state lives in process memory.
- **Mining abstractions:** `BaseFactorMiner` and `FactorExpression` provide a shared contract.
- **Inspection:** `core/inspector` includes `FactorResolver`, `InspectorMetricEngine`, `InspectorReporter` and `FactorInspectorEngine`.
- **Engine services:** Paradigm-specific storage, a concurrent evaluation pool and automatic detection of market data that needs downloading.

## 5. Priorities for V4 and future releases

- Extend native multi-asset calculation in `cross_asset` mode.
- **Model Context Protocol (planned).** Expose core services for agents and LLMs through MCP. Proposed operations include `get_market_data`, `compute_factor`, `evaluate_factor`, `mine_factors` and `inspect_factor`.

## 6. Nonfunctional requirements

- **Environment.** Python ≥3.10 (tested with 3.12) and Node.js ≥22.12. Memory and disk requirements vary with the data and model. Install the source checkout with `pip install -e '.[test]'`. In the frontend, use `npm ci`, `npm test` and `npm run build`.
- **Usability.** Provide actionable errors, logs and live progress in the web interface.
- **Extensibility.** Keep modules independent so that new data sources, factor types and evaluation algorithms can be added.
