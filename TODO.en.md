# FactorMiner V4 development checklist

[简体中文](TODO.md) · [English documentation](docs/README.en.md)

> **Historical snapshot, July 17, 2026.** The original checklist recorded a working engine for GP, RL, NN and LLM, sandbox and historical-data checks for `ParallelEvaluator` and `custom_fitness`, and ongoing work on persistence and Web UI integration. The checkboxes below preserve that snapshot. For the current status of configurable workers, GP search, the Launchpad and other features, see the [product and installation guide](docs/FactorMiner_Product_Report_EN.md).

## Work in progress at the time of this checklist

### 1. Web UI integration and configuration (marked complete)

**Priority:** Highest. Connect the Python engine to a React frontend alongside the CLI.

- [x] **Configuration panel.** Present the universe, mining period, test period and required market features from workspace JSON configurations for selection and review.
- [x] **Dynamic paradigm selection.** Let users select `MyCustomGP` or `MyCustomRL` and show the corresponding parameter form.
- [x] **Run controls and background tasks.** Send the form to `/api/launch`, run `FactorMinerDirector` in a background thread and broadcast progress to the frontend task drawer over WebSocket.
- [x] **Research Dashboard.** Use `/api/dashboard` to display real task state, the archive and review coverage, top fitness scores, miner mix and recent runs. Link to Launchpad and Inspector.
- [x] **Three interface languages, phase one.** Detect the browser language, persist the local preference and support Chinese, English and German in navigation, Research Dashboard and Factor Inspector. Keep research artifacts in their original form.
- [x] **README overview, Web UI and extension tutorial.** Update the introduction, capabilities and directory map. Add actual screenshots of Dashboard and Inspector. Document the output contracts of all four paradigms, operator registration and fitness hook configuration.
- [ ] **Three interface languages, phase two.** Extend the translation of Launchpad and Data Downloader forms, live log prompts and backend error messages. Preserve API enums and factor content across languages.

### 2. Operators and computation (in progress)

**Priority:** Medium. Extend the existing sequential single-asset engine.

- [x] **Cross-asset computations.** Align the inputs for `mining_mode` set to `cross_asset` so cross-sectional operators such as `cs_rank` and `cs_zscore` can run.
- [x] **User operators in the search loop.** Feed dynamically loaded operators into GP/RL search, generate ASTs using operator arity, evaluate them through the shared path and report runtime errors. The default MyCustomGP search includes arithmetic, ts_mean/ts_std and user time-series operators for decay, z-score, change, rank and volatility.
- [x] **Extension validation and four-paradigm CLI smoke tests.** Validate miners, operators, fitness hooks and module imports before reading market data in the CLI or Launchpad. The checklist records a one-iteration CLI test using real Feather data for MyCustomGP, MyCustomRL, MyCustomLLM and NN.
- [x] **MyCustomGP data regression and metric contract.** After restarting FastAPI, run the BTC and SUI perpetuals configuration from `config.json`. Verify that user operators enter the AST, candidates can receive nonzero scores, and `IC`, `RankIC`, `Turnover` and `fitness_score` are stored with Phase II factor and forward-return snapshots. Restarting clears Tracker memory. The persistent archive is `factor_db`.
- [ ] **More native operators.** Provide additional financial primitives such as `ts_decay` and `ts_corr` in `OperatorRegistry`.

### 3. Deduplication and evaluation (in progress)

**Priority:** Medium. At the time of the checklist, `DiversityFilter` rejected exact source-code MD5 matches across the archive, general correlation-based filtering was a placeholder and `ParallelEvaluator` used an eight-thread `ThreadPoolExecutor`. Ray and Celery were not integrated.

- [ ] **Correlation-based near-duplicate filtering.** Compute correlations between candidate outputs, filter above a threshold and check orthogonality against historical factors.
- [ ] **Configurable concurrency.** Configure worker count and define thread-pool resource limits, timeouts and exception cleanup.
- [ ] **Distributed evaluation.** Assess Ray or Celery for dispatch, result caching and retries at larger candidate volumes.

### 4. Factor Inspector (phase two marked complete)

**Priority:** High. Replace the static demo with an archive-backed catalog, paradigm-specific logic views, snapshot diagnostics and batch review.

- [x] **Persistent catalog.** Add FactorStorage listing and lifecycle updates along with `/api/factors` and `/api/factors/{id}`. Source the catalog independently of task memory.
- [x] **Logic by paradigm.** Show stored GP ASTs, LLM source, RL action histories and NN model versions/channels. Link Launchpad results to Inspector.
- [x] **Lifecycle updates.** Persist explicit changes to `DISCOVERED`, `INSPECTED`, `PAPER_TRADING`, `LIVE` and `RETIRED`.
- [x] **Review snapshots and tearsheets.** Save aligned factor values, forward returns and data lineage on archive. Compute rolling IC, quantile returns, turnover and date ranges from them. Prompt for re-mining when a snapshot is absent.
- [x] **Multi-factor comparison and bulk review.** Compare rolling IC for 2–5 factors that have snapshots and update lifecycle labels in bulk.
- [ ] **Advanced attribution and portfolio backtesting.** Add NN feature attribution, sector/market neutralization, quantile portfolio equity curves and trading cost models using saved snapshots.

## Recorded V4 milestones

- [x] **Shared engine.** Introduced `BaseFactorMiner` and `FactorMinerDirector` for a common workflow across mining methods.
- [x] **Shared expression contract.** Defined how `FactorExpressionAST` nodes execute as pandas computations.
- [x] **Historical data slicing.** Implemented `RealDataClient` to read high-frequency `.feather` files and assemble intervals for `mine_period` and `test_period`.
- [x] **Configuration-driven runs.** Moved hard-coded parameters into a Freqtrade-style `config.json` and the `factorminer` command.
- [x] **GP.** The July checklist recorded mutation, crossover and elite retention for `MyCustomGPMiner`. The current product guide describes the shipped reference search and identifies full crossover as an extension.
- [x] **RL.** Implemented `MyCustomRLMiner` without a PyTorch dependency. A policy-gradient-style weight dictionary drives probabilistic sampling and incorporates feedback.
- [x] **LLM.** Implemented reflection and API fallback in `MyCustomLLMMiner`. It generates Python source and passes it to the restricted evaluator.
- [x] **NN.** Implemented a replaceable `MyCustomNNMiner` using a NumPy MLP with backpropagation, gradient clipping, early stopping and a channel diversity constraint to exercise end-to-end neural factors in V4.
- [x] **NN artifacts.** Support single-asset and cross-asset inputs, train on `mine_period` and use `test_period` metrics to select channels. Freeze the best channels across epochs and store weights, scaler and feature schema in `.npz` packages. Support the existing Web UI and older `.pt` archives. Selection makes this second period a validation set.
- [x] **Temporal NN tutorial.** Add `MyTemporalNN` with time-aligned multi-period OHLCV features designed to avoid leakage, configurable forward-return labels, a joint MSE + IC objective, signed RankIC selection and model-format reloading. Supply a roughly ten-minute single-asset learning configuration.
- [x] **Evaluation and sandbox integration.** Separate `user_workspace/custom_fitness/` and exercise `EvaluatorRegistry` hooks such as `my_bear_market_hunter` in the execution path. Exercise `RestrictedSandbox` and the NN tensor evaluation shortcut.
- [x] **Persistent factor archive.** Extend `LocalFactorStorage` to save selected factors, metadata and metrics to `factor_db/` at the end of each epoch.
- [x] **Sequential and cross-sectional mining.** Implement `sequential_single`, matrix-based cross-sectional IC in `cross_asset` and a consolidated multi-asset CLI report.
- [x] **Automatic data retrieval.** When a configured `.feather` file is missing, `RealDataClient` can call `DataDownloader` to fetch and align the requested segment.
- [x] **IC handling for constant outputs.** Handle pandas `corrwith` warnings and fallback behavior for constant factors, including expressions produced by GP bloat.
- [x] **Global exact deduplication.** Load historical hashes from `FactorStorage` into `DiversityFilter` on startup to skip repeated logic.
- [x] **Batch downloader.** Integrate dynamic CCXT metadata, network fallback, Cartesian-product batch selection and coverage inspection in the backend.
- [x] **Web download console.** Show download logs and progress during active jobs.
- [x] **Market selection.** Order symbols by a liquidity proxy (quote volume) and filter by Exchange → Trade Type → Symbol to avoid selecting the wrong market.
- [x] **File naming.** Use `{safe_symbol}-{timeframe}-{trade_type}.feather` across the Batch Downloader, main API and data client. Futures configurations require CCXT’s original symbol, e.g. `1000CAT/USDT:USDT`. The new reader excludes legacy filenames.

**Last updated in the source:** July 17, 2026. **Maintainer:** @CharlesJ-ABu. This checklist records milestones and outstanding work from the V4 restructuring.
