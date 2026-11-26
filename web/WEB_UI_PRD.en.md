# FactorMiner Web UI features and interactions

[简体中文](WEB_UI_PRD.md) · [English documentation](../docs/README.en.md)

Updated November 25, 2026. The released workbench runs locally for one user from a source checkout. The status of each feature is given below. For setup, see the [product guide](../docs/FactorMiner_Product_Report_EN.md) and [frontend README](README.en.md).

## 1. Dashboard and Launchpad

### 1.1 Dashboard (implemented)

The home page (`/`) reads the API and factor archive. It shows engine availability, task counts, the factor inventory, distributions by miner and lifecycle stage, high-scoring candidates and the six latest tasks. A factor link opens Inspector. A task link opens the Launchpad list, where the user can select its details. An API restart clears task history. Archived factors persist.

### 1.2 Configuration and launch (implemented)

Launchpad discovers miners and JSON configurations in the workspace. Its preview includes trading pairs, timeframe and iterations. To change a parameter, edit the configuration file. `POST /api/launch` returns a task ID. The backend reads data, initializes the miner and runs the job through `asyncio.to_thread`. The UI displays the reason if startup validation fails. A queued task still has to execute.

### 1.3 Task tracking (implemented)

- Each mining task has its own row. Select it to open a centered dialog with the status, configuration, errors, results and logs.
- WebSocket events update each task, log and scatter plot by task ID. A late HTTP snapshot cannot overwrite newer live state or take a completed task back to an earlier progress value.
- On reconnect, the API replays the latest state held in its current process. Task history is lost on restart. There is no microsecond latency guarantee.

## 2. Mining monitor

### 2.1 IC and complexity (implemented)

Each point represents the best candidate in one epoch. The horizontal axis gives the epoch, the vertical axis its IC, the color its measured complexity, and the tooltip the complexity value. This plot covers one candidate per epoch. It does not measure population-wide performance, lineage, Sharpe ratio or trading returns. Interpret positive and negative IC in light of the direction of the signal.

### 2.2 LLM reflection (partial)

The Execution Console displays task logs. Inspector presents saved source and reflections. A dedicated live sidebar or drawer is planned. Catching an exception and revising code establishes neither causal validity nor out-of-sample performance.

## 3. Factor Inspector

### 3.1 Viewing logic by paradigm

- **GP (basic tree implemented).** The ECharts AST supports expansion, collapse, node tooltips and expression text. A viewer that highlights and explains subexpressions is planned.
- **LLM and RL (archive view implemented).** Inspect saved source, reflections or action sequences. Reference RL formulas that can be evaluated appear as ASTs.
- **NN (partial).** Displays model version, channel and artifact status. The 3D latent-space plot and feature importance chart remain planned.
- Public interfaces use NN. The backend normalizes legacy `dl_channel` archives on read.

### 3.2 Snapshot diagnostics (implemented)

Inspector uses saved factor values and forward returns to compute rolling IC, mean forward return by group and turnover based on changes in the factor. Compare 2–5 factors with snapshots. If a snapshot is missing, the diagnostics show as unavailable. These charts describe factor behavior and do not include trading costs.

### 3.3 Advanced charts (planned)

Portfolio equity and maximum drawdown, long–short strategy returns and decay heatmaps remain planned. Current charts provide axis tooltips. Box zoom and synchronized crosshairs are also planned.

## 4. Lifecycle and execution

### 4.1 Lifecycle management (implemented)

Users can change `DISCOVERED / INSPECTED / PAPER_TRADING / LIVE / RETIRED` individually or in bulk. All are manual labels. Changing one does not start a paper or live trading session.

### 4.2 Compilation and trading deployment (planned)

FactorCompiler reconstructs supported archives as local Python callables. C++ compilation, remote publishing, a trading gateway and a deployment button are outside the current implementation. A future trading integration would need confirmation from the execution system before it could report a successful deployment.

## 5. Data Center

### 5.1 Exchange metadata (basic path implemented)

Selecting Binance, OKX or another market type calls `/api/exchange_meta` for USDT symbols and timeframes. Where available, the backend sorts by 24-hour quote volume (`quoteVolume`) and returns at most 200 symbols. On a network failure it may fall back to a static list. The list alone cannot confirm connectivity or symbol availability.

### 5.2 Batch selection and coverage (implemented)

Select multiple symbols and timeframes and one market type. The API accepts an array of market types. The current UI exposes a single market selector. After a short delay following a selection change, the table shows file existence, first and last timestamps and row count for each combination. Check for individual missing bars separately.

### 5.3 Download and final status (implemented)

The downloader supports `merge`, `fill_gap` and `overwrite`. Combinations within a batch run sequentially. Separate jobs can run concurrently. WebSocket conveys status updates. The console shows progress, gap repair and summarized errors without logging every bar.

Each task has a unique ID and ends as `completed`, `failed` or `partial_failed`. Progress can reach 100% even when some or all downloads failed. The UI buffers current-task events while submitting and retains the final state if a stale progress event arrives. A monitor disconnect displays a connection warning. Reconnecting restores the latest in-memory state if the backend is still running.

## Release and verification scope

Use Python ≥3.10 (tested with 3.12) and Node.js ≥22.12. Install from source with `pip install -e '.[test]'`. Use `npm ci`, `npm test` and `npm run build` in the frontend. The frontend connects to local port 8000 and the development page uses port 5173. The service has no complete authentication or multi-tenant system.

Chinese, English and German are available for navigation, Dashboard and Inspector. Some pages and logs retain their original language. Controlled component and hook tests cover event handling. Browser network behavior, recovery across processes, remote deployment and trading performance require separate evaluation.
