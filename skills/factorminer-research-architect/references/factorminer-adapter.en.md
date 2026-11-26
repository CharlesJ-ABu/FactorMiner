[简体中文](factorminer-adapter.md)

# FactorMiner Enhanced-Mode Adapter Protocol

## Contents

1. Environment detection
2. Reading order
3. Choosing an extension point
4. Current V4 contracts
5. Configuration and execution
6. Inspector
7. Evidence and modification boundaries

## 1. Environment detection

Treat a directory as a FactorMiner repository only when most of these identifiers are present together:

- `README.md` describes FactorMiner.
- `core/miner/registry.py`.
- `core/miner/paradigms/base.py`.
- `core/miner/operator_runtime.py`.
- `core/commands/mine.py` or a working `factorminer mine` command.
- `user_workspace/`.

If you find only a similarly named folder or an incomplete copy, remain in standalone mode and report what is missing.

## 2. Reading order

Read according to the task's needs rather than loading the entire repository at once:

1. `README.md`: version, CLI, directories, and result contracts.
2. `core/miner/registry.py`: actual registration decorator signatures.
3. `core/miner/operator_runtime.py`: built-in operators and AST constraints.
4. `core/miner/paradigms/base.py`: Miner lifecycle.
5. `core/startup_validation.py`: validation before startup.
6. The closest relevant example in `user_workspace`.
7. Related configurations and Inspector implementation.

Repository code takes precedence over documentation. Explicitly identify any discrepancies.

## 3. Choosing an extension point

### Generate only an expression or configuration

Choose this when existing data streams and operators can express the research hypothesis. Do not create a Python file merely to demonstrate capability.

### Custom Operator

Use this only for a missing data transformation. Before creating one, confirm:

- Time-series or cross-sectional semantics.
- Input arity.
- Return type, index, and shape.
- Rolling window and `min_periods`.
- Behavior for NaN, Inf, division by zero, and constant series.
- Whether it is explicitly enabled in the GP/RL search space.

### Fitness Hook

Choose this when evaluation preferences change, such as low turnover, coverage, stability across periods, or a complexity penalty. The search mechanism stays the same.

### Custom Miner

Choose this only when candidate generation, search state, feedback-based learning, or artifact persistence must change. Prefer copying an example from the same paradigm.

### Label or horizon adaptation

First inspect the forward returns actually used by `RealDataClient`, the Evaluator, and Inspector. Prefer the top-level
`target` for specifying entry, exit, horizon, and return type. If the requirement still exceeds the current Target Builder:

- Do not silently accept the default next-period return.
- Prefer an auditable evaluation entry point in `user_workspace/experiment_tools`.
- Create the thinnest Custom Miner or user-space data adapter only if the search loop itself must use that label.
- Preserve the exact `forward_return_definition` and actual evaluation slice on the candidate.
- Ensure Inspector re-evaluation uses the same label formula instead of falling back to the framework default.

### Inspector

Choose this when an existing factor only needs review across assets or periods, or checks of decay, quantile returns, and turnover. Do not create a new Miner for that purpose.

## 4. Current V4 contracts

The following information is based on the local FactorMiner V4 implementation as of 2026-07-27. Verify it against the current repository before use.

### Operator

```python
from core.miner.registry import OperatorRegistry

@OperatorRegistry.register(arity=1)
def my_operator(series):
    return series
```

- Arity accepts only `1` or `2`.
- The function name is its registered name.
- Return a `pandas.Series` or `pandas.DataFrame`.
- Built-in operators include `add`, `sub`, `mul`, `div`, `ts_mean`, and `ts_std`.
- Enable operators through `search_space.allowed_operators`.

### Fitness Hook

```python
from core.miner.registry import EvaluatorRegistry

@EvaluatorRegistry.register_fitness_hook("my_hook")
def my_hook(factor_values, returns, base_metrics):
    return {"fitness_score": 0.0}
```

- The function must accept `factor_values, returns, base_metrics`.
- The core fields in `base_metrics` are `IC`, `RankIC`, and `Turnover`.
- Return a number or a dictionary containing `fitness_score`.
- Reference it through `fitness.hook`.

Do not copy `register_evaluator` from older documentation unless the current code actually provides that alias.

### Custom Miner

A Custom Miner inherits from `BaseFactorMiner` and implements:

1. `initialize_search_space()`.
2. `generate_candidates()`.
3. `evaluate_candidates()`.
4. `update_model()`.

Candidates must be subclasses of `FactorExpression`. Final artifacts must reach the result path that supports persistence. The current base class first retrieves results from `hall_of_fame`, then from `state.population` or `state.replay_buffer`. Do not update only the model without retaining candidates.

### Result metrics

The current core archive fields are:

- `metrics.IC`
- `metrics.RankIC`
- `metrics.Turnover`
- `metrics.fitness_score`

Preserve capitalization. Do not invent lowercase aliases.

## 5. Configuration and execution

A typical configuration includes:

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

Do not copy dates, assets, or the hook mechanically; generate them for the research task. FactorMiner primarily uses `mine_period` and `test_period`, but the research design must still distinguish training, validation for candidate selection, and the final holdout. If the configuration interface cannot directly express three splits, use separate configurations, experiment tools, or staged runs, and name each period's role accurately.

The top-level `target` configuration is shared by mining, snapshots, and Inspector. Omitting it preserves the historical default,
`close[t+1] / close[t] - 1`. For example, enter at the next bar's open and exit at the third bar's close:

```json
"target": {
  "type": "forward_return",
  "entry_price": "next_open",
  "exit_price": "close",
  "horizon_bars": 3,
  "return_type": "simple"
}
```

The current implementation supports `current_close` / `next_open` entry, `close` exit, and `simple` / `log`
returns. Labels are built independently for each research interval; records near an interval's end with insufficient future data are excluded. After a run, still verify
`data_lineage.target` and `forward_return_definition`. Do not present default 1-bar metrics as 3-bar results.

Typical execution entry points are:

```bash
factorminer mine --miner GP --config user_workspace/configs/example.json
factorminer mine --miner MyCustomGP --config user_workspace/configs/example.json --user-dir user_workspace
```

If the CLI is not installed, use:

```bash
python -m core.cli mine --miner MyCustomGP --config user_workspace/configs/example.json --user-dir user_workspace
```

Before running:

- Verify that dependencies and commands are available.
- Check data coverage for the required interval.
- Check configuration, registered names, and imports.
- Estimate experiment cost.
- Do not print or record API keys.
- Run a small smoke test before the full experiment.

## 6. Inspector

Inspector can resolve candidates from a factor ID, AST, or code. Common entry points:

```bash
factorminer inspect --factor "fac_id" --pairs "BTC/USDT:USDT"
factorminer inspect --ast "{...}" --pairs "BTC/USDT:USDT,ETH/USDT:USDT" \
  --start 2025-08-01 --end 2025-12-31 --timeframe 5m
```

During review, make explicit:

- The assets, periods, frequency, and instrument type used.
- Whether the period is validation data or the final holdout.
- Coverage, IC, RankIC, IR, Decay, Quantiles, and Turnover.
- Whether mining scores and Inspector re-evaluation agree.
- Missing snapshots or data must not be replaced with fabricated charts or metrics.

When the standard Inspector receives a factor ID, it inherits the original `target` from factor metadata and rebuilds labels using the same definition.
If the Inspector configuration explicitly supplies a different `target`, it overrides the metadata and a warning is logged. When inspecting an AST or code without a
factor ID, Inspector uses the configured `target`, or the compatible default when none is supplied. Retain the actual
`target`, exact formula, original JSON, and logs in the report.

Protect the final holdout with a separate freezing step:

1. Freeze the formula, search space, labels, code, configuration, and acceptance criteria.
2. Record every period whose results have already been inspected.
3. Choose only a period from which no results have previously been exposed as the untouched holdout.
4. Do not revise the plan after inspecting it.
5. If an implementation error requires a change, reclassify the old interval as diagnostic data and choose a new holdout, or lower the level of evidence claimed.

## 7. Evidence and modification boundaries

By default, suggestions or modifications may target:

```text
user_workspace/
├── configs/
├── custom_operators/
├── custom_fitness/
├── custom_miners/
└── experiment_tools/
```

Do not proactively modify these directories without an explicit user request:

```text
core/
api/
web/
```

Record the following in every enhanced-mode delivery:

- Detected repository version or supporting evidence.
- Contract files read.
- Files modified.
- Execution entry point and exit status.
- Actual artifact paths.
- Actual metrics.
- Failures, warnings, and unverified items.
