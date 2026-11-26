# Writing a fitness hook and interpreting its metrics

[简体中文](fitness_hook_guideline.md) · [English documentation](README.en.md)

Use the [product guide](FactorMiner_Product_Report_EN.md) to install the source checkout and prepare local data. A custom hook replaces the default score. To retain the coverage penalty, implement it in your hook.

## Default score and coverage

The evaluator records Pearson IC, Spearman RankIC and turnover. The default score is `abs(RankIC) × 100`. When output coverage falls below 20%, the evaluator multiplies this by `(coverage / 0.20) ** 2`. A strongly negative RankIC can produce a high score. Check the signal direction separately. The coverage penalty alone cannot rule out overfitting.

| Where it appears | What the value measures |
| --- | --- |
| Default evaluator coverage | For a Series, non-NaN outputs divided by its length. For a DataFrame, drop each row containing any missing value, then divide the remaining cell count by the original matrix size. There is no additional joint mask with the labels. |
| CLI Inspector coverage | Observations with both a valid factor and a valid label divided by all observations. Cross-asset coverage counts matrix cells. |
| Data Center coverage table | First and last timestamps and row count for each local file. Bar-level completeness and factor coverage require separate checks. |
| Evaluator turnover | Mean absolute change in factor values across time, averaged across assets for cross-sectional input. Its units are those of the factor. |
| CLI single-asset turnover diagnostic | Mean absolute change divided by the mean absolute factor value, using jointly valid observations. This is a normalized measure. |

CLI Inspector scales IR by a fixed `sqrt(252)` regardless of bar frequency or crypto calendar. The fraction of positive IC observations measures the sign of a correlation, not a trade win rate. Treat a missing value, a genuine zero, a negative score and a metric that was never computed distinctly.

## Register a hook

Save this example as `user_workspace/custom_fitness/coverage_rankic.py`. The loader scans the workspace’s `custom_miners`, `custom_operators` and `custom_fitness` subdirectories. A file placed directly at the workspace root is outside that discovery path.

```python
from core.miner.registry import EvaluatorRegistry

@EvaluatorRegistry.register_fitness_hook("coverage_rankic")
def coverage_rankic(factor_values, returns, base_metrics):
    # Values and returns are aligned Series or DataFrames with matching asset columns.
    # The target configuration sets the return horizon, entry and return type.
    # This uses the default output coverage, without a joint valid-label mask.
    valid_count = factor_values.dropna().count()
    if hasattr(valid_count, "sum"):
        valid_count = valid_count.sum()
    total_count = factor_values.size
    coverage = float(valid_count / total_count) if total_count else 0.0
    penalty = (coverage / 0.20) ** 2 if coverage < 0.20 else 1.0
    score = abs(float(base_metrics["RankIC"])) * 100.0 * penalty
    return {
        "fitness_score": score,
        "coverage": coverage,
        "coverage_penalty": penalty,
    }
```

The hook accepts `factor_values, returns, base_metrics` and must return a finite number or a dictionary with a finite `fitness_score`. Zero and negative scores are valid. A missing, NaN or infinite primary score fails that candidate, and the remaining candidates continue. Extra diagnostics are preserved in the archive. HTTP and WebSocket responses serialize nonfinite diagnostic values as `null` to indicate unavailability.

`base_metrics` includes `IC`, `RankIC` and `Turnover`. Consider the factor’s units and scale before penalizing changes in its value. A `Turnover` value of `0.8` does not establish 80% account turnover.

## Complete configuration example

Follow the product guide to prepare 1h BTC perpetual futures data. Save this JSON in a new file, `user_workspace/configs/fitness_example.json`, so that you retain existing experiment configurations.

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
  "search_space": {
    "allowed_operators": ["add", "sub", "mul", "div", "ts_mean", "ts_std"]
  },
  "fitness": {"hook": "coverage_rankic"},
  "evaluation": {"max_workers": 2}
}
```

From the root of the checkout, run:

```bash
factorminer mine --miner MyCustomGP \
  --config user_workspace/configs/fitness_example.json \
  --user-dir user_workspace --iterations 1
```

The CLI’s `--miner` option selects the algorithm. A `miner` field in the configuration does not select it. An unknown hook name or a workspace extension that fails to import stops startup before the engine reads market data. Correct the name or dependency. Loading the complete workspace requires all its extensions to import successfully.

This GP score uses the mining window. Filling in `test_period` does not automatically run a separate test. Use Inspector to re-evaluate another period and reserve a final test set untouched by candidate selection. Record changes to labels, coverage or scoring as distinct experiments.
