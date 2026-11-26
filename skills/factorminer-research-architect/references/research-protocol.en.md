[简体中文](research-protocol.md)

# Research Design Protocol

## Contents

1. Normalize the input
2. Timing semantics
3. Hypotheses and controls
4. Features and labels
5. Method selection
6. Data splits
7. Evaluation framework
8. Leakage and overfitting review
9. Levels of evidence

## 1. Normalize the input

Extract the following from the user's input:

- Research subject: market, asset class, asset, or asset universe.
- Signal source: price, volume, order book, fundamentals, news, or other data.
- Research type: single-asset time series, multi-asset cross section, or a combination.
- Decision frequency and prediction horizon.
- Direction: trend continuation, reversal, risk, cross-sectional ranking, or conditional filtering.
- Objective constraints: low turnover, stability, coverage, interpretability, or computational cost.

When information is missing, adopt conservative provisional values and explain how they affect the conclusion. Do not invent an optimal asset or timeframe.

## 2. Timing semantics

Specify four times for every experiment:

1. `feature_end`: the latest timestamp of data used by the feature.
2. `decision_time`: when the signal is formed.
3. `entry_time`: the earliest time at which execution is permitted.
4. `label_end`: the endpoint of the forward-return calculation.

Principles:

- If a feature uses the close, high, or volume of bar `t`, assume that information becomes available only after the bar closes.
- Without a more granular execution assumption, do not use `close[t]` to form the signal and simultaneously assume frictionless execution at that same `close[t]`.
- Write forward returns explicitly, for example as `close[t+h] / entry_price[t+1] - 1`.
- Rolling features may use only historical windows ending at or before `feature_end`.
- Account for publication delays, exchange timestamps, time zones, and data revisions when determining availability.

## 3. Hypotheses and controls

A research hypothesis must be falsifiable. For example:

> In liquidity-filtered BTC perpetual futures data at a 5-minute frequency, a signal combining unusually high relative volume with a price breakout above the previous window's high has a positive RankIC against returns over the next 3 bars.

Design at least one counterhypothesis or alternative explanation:

- Extreme volume indicates short-term exhaustion rather than continuation.
- The signal merely represents volatility exposure.
- Returns come from a small number of market regimes.
- Results are driven by trading hours, asset size, or liquidity.
- The factor is only another expression of an existing baseline.

Prioritize these controls:

- Price breakout alone.
- Relative volume alone.
- The inverted signal.
- A shuffled placebo test that preserves temporal block structure.
- Different windows, horizons, and market regimes.
- Comparisons with simple momentum, reversal, or volatility factors.

## 4. Features and labels

### Feature specification

Record the following for each feature:

- Data fields and sources.
- Calculation formula.
- Window and `min_periods`.
- Whether it is lagged.
- Treatment of missing values, zeros, outliers, and division by zero.
- Normalization method and fitting sample.
- Time-series or cross-sectional semantics.
- Index alignment after calculation.

Prefer rolling normalization or normalization fitted on the training period. Do not calculate means, variances, or quantiles over the full sample before splitting training and test data.

### Label specification

Specify:

- Price reference.
- Entry and exit times.
- Horizon.
- Simple or log returns.
- Whether market, sector, or risk exposures are removed.
- Treatment of overlapping labels.
- Treatment of missing assets in the cross section and survivorship bias.

### Code review checks

Look especially for:

- `shift(-n)` used in features rather than labels.
- Rolling windows with `center=True`.
- Full-sample `fit_transform`.
- Backward filling that carries future values into the past.
- Resampling that incorrectly makes an unfinished higher-timeframe bar available early.
- Merges without point-in-time/as-of semantics.
- Signals and returns using an unavailable execution price from the same bar.
- Filters that refer to a complete future interval.
- State, caches, or model weights shared across training and test splits.

## 5. Method selection

Ask three questions first:

1. Is the candidate space already small and explicit? If so, directly validate the formula or use a grid search.
2. Does the user need a new data transformation, evaluation preference, or search mechanism? Keep these three needs distinct.
3. Does added model complexity address a specific research need?

| Method | Suitable use | Required artifacts | Main risks |
| --- | --- | --- | --- |
| Explicit formula/grid | Specific hypothesis with few parameters | Formula, parameter grid, controls | Parameter snooping |
| GP | Search over short AST formulas | AST, operator space, elite candidates | Multiple testing, expression bloat |
| RL | Sequential construction is itself the research subject | State/action/reward, trajectories, policy | Reward hacking, unstable training |
| NN | Complex continuous relationships or representations with multiple channels | Schema, scaler, weights, channels | Leakage, limited interpretability, overfitting |
| LLM | Text-to-code candidate generation and iterative revision | Prompts, source code, feedback history | Unsafe code, hallucinations, test-set leakage |

An LLM is better suited to candidate generation or research assistance than to unconstrained conclusion generation. Every generated candidate still requires actual execution under a common evaluation protocol.

## 6. Data splits

Separate the roles:

- Training set: fit models and scalers, or perform candidate search.
- Validation set: select hyperparameters, formulas, and stopping conditions.
- Final test set: evaluate once after freezing the research plan.
- Rolling re-evaluation: examine stability across periods; it does not replace the final test.

The final test is considered used according to information exposure, not the number of successful commands. An incorrect formula, failed pipeline, debugging script, or incomplete report consumes the holdout if it reveals results from that period. After fixing the implementation, use a new, uninspected period or reclassify the old period as diagnostic or validation data.

Maintain chronological order and account for:

- Warm-up for the longest feature lookback window.
- Boundary contamination caused by the label horizon.
- Purge/embargo.
- Changes in market structure.
- Asset listings, delistings, and universe changes.
- Reduced effective sample size from overlapping labels.

When data is limited, narrow the conclusions honestly. Do not use random cross-validation to manufacture sample size.

## 7. Evaluation framework

First establish the statistical unit:

- Single-asset time series: calculate correlations and stability over time.
- Multi-asset cross section: calculate IC across assets at each timestamp, then summarize the resulting time series.
- Mixed research: report these separately rather than combining them into a single number with no clear meaning.

Core metrics:

- Pearson IC: linear correlation.
- RankIC: rank correlation, which is more robust to extreme values.
- ICIR: mean IC divided by the standard deviation of IC; state the annualization convention.
- Coverage: valid observations as a proportion of the theoretically available observations.
- Turnover: specify whether this measures changes in raw factor values, changes in rankings, or turnover in portfolio weights.
- Decay: signal decay across different forward horizons.
- Quantiles: forward returns by quantile group and their monotonicity.
- Stability: performance by period, asset, volatility level, and market regime.

Do not assess significance using only an ordinary t-statistic. Consider HAC, block bootstrap, or another robust method appropriate for serial correlation, cross-sectional dependence, and overlapping labels. When searching many candidates, report selection bias and consider suitable methods such as FDR, White's Reality Check, or the Deflated Sharpe Ratio.

Set acceptance thresholds before inspecting the final test set. Match thresholds to the frequency, assets, and sample length; do not use universal magic numbers.

## 8. Leakage and overfitting review

### Data leakage

- Was point-in-time data actually available?
- Were scalers, encoders, and missing-value treatments fitted only on training data?
- Did feature selection inspect validation or test labels?
- Does the multi-asset data contain survivorship bias?
- Does the same event or asset leak across splits?

### Search bias

- How many candidates and hyperparameter settings were tested?
- Were failed candidates and the complete search history retained?
- Was the final test set repeatedly inspected?
- Are only the best periods or assets reported?
- Was the hypothesis revised based on test results?

### Implementation risks

- Indexes, time zones, and bar-closing rules.
- Data ordering and duplicate timestamps.
- Propagation of missing values.
- Numerical overflow, division by zero, and infinite values.
- Time-series operators applied to cross-sectional data, or vice versa.
- Caches, random seeds, and dependency versions.

Add checks of semantic invariants for automatically generated or searched expressions:

- Does the candidate still express the original hypothesis, rather than merely share a few input fields?
- Can allowed operators bypass or reverse the direction, thresholds, or joint conditions?
- Do division, reciprocals, or normalization change the economic meaning near zero?
- Are main candidates clearly separated from inverted-signal and ablation controls?
- Does the actual label formula match the task card's entry, exit, and horizon exactly?

## 9. Levels of evidence

Use these levels when describing results:

1. **Research proposal**: only a hypothesis and plan exist; nothing has run.
2. **Candidate result**: performs well during training or search.
3. **Validation result**: performs well on the validation period used for selection.
4. **Final test result**: evaluated on a one-time holdout after the plan was frozen.
5. **Strategy-level evidence**: additionally incorporates costs, slippage, capacity, portfolio constraints, and execution constraints.
6. **Live-trading evidence**: comes from actual execution records.

Do not claim a higher level than the evidence supports. Statistical significance of a factor does not establish strategy tradability, and historical results do not guarantee future returns.
