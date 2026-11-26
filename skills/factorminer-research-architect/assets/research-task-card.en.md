[简体中文](research-task-card.md)

# Factor Research Task Card

> Status labels: Provided / Provisional assumption / Verified / Unverified

## 1. Research question

- Falsifiable proposition to test:
- Research type: single-asset time series / multi-asset cross section / mixed
- Current level of evidence:

## 2. Research hypothesis and counterhypothesis

### Main hypothesis

### Counterhypothesis or alternative explanation

## 3. Market, assets, frequency, and timing

- Market and instrument type:
- Asset or asset universe:
- Data frequency:
- `feature_end`:
- `decision_time`:
- `entry_time`:
- Prediction horizon and `label_end`:

## 4. Data requirements

- Data fields and sources:
- Point-in-time availability:
- Data coverage requirements:
- Treatment of missing values, anomalies, corporate-action adjustments, or contracts:

## 5. Feature design

| Feature | Definition | Window/lag | Normalization | Time of availability |
| --- | --- | --- | --- | --- |

## 6. Label design

- Label formula:
- Entry/exit prices:
- Horizon:
- Treatment of overlapping labels:
- Risk or market neutralization:

## 7. Baselines, controls, and ablations

- Naive baseline:
- Inverted signal:
- Ablations:
- Breakdowns by market regime, period, or asset:

## 8. Method selection

- Preferred method: explicit formula / parameter grid / GP / RL / NN / LLM
- Reason for the choice:
- Required artifacts:
- Main costs and risks:
- Reasons for not selecting other methods:

## 9. Data splits

| Data role | Time range | Purpose | Repeated inspection allowed? |
| --- | --- | --- | --- |
| Training |  |  |  |
| Validation |  |  |  |
| Final test |  |  | No |

- Warm-up:
- Purge/embargo:
- Fitting scope for scalers, thresholds, and feature selection:

## 10. Evaluation metrics and acceptance criteria

- IC / RankIC definitions and statistical axis:
- ICIR and statistical significance:
- Coverage:
- Turnover definition:
- Decay:
- Quantiles:
- Stability:
- Acceptance/rejection rules frozen in advance:

## 11. Leakage, misalignment, and overfitting review

- Future data:
- Feature–label alignment:
- Full-sample fitting:
- Overlapping labels:
- Survivorship and point-in-time bias:
- Multiple testing and test-set peeking:
- Dependence on a single regime or asset:

## 12. Implementation plan

- Framework-independent implementation steps:
- Configuration, source code, environment, and random seeds to retain:
- Minimum validation:
- Full experiment:

## 13. FactorMiner implementation (enhanced mode only)

- Evidence from environment detection:
- Existing capabilities to reuse:
- Extension point: configuration / Operator / Fitness Hook / Custom Miner / Inspector
- Files to add or modify:
- Configuration mapping:
- Execution entry point:

## 14. Actual execution and Inspector results (only after execution)

- Run status and artifacts:
- Mining metrics:
- Inspector re-evaluation:
- Differences between mining and re-evaluation:
- Warnings and failures:

## 15. Limits of the conclusions

- What the current evidence supports:
- What the current evidence does not support:
- Costs, slippage, capacity, portfolio constraints, or execution constraints not yet included:
- Evidence needed to support a stronger conclusion:

## 16. Key decisions still requiring confirmation

1.
2.
3.
