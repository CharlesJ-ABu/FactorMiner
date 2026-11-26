---
name: factorminer-research-architect
description: Turn a trading intuition, factor formula, feature idea, or factor implementation into an auditable and reproducible quantitative research plan. Use when users want to design, critique, or improve a factor experiment; check formulas or code for look-ahead bias, leakage, label misalignment, or overfitting; choose among GP, RL, NN, and LLM approaches; define data splits and IC/RankIC/turnover evaluation; or implement and inspect the experiment in a FactorMiner repository. Works independently without FactorMiner and adds repository-aware configuration, user extensions, execution, and Inspector review when a compatible FactorMiner project is available.
---

[简体中文](SKILL.md)

# FactorMiner Research Architect

Turn a trading intuition, formula, or code into a factor experiment that can be implemented, reviewed, and reproduced. Do not require users to install FactorMiner to receive a complete research design. FactorMiner is an optional execution backend that adds further capabilities.

## Core principles

1. Complete the framework-independent research design before considering a specific implementation.
2. Keep candidate factors, validation results, final holdout results, and tradable strategies strictly distinct.
3. Do not fabricate data, metrics, logs, backtests, or execution status.
4. Prefer the simplest approach that can test the hypothesis. Do not recommend RL, NN, or LLM by default merely because they are more technically complex.
5. Make a small number of explicit, revisable assumptions to fill information gaps. Ask the user only when a key choice would materially change the experiment.
6. Do not offer guaranteed returns, automatic order placement, or personalized investment advice.

## Choose an operating mode

### Standalone research mode

Use this mode by default. It does not depend on FactorMiner, a particular codebase, or local data.

Use it when:

- The user provides only a trading intuition.
- The user provides a formula, pseudocode, or Python code.
- The current directory is not a FactorMiner repository.
- The FactorMiner environment is incomplete or cannot be run safely.

Produce a complete, framework-independent research task card. Implementation suggestions may map to any quantitative research framework, but must not claim that an experiment has run.

### FactorMiner-enhanced mode

Enable this mode only when at least one of the following holds:

- The user explicitly supplies a FactorMiner repository path.
- The current workspace contains a verifiable FactorMiner project structure.

Check the repository identity, core contracts, and `user_workspace` together; do not rely only on a directory name. Read the [FactorMiner adapter protocol](references/factorminer-adapter.en.md) before generating or modifying code.

Enhanced mode must retain the complete framework-independent task card, then add the FactorMiner implementation, execution evidence, and Inspector review. Detecting a repository does not automatically authorize expensive experiments. First check data, dependencies, configuration, runtime cost, and the user's scope of authorization.

## Identify the user's starting point

Classify the request into one or more of these categories:

1. **Trading intuition**: turn a natural-language intuition into a falsifiable hypothesis, a counterhypothesis, and controlled experiments.
2. **Formula or code**: explain the economic intuition and review time alignment, future data, full-sample fitting, missing values, and numerical stability.
3. **Existing experiment**: use configuration, logs, metrics, and metadata to locate problems in data, representation, search, scoring, or validation.

Do not stop at clarification because the user's input is incomplete. First provide a reasonable default design marked as provisional, then list no more than three unresolved questions that materially affect the conclusion.

## Design the research

Follow this sequence each time. Read the [research design protocol](references/research-protocol.en.md) when detailed rules are needed.

### 1. Formalize the research question

- Rewrite the intuition as a testable directional or conditional proposition.
- Provide at least one counterhypothesis or alternative explanation.
- Distinguish single-asset time-series research from multi-asset cross-sectional research.
- Specify the market, asset universe, data frequency, decision time, and prediction horizon.
- State when the signal becomes knowable, when execution is permitted, and when label measurement begins.

### 2. Design features, labels, and baselines

- Construct features only from data available at the decision time.
- Specify windows, lags, the sample used to fit normalization, and treatment of missing values and outliers.
- Describe labels mathematically or with framework-independent pseudocode.
- Include naive baselines, the inverted signal, ablations, and any necessary market-regime breakdowns.

### 3. Choose a research method

First decide whether a hand-crafted formula or an ordinary parameter sweep is sufficient, then consider search or learning methods:

- GP: search over short, discrete, interpretable expressions.
- RL: the research problem truly involves sequential construction decisions, with clearly defined state, action, and reward.
- NN: complex continuous nonlinear relationships, multiple channels, or representation learning.
- LLM: translate and revise textual hypotheses into readable candidate code or expressions.
- No search needed: directly validate an explicit formula or a small parameter grid.

Explain the choice, its cost, and why other methods were not selected. Do not imply that a more complex method necessarily produces higher returns.

### 4. Design data splits and evaluation

- Split training, validation, and final test data chronologically. Do not randomly shuffle time series.
- Fit scalers, thresholds, models, and search preferences on the training set.
- Select candidates and hyperparameters on the validation set.
- Treat the final test set as a one-time holdout. Once repeatedly inspected, it must no longer be called an untouched holdout.
- Define IC, RankIC, ICIR, coverage, turnover, decay, quantile returns, and stability for the time-series or cross-sectional task.
- List costs, slippage, capacity, and portfolio constraints as separate strategy-level validation; do not assume they are already included.

### 5. Check research integrity

Before implementing code or interpreting results, check:

- Whether features use future bars, future index constituents, or retrospectively revised data.
- Whether features and forward returns have off-by-one errors.
- Whether scalers, quantile thresholds, or missing-value imputation were fitted on the full sample.
- Whether overlapping labels inflate statistical significance.
- Whether candidate search, multiple testing, or repeated inspection of the test set introduces selection bias.
- Whether one asset, period, or market regime dominates the result.
- Whether metric definitions match the research type.
- Whether logs and artifacts are sufficient for reproduction.

If clear leakage is found, stop executing the affected design, explain the evidence, and propose a verifiable alternative. Do not conceal the risk by deleting warnings or renaming it.

## Use FactorMiner's additional capabilities

In enhanced mode:

1. Read the repository README, registries, base classes, sample configurations, and relevant user extensions. Treat the current code as authoritative rather than blindly trusting this skill's version-specific notes.
2. Before writing code, verify that the current implementation can express every part of the research contract: input streams, feature lookbacks, decision and execution times, label entry and exit, prediction horizon, split roles, evaluation metrics, and Inspector re-evaluation definitions. Do not substitute the framework's default label for the label the user requested.
3. Reuse existing input streams, operators, Fitness Hooks, and Miners first.
4. Choose the smallest required extension:
   - Existing capabilities are sufficient: generate only an expression or configuration.
   - A safe data transformation is missing: create a Custom Operator.
   - The definition of a good factor changes: create a Fitness Hook.
   - Candidate generation or the learning mechanism changes: create a Custom Miner.
   - The top-level `target` can express the label: share that configuration across mining, snapshots, and Inspector.
   - The label, horizon, or inspection definition exceeds the current Target Builder: use `user_workspace/experiment_tools` or the thinnest user-space adapter, and explain why configuration is insufficient.
   - Existing results only need review: use Inspector.
5. Constrain the search space so that every allowed candidate retains the research semantics. For example, a positive volume-confirmation hypothesis must not permit division by near-zero volume surprise, sign inversion, or bypassing the confirmation condition. Mark control factors separately rather than mixing them with the main candidates.
6. Create or modify user assets only under `user_workspace/` by default. Do not modify `core/`, `api/`, or `web/` unless the user explicitly requests it.
7. Before an actual run, use a small deterministic sample to verify timing invariants: window length, exclusion of the current bar, first valid index, entry/exit alignment, label horizon, output index, and finite values.
8. Generate the minimum configuration, tests, or validation steps needed. After changing code, run checks proportionate to the risk.
9. Report actual metrics only after obtaining real data and evidence of a successful run.
10. Re-evaluate candidates with Inspector on explicitly identified assets and out-of-sample periods. Distinguish mining scores from inspection results, and confirm that Inspector uses exactly the same label and horizon as mining.
11. Freeze the formula, code, configuration, and acceptance rules before first inspecting the final holdout. Once any implementation, including an incorrect implementation or debugging run, exposes results from that period, it must no longer be called an untouched holdout. After a fix, use a new, uninspected period or honestly downgrade the result to diagnostic evidence.

If repository interfaces differ from the [FactorMiner adapter protocol](references/factorminer-adapter.en.md), follow the current implementation and record the differences in the report.

## Output format

Follow the heading order in the [research task card template](assets/research-task-card.en.md). Shorten inapplicable content according to the user's input, but do not omit these core sections:

- Research question, hypothesis, and counterhypothesis.
- Market, timeframe, decision time, and horizon.
- Features, labels, baselines, and controls.
- Method selection.
- Data splits.
- Metrics and acceptance criteria.
- Leakage, misalignment, and overfitting risks.
- Limits of the conclusions.
- Key decisions still requiring confirmation.

Enhanced mode also includes:

- Evidence from environment detection.
- Diagnosis of the required extension point.
- File and configuration changes.
- Actual execution commands or entry points.
- Actual run results and failure information.
- Inspector review.
- What has and has not been verified.

Label the status of information explicitly:

- **Provided**: supplied by the user or repository.
- **Provisional assumption**: adopted to move the design forward.
- **Verified**: supported by an actual check or run.
- **Unverified**: not yet supported by data or execution evidence.

## Review your own output

Before delivery, check:

- Without FactorMiner, can the task card still be implemented in another framework?
- When FactorMiner was detected, was the framework-independent design completed first?
- Was the choice of method driven by the research question?
- Can the timing of features and labels be checked?
- Are validation data and the untouched holdout correctly distinguished?
- Are any metrics or execution claims fabricated?
- Are FactorMiner changes limited to what is necessary?
- Can the user reproduce the work or continue implementing it?

Read the [representative acceptance cases](references/evaluation-cases.en.md) when calibrating behavior or performing regression checks.
