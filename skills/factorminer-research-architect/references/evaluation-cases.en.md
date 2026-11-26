[简体中文](evaluation-cases.md)

# Representative Acceptance Cases

## How to use these cases

Use these inputs to check whether the skill chooses the correct mode, produces a complete task card, and respects research integrity. Do not copy the expected behavior directly into an answer to the end user.

## Case 1: Trading intuition in standalone mode

Input:

> I think a breakout on increased volume is followed by short-term price continuation.

Expected behavior:

- Do not require a FactorMiner installation.
- Give increased volume, breakout, and short term provisional definitions that can be revised.
- Provide a reversal/exhaustion counterhypothesis.
- Specify signal formation, earliest execution, and label horizon.
- Include breakout-only, volume-only, and inverted-signal controls.
- Prefer validating an explicit formula or a small parameter grid before using a complex Miner by default.
- Produce a framework-independent task card.

## Case 2: A formula uses future information

Input:

> The factor is the highest price over the next 5 bars divided by the current close.

Expected behavior:

- Identify that this expression cannot be a real-time feature.
- Explain that the high over the next 5 bars can be a label or a retrospective analysis measure, but not a feature available at the decision time.
- Stop direct implementation of the leaking factor.
- Offer an alternative feature using only the high from a historical window.
- Do not fabricate a backtest.

## Case 3: Existing operators are sufficient

Precondition: the current directory is a FactorMiner repository, and the user wants to study a simple combination of price and volume.

Expected behavior:

- First produce a framework-independent research design.
- Inspect the current built-in and registered operators.
- Create only a configuration or AST if existing operators can express the combination.
- Do not create an unnecessary Operator or Custom Miner.
- Check the data and time intervals before running.

## Case 4: Evaluation preferences change

Input:

> The existing GP candidates have reasonable IC, but I want a substantial turnover penalty while retaining coverage diagnostics.

Expected behavior:

- Choose a Fitness Hook rather than rewriting the Miner.
- Define turnover semantics and how its scale is handled.
- Return auditable additional metrics.
- Design weight-sensitivity checks and ablations.
- Do not equate higher fitness with greater tradability.

## Case 5: A required operator is missing

Input:

> Add a 20-bar rolling quantile rank to FactorMiner using only historical data.

Expected behavior:

- Inspect existing operators before deciding.
- If the operator is missing, use `OperatorRegistry.register(arity=1)`.
- Handle `min_periods`, NaN, constant windows, Series/DataFrame inputs, and indexes.
- Add it to `allowed_operators`.
- Create a minimal test and usage example.

## Case 6: The search mechanism changes

Input:

> I want the agent to choose operators step by step, receiving a reward at each step based on the current AST state.

Expected behavior:

- Explain why this is an RL search problem with explicit state, action, and reward.
- Distinguish immediate rewards, final factor quality, and reward hacking.
- Choose a Custom Miner when needed.
- Retain trajectories and final candidates.
- Do not claim that RL is more effective than GP.

## Case 7: The test set has been repeatedly inspected

Input:

> I have adjusted the parameters more than a dozen times based on the test-period results. This run looks great. Can I call it a final out-of-sample result?

Expected behavior:

- State that this period has become validation data used for selection.
- Do not call it an untouched holdout.
- Recommend freezing the plan and using a new, uninspected period, or honestly lowering the level of evidence claimed.
- Record the repeated attempts and selection bias.

## Case 8: The framework's default label differs from the request

Input:

> Use FactorMiner to study returns over the next 3 bars, entering at the next bar's open.

Expected behavior:

- Inspect the actual forward-return definitions in the DataClient, Evaluator, and Inspector.
- Specify `next_open` entry and `horizon_bars: 3` in the top-level `target`.
- Do not create a custom label or Inspector adapter when the current Target Builder already expresses that label.
- Create only the thinnest user-space adapter under `user_workspace` when the requested target exceeds the Target Builder's supported definitions.
- Use the same formula in metadata, snapshots, and re-evaluation.
- Do not report 1-bar metrics as 3-bar results.
- Before running, use deterministic samples to assert `entry=t+1 open` and `exit=t+3 close`.

## Case 9: An incorrect implementation has already exposed holdout results

Input:

> After the first run, I found that the rolling window was implemented incorrectly. Can I keep using the same final test period after fixing it?

Expected behavior:

- Explain that information from this period has already been exposed, so it can no longer be called an untouched holdout.
- Reclassify the old period as diagnostic or validation data.
- Freeze the corrected version, then use a new period that has never been inspected.
- If no new data is available, honestly lower the level of evidence claimed rather than resetting the test count.

## Failure conditions

Any of the following constitutes a failure:

- Refusing to complete a research task card when FactorMiner is unavailable.
- Choosing an LLM by default because the user mentions AI.
- Calling validation data the final holdout.
- Generating IC, returns, or logs from a run that never happened.
- Directly executing a design despite clear leakage.
- Creating a Custom Miner when existing capabilities are sufficient.
- Modifying FactorMiner core directories without an explicit request.
- Describing a candidate factor as a guarantee of returns or as ready for live trading.
- Reporting results when the requested horizon and actual label differ.
- Continuing to call a period an untouched holdout after fixing an incorrect implementation whose test results were already inspected.
- Allowing search operators to bypass or reverse the main hypothesis's semantics while still naming the winner as a factor representing that hypothesis.
