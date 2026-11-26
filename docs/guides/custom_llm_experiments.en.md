# Custom LLM miners and experiment records

[简体中文](custom_llm_experiments.md) · [English documentation](../README.en.md)

`MyCustomLLM` is a reference implementation in the user workspace. You can change prompts, candidate generation, reflection and top-K selection without editing `core/`. A separate `LLMExperimentRecorder` handles files, JSONL events and output diagnostics.

Install core dependencies from the source checkout using the [product guide](../FactorMiner_Product_Report_EN.md). This guide uses the checkout’s `user_workspace`, which is distributed separately from the wheel.

## Files and responsibilities

| File | Responsibility | Usually edited by the researcher? |
| --- | --- | --- |
| `user_workspace/custom_miners/my_custom_llm.py` | Prompts, candidates, evaluation, reflection and top-K selection | Yes |
| `user_workspace/experiment_tools/llm_recorder.py` | Manifest, events, replay and output hashes | No |
| `user_workspace/configs/configLLM_experiment.json` | Data, model, concurrency and experiment switches | Yes |
| `user_workspace/experiment_tools/summarize_llm_run.py` | Candidate table and research notes | No |
| `user_workspace/experiment_tools/audit_llm_holdout.py` | Re-evaluation on a fixed window and similarity checks | No |

## Extension points

In `MyCustomLLMMiner`:

1. `initialize_search_space()` configures the fields, sandbox and API.
2. `generate_candidates()` defines the prompt and how model responses become code.
3. `_extract_code()` handles the provider’s response format.
4. `update_model()` selects successful and failed examples for reflection.
5. `state.population` holds the final top-K candidates the Director can archive.

The recorder observes these events. It does not rank candidates or change their output values.

## API credentials

Put environment variable names in the configuration and keep key values in the environment:

```json
{
  "llm_api_config": {
    "keys_env": ["AI_API_KEY", "OPENAI_API_KEY"],
    "model_env": "AI_MODEL",
    "base_url_env": "AI_API_BASE",
    "base_url": "https://api.openai.com/v1/chat/completions",
    "model": "gpt-4.1-mini"
  }
}
```

The runtime checks available keys in order. Environment variables can override the default model and URL. The client appends `/chat/completions` to compatible API root URLs as needed. Keep real keys out of JSON, Python, READMEs and commits. Set variables in the process that starts the CLI or backend. Creating a `.env` file alone does not load it.

The project’s `.gitignore` excludes:

- `.env` and common local variants.
- `factor_db/`.
- `user_workspace/experiments/`.

The recorder manifest stores the model, endpoint and environment variable names without saving key values. The factor archive recursively removes credential fields from provenance and records the model actually used for live API generation. Fallback candidates are identified separately. Parent references retain the values the miner declared and may not resolve to an archived factor ID.

## Enable or disable recording

Omit `experiment.record_dir` for a recorder that creates no files:

```json
{
  "paradigm": "MyCustomLLM",
  "population_size": 3
}
```

To retain the full experiment record, use:

```json
{
  "experiment": {
    "name": "llm_btc_1m_research",
    "record_dir": "user_workspace/experiments/llm_btc_1m",
    "create_run_subdir": true,
    "require_live_api": true,
    "allow_fallback": false
  }
}
```

- `create_run_subdir` gives each run its own directory.
- `require_live_api` stops immediately if no usable credential is available.
- `allow_fallback=false` prevents a random formula from standing in for a live model candidate when the API fails.

Recording saves full prompts and raw model responses. Control who can access these files and which backups include them even when Git ignores the directory.

## Run without the Web UI

Set the variables referenced by the configuration in the execution environment, then run:

```bash
python -m core.cli mine \
  --miner MyCustomLLM \
  --config user_workspace/configs/configLLM_experiment.json \
  --user-dir user_workspace
```

The Director still writes archived factors to `factor_db/metadata`, `factor_db/sources` and `factor_db/values`. The experiment directory records how the candidates were generated.

## Generate reports

```bash
python user_workspace/experiment_tools/summarize_llm_run.py \
  user_workspace/experiments/llm_btc_1m

python user_workspace/experiment_tools/audit_llm_holdout.py \
  user_workspace/experiments/llm_btc_1m \
  --config user_workspace/configs/configLLM_experiment.json
```

The main outputs are:

- `events.jsonl` — raw event records.
- `candidate_results.csv` — candidates, code, errors and training metrics.
- `research_report.md` — run summary.
- `holdout_results.csv` — re-evaluation on the specified test window.
- `holdout_similarity_spearman.csv` — similarity among candidate outputs.
- `holdout_report.md` — test-window conclusions and research limitations.

## Interpreting the evidence

- Passing the execution sandbox leaves the research explanation to be checked separately.
- Positive test-window RankIC alone does not establish tradability. A window used for candidate selection or tuning is validation data. Reserve a separate untouched final holdout.
- Two implementations with different comments and identical outputs represent the same result.
- Live-trading conclusions require portfolio construction and a model of fees, slippage and funding costs.
