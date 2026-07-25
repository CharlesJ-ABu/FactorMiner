# 自定义 LLM 与实验留痕指南

`MyCustomLLM` 是用户工作区中的参考实现。用户可以修改 Prompt、候选生成、
Reflection 和 Top-K 选择，而不需要改动 `core/`。实验文件写入、JSONL 事件和输出
诊断由独立的 `LLMExperimentRecorder` 负责，避免研究留痕逻辑淹没自定义 Miner。

## 文件分工

| 文件 | 职责 | 通常是否需要修改 |
|---|---|---|
| `user_workspace/custom_miners/my_custom_llm.py` | Prompt、候选、评估、Reflection、Top-K | 是 |
| `user_workspace/experiment_tools/llm_recorder.py` | Manifest、事件、回放、输出哈希 | 否 |
| `user_workspace/configs/configLLM_experiment.json` | 数据、模型、并发与实验开关 | 是 |
| `user_workspace/experiment_tools/summarize_llm_run.py` | 候选表和研究底稿 | 否 |
| `user_workspace/experiment_tools/audit_llm_holdout.py` | 固定测试窗口复评与相似度 | 否 |

## 用户主要扩展点

在 `MyCustomLLMMiner` 中：

1. `initialize_search_space()` 配置字段、沙盒和 API；
2. `generate_candidates()` 定义 Prompt 与模型回答如何转为代码；
3. `_extract_code()` 处理供应商返回格式；
4. `update_model()` 决定哪些成功/失败案例进入 Reflection；
5. `state.population` 保存最终允许 Director 入库的 Top-K 候选。

Recorder 只观察这些事件，不参与候选排名，也不会改变因子值。

## API 凭证

配置文件只保存环境变量名称：

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

运行时会按顺序读取可用密钥；环境变量提供的模型和地址可以覆盖默认值。
OpenAI-compatible 根地址会自动补全 `/chat/completions`。不要把真实密钥写入
JSON、Python、README 或提交记录。

项目 `.gitignore` 默认忽略：

- `.env` 和常见本地变体；
- `factor_db/`；
- `user_workspace/experiments/`。

Recorder 的 Manifest 只保存模型、地址和环境变量名称，不保存密钥值。正式因子
provenance 在落盘前也会递归移除凭证字段。

## 启用或关闭实验留痕

不设置 `experiment.record_dir` 时，Recorder 是无输出对象，不创建文件：

```json
{
  "paradigm": "MyCustomLLM",
  "population_size": 3
}
```

需要保留完整研究证据时：

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

- `create_run_subdir`：每次运行独立归档；
- `require_live_api`：没有可用凭证时立即停止；
- `allow_fallback=false`：API 失败时不以随机公式冒充真实模型候选。

实验模式会保存完整 Prompt 与模型原始回答，因此即使目录已被 Git 忽略，也应按
研究数据管理要求限制访问和备份范围。

## 无头运行

先在运行环境中设置配置引用的变量，再执行：

```bash
python -m core.cli mine \
  --miner MyCustomLLM \
  --config user_workspace/configs/configLLM_experiment.json \
  --user-dir user_workspace
```

真实因子仍由 Director 写入 `factor_db/metadata`、`factor_db/sources` 和
`factor_db/values`；实验目录额外保存生成过程，不替代正式因子档案。

## 生成报告

```bash
python user_workspace/experiment_tools/summarize_llm_run.py \
  user_workspace/experiments/llm_btc_1m

python user_workspace/experiment_tools/audit_llm_holdout.py \
  user_workspace/experiments/llm_btc_1m \
  --config user_workspace/configs/configLLM_experiment.json
```

主要输出：

- `events.jsonl`：逐事件原始记录；
- `candidate_results.csv`：候选、代码、错误和训练指标；
- `research_report.md`：运行摘要；
- `holdout_results.csv`：测试窗口复评；
- `holdout_similarity_spearman.csv`：候选同质化检查；
- `holdout_report.md`：测试期结论与研究边界。

## 证据边界

- 代码通过沙盒不代表研究解释正确；
- 测试窗口 RankIC 为正不代表因子可以交易；
- 注释不同但输出相同的代码不能算作独立发现；
- 未加入手续费、滑点、资金费率和组合构建前，不应给出实盘结论。
