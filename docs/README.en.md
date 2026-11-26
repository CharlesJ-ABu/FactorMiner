# FactorMiner V4 documentation

[简体中文](README.md) · [Project overview](../README.md)

FactorMiner is a local, single-user workbench for quantitative factor research. To run your first experiment, start with the **[product and installation guide](FactorMiner_Product_Report_EN.md)**. It covers the source checkout, configuration, a complete GP example, CLI and Web UI workflows, file management and known limits.

## Architecture and requirements

| Document | What you will find |
| --- | --- |
| [V4 architecture](architecture/v4_architecture_design.en.md) | The inversion-of-control design, data feeds, evaluation, storage for different factor types, and restoration of executable factors. Design examples are marked separately from implemented behavior. |
| [Product requirements](FactorMiner_PRD.en.md) | Requirements and implementation status for GP, RL, NN and LLM mining, evaluation and research workflows. |
| [Web UI requirements](../web/WEB_UI_PRD.en.md) | Dashboard, Launchpad, Execution Console, Inspector and batch downloads. |
| [V4 development checklist](../TODO.en.md) | A historical milestone and task list. For current behavior use the product guide. |

## User guides

- [Networking and optional proxies](guides/vpn_setup.en.md) covers the local API, exchange connectivity and offline use.
- [Custom LLM experiments](guides/custom_llm_experiments.en.md) describes `MyCustomLLM`, environment variables, the independent recorder, headless runs and reports.
- [Writing a fitness hook](fitness_hook_guideline.en.md) explains the scoring signature, definitions of coverage and turnover, an executable example and error handling.
- [Custom miners, operators and fitness hooks](../README.md#custom-miners-operators-and-fitness-hooks) shows how to register extensions.
- [Frontend setup and checks](../web/README.en.md) covers Node.js, Vite and the connection to the backend.
- [Research skills](../skills/README.en.md) introduces the [research architect](../skills/factorminer-research-architect/SKILL.en.md), [task card](../skills/factorminer-research-architect/assets/research-task-card.en.md), [research protocol](../skills/factorminer-research-architect/references/research-protocol.en.md), [evaluation cases](../skills/factorminer-research-architect/references/evaluation-cases.en.md) and [FactorMiner adapter](../skills/factorminer-research-architect/references/factorminer-adapter.en.md).
- [LLM factor guardrails review](reviews/llm_guardrails_20260724.md) is the July 2026 engineering review, originally published in English.

## V3 archive

These historical guides describe earlier versions of `factorlib` and its APIs. Use them to interpret old artifacts and compare designs. Use the [V4 product guide](FactorMiner_Product_Report_EN.md) for current installation and testing.

| Archived guide | Subject |
| --- | --- |
| [V3 API reference](legacy_v3/api.en.md) | Historical endpoints and interfaces |
| [Factor storage guide](legacy_v3/factor_storage_guide.en.md) | V3 factor types, formats, example inventory and management operations |
| [FactorLib directory layout](legacy_v3/factorlib_structure_guide.en.md) | The split between technical indicators and mined factors |
| [Simplified architecture](legacy_v3/new_architecture_guide.en.md) | Migration to calling user functions directly |
| [Storage API guide](legacy_v3/storage_api_guide.en.md) | Saving definitions, models, evaluations and mining sessions |
| [User algorithm conventions](legacy_v3/user_algorithm_paradigm.en.md) | V3 metadata, algorithm entry points and validation |
