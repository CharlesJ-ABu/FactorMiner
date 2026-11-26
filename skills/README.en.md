[简体中文](README.md)

# FactorMiner Skills

This directory contains Agent Skills that work independently and gain additional capabilities when used in a FactorMiner repository.

## Available skills

### FactorMiner Research Architect

Path: [`factorminer-research-architect/SKILL.en.md`](factorminer-research-architect/SKILL.en.md)

Turn a trading intuition, formula, code, or existing experiment into a factor research task card that can be implemented, reviewed, and reproduced.

- **Standalone research mode**: no FactorMiner installation is required. Produce a framework-independent plan covering hypotheses, features, labels, data splits, evaluation, and leakage checks.
- **FactorMiner-enhanced mode**: after detecting a compatible repository, inspect existing operators and extension points, create `user_workspace` configurations or extensions, run actual experiments, and review them with Inspector.

Example invocation:

```text
Use $factorminer-research-architect:
I think a breakout on increased volume is followed by short-term price continuation. Please turn this into a rigorous factor experiment.
```

The three user-facing strings in [`agents/openai.yaml`](factorminer-research-architect/agents/openai.yaml) mean:

| Field | English text |
| --- | --- |
| `display_name` | FactorMiner Research Architect |
| `short_description` | Design quantitative factor experiments independently; execute and inspect them when connected to FactorMiner. |
| `default_prompt` | Use $factorminer-research-architect to turn my factor idea into a rigorous research task card. |

These translations are for English readers. The original configuration file remains the skill's UI metadata.

## Installation

After cloning the repository, copy the skill directory into the Codex skills directory:

```bash
git clone https://github.com/CharlesJ-ABu/FactorMiner.git
cd FactorMiner
mkdir -p "${CODEX_HOME:-$HOME/.codex}/skills"
cp -R skills/factorminer-research-architect "${CODEX_HOME:-$HOME/.codex}/skills/"
```

Restart Codex or open a new task, then invoke it as follows:

```text
Use $factorminer-research-architect:
I think a breakout on increased volume is followed by short-term price continuation. Please turn this into a rigorous factor experiment.
```

### Updating

Run `git pull` in the FactorMiner repository, then replace the installed directory:

```bash
rm -rf "${CODEX_HOME:-$HOME/.codex}/skills/factorminer-research-architect"
cp -R skills/factorminer-research-architect "${CODEX_HOME:-$HOME/.codex}/skills/"
```

### Uninstalling

```bash
rm -rf "${CODEX_HOME:-$HOME/.codex}/skills/factorminer-research-architect"
```

You can also skip installation and ask an agent to read `SKILL.md` directly from the repository. `SKILL.md` remains the standard discovery and installation entry point; [`SKILL.en.md`](factorminer-research-architect/SKILL.en.md) is its English companion.

## Compatibility and execution boundaries

- **Standalone research mode** does not require FactorMiner, market data, or a FactorMiner Python environment. It produces a framework-independent research task card and does not claim that an experiment has run.
- **FactorMiner-enhanced mode** requires a verifiable FactorMiner repository, local data, and project dependencies. Configurable forward-return labels require a version with the unified Target Builder, starting from commit `f97e249` or a later version.
- This skill follows the Codex `SKILL.md` directory format. Other agents can read the Markdown directly, but skill discovery and UI metadata compatibility depend on their products.
- Enhanced mode creates configurations or extensions only under `user_workspace/` by default. Detecting the repository does not automatically authorize expensive experiments.
- Check dependencies, data coverage, cost, and the scope of authorization before actual execution. The skill contains no market data, API keys, or actual experiment metrics.
- The skill does not itself upload local market data. How context and files are handled depends on the agent, model, and runtime the user chooses.

## Validation and scope of use

Check the skill's static structure with `python skills/validate.py`. For FactorMiner installation and product tests, see the [product guide](../docs/FactorMiner_Product_Report_EN.md). The supported setup is a local, single-user source checkout; the wheel alone does not include the complete workspace and frontend.

Standalone mode produces a research design and does not claim execution. Enhanced mode must verify the actual source, market data, configuration, labels, and artifacts. Configured labels and windows must agree. A test period used for selection can no longer be called an unseen holdout. Historical run records and passing static validation do not prove factor effectiveness, tradability, or future returns.

## Contribution conventions

- Give each skill its own directory with a short, hyphenated name and a `SKILL.md` file.
- The `SKILL.md` frontmatter must contain only `name` and `description`.
- Put detailed methods and version-specific interfaces in `references/`, and output templates in `assets/`.
- Base FactorMiner integration guidance on the current repository code.
- Do not commit fabricated metrics, real keys, private data, or local experiment artifacts.
- Run `python skills/validate.py` before committing.
