# FactorLib directory layout

[简体中文](factorlib_structure_guide.md) · [English documentation](../README.en.md)

> Archived V3 documentation. For the current interface, see the [V4 product guide](../FactorMiner_Product_Report_EN.md).

## Overview

V3 reorganized FactorLib into two directories: `technicals/` for traditional technical indicators and `minactors/` for mined factors.

## Directory tree

```
factorlib/
├── technicals/                    # Traditional technical indicators
│   ├── definitions/              # Factor definitions (JSON)
│   ├── functions/                # Factor functions (Python files)
│   └── evaluations/              # Factor evaluation results
├── minactors/                    # Mined factors
│   ├── definitions/              # Factor definitions (JSON)
│   ├── models/                   # ML model files (.pkl)
│   ├── evaluations/              # Factor evaluation results
│   └── mining_history/           # Mining history
└── temp/                         # Temporary cache
```

## Directory contents

### technicals/ — traditional technical indicators

Contains technical indicators such as moving averages, RSI and MACD.

- **definitions/** holds JSON definitions.
  - Includes metadata, parameters and computation logic.
  - Filename pattern: `{factor_id}.json`

- **functions/** holds Python functions for factor calculation.
  - Contains each factor’s calculation logic.
  - Filename pattern: `{factor_id}.py`

- **evaluations/** holds evaluation results.
  - Includes performance metrics and backtest results.
  - Filename pattern: `{factor_id}_evaluation.json`

### minactors/ — mined factors

Contains algorithmically mined factors, primarily machine learning factors.

- **definitions/** holds JSON definitions.
  - Includes information about the algorithm and the path to its model.
  - Filename pattern: `{algorithm_id}_{factor_name}.json`

- **models/** holds machine learning models.
  - Contains trained models in .pkl format.
  - Filename pattern: `{algorithm_id}_{factor_name}.pkl`

- **evaluations/** holds evaluation results.
  - Contains performance evaluations for mined factors.
  - Filename pattern: `{factor_id}_evaluation.json`

- **mining_history/** holds the mining history.
  - Includes session records and detailed results.
  - Filename pattern: `mining_results_{session_id}.json`

## Classification rules

### Traditional technical indicators (technicals/)

The following factors belong under `technicals/`.

- **Computation type** — `formula` or `function`
- **Categories** include `trend`, `momentum`, `volatility`, `volume` and `oscillator`.
- **Examples**
  - `sma.json` — simple moving average
  - `rsi.json` — relative strength index
  - `macd.json` — moving average convergence/divergence
  - `bb_upper.json` — upper Bollinger Band

### Mined factors (minactors/)

The following factors belong under `minactors/`.

- **Computation type** — `ml_model`
- **Category** — `ml` or a category containing `mining`, `ml_`, `statistical_`, `advanced_` or `technical_`
- **Examples**
  - `technical_mining_rsi_20.json` — RSI factor from a technical mining algorithm
  - `ml_momentum_5d.json` — ML momentum factor
  - `adaptive_ml_factor.json` — adaptive ML factor

## Storage changes

### The TransparentFactorStorage class

V3 updated `TransparentFactorStorage` to support this layout.

```python
class TransparentFactorStorage:
    def __init__(self, storage_dir: str = None):
        # New directory structure
        self.technicals_definitions_dir = self.storage_dir / "technicals" / "definitions"
        self.technicals_functions_dir = self.storage_dir / "technicals" / "functions"
        self.technicals_evaluations_dir = self.storage_dir / "technicals" / "evaluations"

        self.minactors_definitions_dir = self.storage_dir / "minactors" / "definitions"
        self.minactors_models_dir = self.storage_dir / "minactors" / "models"
        self.minactors_evaluations_dir = self.storage_dir / "minactors" / "evaluations"
        self.minactors_mining_history_dir = self.storage_dir / "minactors" / "mining_history"
```

### Automatic classification

The storage class selects a destination using the following rules.

1. **ML model factors** satisfy `computation_type == "ml_model"` or `category == "ml"`.
   - Write to `minactors/definitions/`.

2. **Traditional technical indicators** include all remaining factors.
   - Write to `technicals/definitions/`.

### Loading definitions

`load_factor_definition()` searches the directories in this order.

1. First look in `minactors/definitions/`.
2. Then look in `technicals/definitions/`.

## Migration statistics

### Files migrated

- **technicals/definitions/** — 306 files
- **technicals/functions/** — 162 files
- **technicals/evaluations/** — 268 files
- **minactors/definitions/** — 30 files
- **minactors/models/** — 20 files
- **minactors/mining_history/** — 9 files

### Totals

- **Traditional technical indicators** — 736 files
- **Mined factors** — 59 files
- **Total** — 795 files

## Benefits

1. **Separation.** Traditional indicators and mined factors have distinct directories.
2. **Maintenance.** Related files are kept together.
3. **Extensibility.** New factor types can use the appropriate directory.
4. **Backward compatibility.** The existing API remains available.
5. **Performance.** The layout reduces time spent looking for files.

## Usage examples

### Save a factor definition

```python
from factor_miner.core.factor_storage import TransparentFactorStorage

storage = TransparentFactorStorage()

# Save a traditional technical indicator
storage.save_function_factor(
    factor_id="sma_20",
    name="Simple moving average",
    category="trend",  # Saved under technicals/
    # ...
)

# Save a mined factor
storage.save_ml_factor(
    factor_id="ml_momentum_5d",
    name="ML momentum factor",
    category="ml",  # Saved under minactors/
    # ...
)
```

### Load a factor definition

```python
# The system searches the appropriate directory automatically.
factor_def = storage.load_factor_definition("sma_20")
factor_def = storage.load_factor_definition("ml_momentum_5d")
```

## Summary

The V3 layout grouped traditional and mined factors separately, retained compatibility with existing interfaces and supported efficient lookup. Grouping related files also simplified subsequent maintenance and extension.
