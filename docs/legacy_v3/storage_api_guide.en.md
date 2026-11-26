# Factor storage API guide

[简体中文](storage_api_guide.md) · [English documentation](../README.en.md)

> Archived V3 documentation. For the current interface, see the [V4 product guide](../FactorMiner_Product_Report_EN.md).

## Overview

The V3 storage API accepts the fields needed for each operation and chooses the destination directory automatically.

## Core API

### 1. Save a technical indicator

```python
storage.save_technical_factor(
    factor_id: str,           # Unique factor ID
    name: str,               # Factor name
    function_code: str,      # Python function source
    description: str = "",   # Factor description
    category: str = "technical",  # Factor category
    entry_point: str = "calculate",  # Entry point function name
    imports: List[str] = None  # List of import statements
) -> bool
```

Saves a technical indicator under `technicals/`.
**Files written**
- Function source → `technicals/functions/{factor_id}.py`
- Factor definition → `technicals/definitions/{factor_id}.json`

**Example**
```python
from factor_miner.core.factor_storage import get_global_storage

storage = get_global_storage()

function_code = '''
def calculate(data: pd.DataFrame, **kwargs) -> pd.Series:
    """Calculate a simple moving average factor."""
    return data['close'].rolling(20).mean()
'''

success = storage.save_technical_factor(
    factor_id="sma_20",
    name="20-day simple moving average",
    function_code=function_code,
    description="Calculate the 20-day simple moving average",
    category="technical",
    imports=["import pandas as pd"]
)
```

### 2. Save a mined factor

```python
storage.save_minactor_factor(
    factor_id: str,           # Unique factor ID
    name: str,               # Factor name
    algorithm_name: str,      # Algorithm name
    model_file: str = "",     # Model filename
    description: str = "",    # Factor description
    category: str = "ml",     # Factor category
    performance_metrics: Dict = None  # Performance metrics
) -> bool
```

Saves a mined factor under `minactors/`.
**Factor definition** → `minactors/definitions/{factor_id}.json`

**Example**
```python
success = storage.save_minactor_factor(
    factor_id="ml_momentum_5d",
    name="5-day ML momentum factor",
    algorithm_name="ml_momentum_factor",
    model_file="ml_momentum_5d.pkl",
    description="A machine learning momentum factor with a five-day horizon",
    category="ml",
    performance_metrics={
        "ic": 0.15,
        "sharpe": 1.2,
        "win_rate": 0.65
    }
)
```

### 3. Save a model file

```python
storage.save_model(
    factor_id: str,           # Factor ID
    model_data: bytes,        # Model data as bytes
    model_type: str = "pkl"   # Model type (e.g. pkl or joblib)
) -> bool
```

Saves the model under `minactors/models/`.
**Model file** → `minactors/models/{factor_id}.{model_type}`

**Example**
```python
import pickle

# Prepare the model data
model_data = pickle.dumps({"model": "dummy_model", "version": "1.0"})

success = storage.save_model(
    factor_id="ml_momentum_5d",
    model_data=model_data,
    model_type="pkl"
)
```

### 4. Save evaluation results

```python
storage.save_evaluation(
    factor_id: str,           # Factor ID
    evaluation_data: Dict,    # Evaluation data
    source: str = "minactors" # Source (technicals or minactors)
) -> bool
```

Saves evaluation results.
**Evaluation results** → `{source}/evaluations/{factor_id}.json`

**Example**
```python
evaluation_data = {
    "ic_pearson": 0.15,
    "ic_spearman": 0.12,
    "sharpe_ratio": 1.2,
    "win_rate": 0.65,
    "long_short_return": 0.08
}

success = storage.save_evaluation(
    factor_id="ml_momentum_5d",
    evaluation_data=evaluation_data,
    source="minactors"
)
```

### 5. Save mining history

```python
storage.save_mining_history(
    session_id: str,          # Session ID
    session_data: Dict        # Session data
) -> bool
```

Saves mining history under `minactors/mining_history/`.
**Files written**
- Session summary → `minactors/mining_history/mining_sessions.json`
- Detailed results → `minactors/mining_history/mining_results_{session_id}.json`

**Example**
```python
session_data = {
    "session_id": "test_session_001",
    "config": {
        "symbols": ["BTCUSDT"],
        "timeframes": ["1h"],
        "algorithms": ["ml_momentum_factor"]
    },
    "results": {
        "total_factors": 3,
        "algorithms_used": ["ml_momentum_factor"],
        "factors": {
            "ml_momentum_5d": "factor_data_here",
            "ml_momentum_10d": "factor_data_here",
            "ml_momentum_20d": "factor_data_here"
        }
    },
    "status": "completed",
    "completed_time": "2024-01-01T12:00:00"
}

success = storage.save_mining_history(
    session_id="test_session_001",
    session_data=session_data
)
```

## Directory structure

```
factorlib/
├── technicals/               # Traditional technical indicators
│   ├── definitions/          # Factor definition files
│   ├── functions/           # Function source files
│   └── evaluations/         # Evaluation results
├── minactors/               # Mined factors
│   ├── definitions/         # Factor definition files
│   ├── models/              # ML models (.pkl)
│   ├── evaluations/         # Evaluation results
│   └── mining_history/      # Mining history
└── temp/                    # Temporary files
```

## Typical workflow

### 1. Save a technical indicator
```python
# 1. Prepare the function source
function_code = "def calculate(data): return data['close'].rolling(20).mean()"

# 2. Save it via the API
storage.save_technical_factor(
    factor_id="sma_20",
    name="20-day simple moving average",
    function_code=function_code
)
```

### 2. Save a mined factor
```python
# 1. Save the factor definition
storage.save_minactor_factor(
    factor_id="ml_momentum_5d",
    name="5-day ML momentum factor",
    algorithm_name="ml_momentum_factor"
)

# 2. Save the model file
model_data = pickle.dumps(trained_model)
storage.save_model("ml_momentum_5d", model_data)

# 3. Save evaluation results
storage.save_evaluation("ml_momentum_5d", evaluation_results)
```

### 3. Save mining history
```python
# Save the complete mining session
storage.save_mining_history(session_id, session_data)
```

## Benefits

1. **Ease of use.** The caller supplies the values and the API handles paths and layout.
2. **Type-based storage.** The factor type determines the storage location.
3. **Common interface.** All storage operations are available through one API.
4. **Automatic setup.** The API creates directories and writes files in the expected format.
5. **Backward compatibility.** The interface remains compatible with the existing V3 system.

## Implementation notes

1. **Unique IDs.** Each `factor_id` must be unique in the system.
2. **Function source.** A technical indicator requires a complete Python function.
3. **Model data.** Pass serialized bytes.
4. **Evaluation data.** Pass a dictionary.
5. **Error handling.** Each API operation returns a Boolean indicating success or failure.

## Complete example

The V3 example is `examples/storage_api_usage.py`.
