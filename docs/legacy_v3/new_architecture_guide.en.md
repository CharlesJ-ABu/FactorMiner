# Simplified factor architecture

[简体中文](new_architecture_guide.md) · [English documentation](../README.en.md)

> Archived V3 documentation. For the current interface, see the [V4 product guide](../FactorMiner_Product_Report_EN.md).

## Overview

This V3 revision removed the `functions/` directory and simplified storage and computation by calling the functions in `user_algo` directly.

## Design principles

1. **Small storage layout.** The library needs only `definitions/` and `models/` for definitions and models.
2. **User control.** The user supplies the complete factor computation.
3. **Performance.** The engine calls the user function directly.
4. **Flexibility.** Users can choose their ML framework and feature engineering method.

## Directory layout

```
factorlib/
├── definitions/     # Factor definitions (metadata)
├── models/          # ML models (.pkl, .joblib, etc.)
└── temp/           # Temporary cache
```

## User code conventions

### 1. Algorithm file layout

Each algorithm file must include these elements.

```python
# Algorithm metadata
ALGORITHM_INFO = {
    'name': 'Algorithm name',
    'description': 'Algorithm description',
    'category': 'Algorithm category',
    'version': '1.0.0',
    'author': 'Author'
}

def calculate_factors(data: pd.DataFrame) -> Dict[str, pd.Series]:
    """Mining phase: train the model and generate all factors."""
    pass

def calculate_single_factor(data: pd.DataFrame, factor_name: str) -> pd.Series:
    """Computation phase: calculate a single factor."""
    pass
```

### 2. Core functions

#### `calculate_factors(data)`
- **Purpose.** Train the model and generate all factors during mining.
- **Input** — OHLCV data
- **Output** — a mapping from factor names to values
- **Responsibilities**
  - Feature engineering
  - Model training
  - Factor generation
  - Model persistence

#### `calculate_single_factor(data, factor_name)`
- **Purpose.** Calculate one factor during the computation phase.
- **Inputs** — OHLCV data and the factor name
- **Output** — a series of factor values
- **Responsibilities**
  - Load a pretrained model
  - Feature engineering
  - Compute the factor

## How the system invokes an algorithm

### 1. Mining
```python
# factor_builder.py
def _execute_algorithm(self, algo_id, data, **kwargs):
    algorithm_module = self._load_algorithm_module(algo_id)
    return algorithm_module.calculate_factors(data, **kwargs)
```

### 2. Real-time computation
```python
# factor_engine.py
def compute_single_factor(self, factor_id, data, **kwargs):
    factor_def = self.storage.load_factor_definition(factor_id)
    algorithm_name = factor_def.computation_data['algorithm_name']
    algorithm_module = self._load_algorithm_module(algorithm_name)
    return algorithm_module.calculate_single_factor(data, factor_name)
```

## Factor definition format

```json
{
  "factor_id": "algorithm_factor_name",
  "name": "factor_name",
  "description": "Factor description",
  "category": "ml",
  "computation_type": "ml_model",
  "computation_data": {
    "algorithm_name": "algorithm_id",
    "model_file": "model_file.pkl",
    "performance_metrics": {}
  },
  "parameters": {},
  "dependencies": [],
  "output_type": "series",
  "metadata": {
    "checksum": "hash",
    "created_at": "timestamp"
  }
}
```

## Example algorithm

### ML momentum factor

```python
# user_algo/ml_momentum_factor.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
from pathlib import Path

ALGORITHM_INFO = {
    'name': 'ML momentum factor',
    'description': 'A machine learning algorithm for mining momentum factors',
    'category': 'ml_momentum',
    'version': '1.0.0'
}

def calculate_factors(data: pd.DataFrame) -> Dict[str, pd.Series]:
    """Mining phase: train the model and generate all factors."""
    # 1. Engineer features
    features = create_features(data)

    # 2. Train models for different lookback windows
    factors = {}
    for window in ['5d', '10d', '20d']:
        # Train the model
        model = train_model(features, data, window)

        # Generate factors
        factor_name = f'ml_momentum_{window}'
        factor_values = predict_factor(features, model)
        factors[factor_name] = pd.Series(factor_values, index=data.index)

    return factors

def calculate_single_factor(data: pd.DataFrame, factor_name: str) -> pd.Series:
    """Computation phase: calculate a single factor."""
    # Extract the window from the factor name
    window = factor_name.split('_')[-1]

    # Load a pretrained model
    model_path = Path("factorlib/minactors/models") / f"ml_momentum_{window}.pkl"
    model = joblib.load(model_path)

    # Engineer features and predict
    features = create_features(data)
    predictions = model.predict(features.fillna(0))

    return pd.Series(predictions, index=data.index)
```

## Benefits

1. **Small storage layout.** The library needs only `definitions/` and `models/` for definitions and models.
2. **User control.** The user supplies the complete factor computation.
3. **Performance.** The engine calls the user function directly.
4. **Flexibility.** Users can choose their ML framework and feature engineering method.
5. **Maintenance.** The computation logic is kept together.

## Migration guide

### Migrating from the earlier layout

1. **Remove the `functions/` directory.** The new layout no longer generates function files.
2. **Update the algorithm file.** Add `calculate_single_factor`.
3. **Update the factor definition.** Use the new `computation_data` format.
4. **Test the new architecture.** Check that each operation still works.

### Compatibility

- The existing API remains available.
- Existing factor definition formats are supported.
- Earlier factor computation methods remain compatible.

## Recommended practices

1. **Algorithm design**
   - Separate feature engineering from model training.
   - Use consistent feature engineering functions.
   - Save both the model and its preprocessor.

2. **Error handling**
   - Handle exceptions where they can occur.
   - Return informative error messages.
   - Handle inputs with too few observations.

3. **Performance**
   - Cache loaded models.
   - Use vectorized operations.
   - Avoid repeated calculations.

4. **Code organization**
   - Keep feature engineering in separate functions.
   - Manage parameters through configuration files.
   - Document behavior and add explanatory comments.

## Summary

In this V3 design, researchers own the full calculation logic and the system handles scheduling and storage. Direct invocation and a smaller file layout simplify the implementation and give researchers flexibility in structuring ML workflows.
