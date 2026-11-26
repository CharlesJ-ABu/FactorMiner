# User algorithm conventions

[简体中文](user_algorithm_paradigm.md) · [English documentation](../README.en.md)

> Archived V3 documentation. For the current interface, see the [V4 product guide](../FactorMiner_Product_Report_EN.md).

## Overview

V3 defined a common layout, required metadata and function signatures for all user algorithms. This made them easier to load and maintain.

## Specification

### 1. File structure

Each algorithm file must include the following sections.

```python
# 1. Module docstring
"""
Algorithm name and description
"""

# 2. Imports
import pandas as pd
import numpy as np
# ... additional imports

# 3. Algorithm metadata (required)
ALGORITHM_INFO = {
    # Required fields
    'id': 'unique_algorithm_id',           # Unique algorithm identifier
    'name': 'Algorithm display name',      # Human-readable name
    'description': 'Algorithm details',    # Description
    'category': 'algorithm_category',      # Algorithm category
    'version': '1.0.0',                    # Version number
    'author': 'Author name',               # Author

    # Optional fields
    'parameters': {                         # Algorithm parameter definitions
        'param1': {
            'type': 'int',
            'default': 10,
            'description': 'Parameter description',
            'min': 1,
            'max': 100
        }
    },
    'requirements': ['pandas', 'numpy'],   # Package dependencies
    'tags': ['ml', 'momentum'],            # Tags
    'created_at': '2024-01-01',            # Creation date
    'updated_at': '2024-01-01'             # Last update
}

# 4. Core functions (required)
def calculate_factors(data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
    """Train the model and generate all factors during mining."""
    pass

def calculate_single_factor(data: pd.DataFrame, factor_name: str, **kwargs) -> pd.Series:
    """Calculate a single factor during computation."""
    pass

# 5. Helper functions (optional)
def validate_data(data: pd.DataFrame) -> bool:
    """Validate the data."""
    pass

def get_factor_info(factor_name: str) -> Dict:
    """Return information about a factor."""
    pass
```

### 2. Required and optional fields

#### Required ALGORITHM_INFO fields

| Field | Type | Description | Example |
|------|------|------|------|
| `id` | str | Unique ID used by the system to identify this algorithm | `'ml_momentum_factor'` |
| `name` | str | Display name | `'ML momentum factor'` |
| `description` | str | Detailed description | `'An ML algorithm for mining momentum factors'` |
| `category` | str | Algorithm category | `'ml'`, `'technical'`, `'statistical'` |
| `version` | str | Version number following semantic versioning | `'1.0.0'` |
| `author` | str | Author name | `'FactorMiner Team'` |

#### Optional ALGORITHM_INFO fields

| Field | Type | Description | Example |
|------|------|------|------|
| `parameters` | dict | Algorithm parameter definitions | See the parameter definition format below |
| `requirements` | list | List of dependencies | `['pandas', 'numpy', 'sklearn']` |
| `tags` | list | List of tags | `['ml', 'momentum', 'regression']` |
| `created_at` | str | Creation date | `'2024-01-01'` |
| `updated_at` | str | Last update | `'2024-01-01'` |

### 3. Parameter definition format

```python
'parameters': {
    'param_name': {
        'type': 'int|float|str|bool|list',  # Parameter type
        'default': default_value,           # Default value
        'description': 'Parameter description',  # Purpose of the parameter
        'min': min_value,                  # Minimum for a numeric parameter
        'max': max_value,                  # Maximum for a numeric parameter
        'choices': [choice1, choice2],     # Available choices for a list parameter
        'required': True|False             # Whether the parameter is required
    }
}
```

### 4. Core function contracts

#### calculate_factors

**Purpose.** Train the model and generate all factors during mining.
**Called when** the user starts factor mining.
**Returns** `Dict[str, pd.Series]`, mapping factor names to values.

```python
def calculate_factors(data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
    """
    Mining: train the model and generate all factors.
    The system calls this function to mine factors.

    Args:
        data: OHLCV data containing 'open', 'high', 'low', 'close' and 'volume'.
        **kwargs: Algorithm parameters from ALGORITHM_INFO['parameters'].

    Returns:
        Dict[str, pd.Series]: Factor names mapped to their values.
        For example: {
            'factor_1': pd.Series(...),
            'factor_2': pd.Series(...)
        }
    """
    # 1. Validate the data
    if not validate_data(data):
        raise ValueError("Data validation failed")

    # 2. Engineer features
    features = create_features(data)

    # 3. Train the model if required
    model = train_model(features, data)

    # 4. Generate factors
    factors = {}
    factors['factor_1'] = generate_factor_1(features, model)
    factors['factor_2'] = generate_factor_2(features, model)

    return factors
```

#### calculate_single_factor

**Purpose.** Calculate one factor during the computation phase.
**Called when** the user requests a specific factor.
**Returns** a `pd.Series` of factor values.

```python
def calculate_single_factor(data: pd.DataFrame, factor_name: str, **kwargs) -> pd.Series:
    """
    Computation: calculate one factor.
    The system calls this function for real-time factor calculation.

    Args:
        data: OHLCV data containing 'open', 'high', 'low', 'close' and 'volume'.
        factor_name: A key returned by calculate_factors.
        **kwargs: Algorithm parameters from ALGORITHM_INFO['parameters'].

    Returns:
        pd.Series: Factor values with the same index as data.
    """
    # 1. Validate the data
    if not validate_data(data):
        raise ValueError("Data validation failed")

    # 2. Load the pretrained model if required
    model = load_model(factor_name)

    # 3. Engineer features
    features = create_features(data)

    # 4. Calculate the factor
    if factor_name == 'factor_1':
        return calculate_factor_1(features, model)
    elif factor_name == 'factor_2':
        return calculate_factor_2(features, model)
    else:
        raise ValueError(f"Unknown factor name: {factor_name}")
```

### 5. Helper function contracts

#### validate_data

```python
def validate_data(data: pd.DataFrame) -> bool:
    """
    Validate the data.

    Args:
        data: Input data.

    Returns:
        bool: Whether validation passed.
    """
    required_columns = ['open', 'high', 'low', 'close', 'volume']

    # Check required columns
    if not all(col in data.columns for col in required_columns):
        return False

    # Check data length
    if len(data) < 100:  # At least 100 observations are required
        return False

    # Check data quality
    if data[required_columns].isnull().all().any():
        return False

    return True
```

#### get_factor_info

```python
def get_factor_info(factor_name: str) -> Dict:
    """
    Return information about a factor.

    Args:
        factor_name: Name of the factor.

    Returns:
        Dict: Factor information.
    """
    factor_info = {
        'factor_1': {
            'name': 'Factor 1',
            'description': 'Description of factor 1',
            'type': 'ml',
            'parameters': {}
        },
        'factor_2': {
            'name': 'Factor 2',
            'description': 'Description of factor 2',
            'type': 'technical',
            'parameters': {}
        }
    }

    return factor_info.get(factor_name, {})
```

### 6. Algorithm categories

| Category | Description | Example |
|------|------|------|
| `ml` | Machine learning | Random forests, neural networks, SVM |
| `technical` | Technical indicators | Moving averages, RSI, MACD |
| `statistical` | Statistical algorithms | Correlation, cointegration, principal component analysis |
| `advanced` | Advanced methods | Deep learning, reinforcement learning |

### 7. Error handling

```python
def calculate_factors(data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
    try:
        # Algorithm logic
        pass
    except Exception as e:
        # Log the error
        print(f"❌ Algorithm failed: {e}")
        # Return an empty result on failure
        return {}
```

### 8. Logging

```python
def calculate_factors(data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
    print("🚀 Starting algorithm...")
    print(f"✅ Loaded {len(data)} records")
    print(f"🔄 Starting feature engineering...")
    print(f"✅ Number of features created: {features.shape[1]}")
    print(f"🔄 Training model...")
    print(f"✅ Model training complete")
    print(f"🔄 Generating factors...")
    print(f"✅ Generated {len(factors)} factors")
```

## Complete example

See `examples/user_algorithm_template.py` for the full V3 algorithm template.

## Validation tool

Use `scripts/validate_algorithm.py` to check whether a V3 algorithm follows these conventions.
