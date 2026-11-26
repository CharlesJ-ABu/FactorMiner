# FactorMiner API Documentation

[简体中文](api.md)

> Archived V3 documentation. For the current interface, see the [V4 product guide](../FactorMiner_Product_Report_EN.md).

## Overview

FactorMiner provides a comprehensive API for programmatic factor mining.

## Core API Class

### FactorAPI

The main API class provides the full factor mining workflow.

#### Initialization

```python
from factor_miner import FactorAPI

api = FactorAPI(config=None)
```

**Parameters:**
- `config`: Optional configuration dictionary

#### Methods

##### load_data()

Load market data.

```python
result = api.load_data(
    symbol='BTC_USDT',
    timeframe='1h',
    start_date='2023-01-01',
    end_date='2023-12-31'
)
```

**Parameters:**
- `symbol`: Trading pair name
- `timeframe`: Timeframe
- `start_date`: Start date (optional)
- `end_date`: End date (optional)

**Returns:**
```python
{
    'success': True,
    'data': DataFrame,
    'info': {
        'symbol': 'BTC_USDT',
        'timeframe': '1h',
        'shape': (1000, 5),
        'date_range': {
            'start': '2023-01-01',
            'end': '2023-12-31'
        }
    }
}
```

##### build_factors()

Build factors.

```python
result = api.build_factors(
    data=market_data,
    factor_types=['technical', 'statistical'],
    windows=[5, 10, 20, 50]
)
```

**Parameters:**
- `data`: Market-data DataFrame
- `factor_types`: List of factor types
- `**kwargs`: Additional arguments

**Returns:**
```python
{
    'success': True,
    'factors': DataFrame,
    'info': {
        'total_factors': 50,
        'factor_names': ['SMA_20', 'RSI_14', ...],
        'data_points': 1000
    }
}
```

##### evaluate_factors()

Evaluate factors.

```python
result = api.evaluate_factors(
    factors_df=factors_data,
    returns=returns_series,
    metrics=['ic', 'ir', 'win_rate', 'effectiveness_score']
)
```

**Parameters:**
- `factors_df`: Factor-data DataFrame
- `returns`: Returns Series
- `metrics`: List of evaluation metrics

**Returns:**
```python
{
    'success': True,
    'evaluation': DataFrame,
    'info': {
        'total_factors': 50,
        'metrics': ['ic', 'ir', 'win_rate', 'effectiveness_score']
    }
}
```

##### optimize_factors()

Optimize a factor combination.

```python
result = api.optimize_factors(
    factors_df=factors_data,
    returns=returns_series,
    method='greedy',
    max_factors=20
)
```

**Parameters:**
- `factors_df`: Factor-data DataFrame
- `returns`: Returns Series
- `method`: Optimization method ('greedy', 'genetic', 'lasso')
- `max_factors`: Maximum number of factors

**Returns:**
```python
{
    'success': True,
    'selected_factors': ['factor1', 'factor2', ...],
    'score': 0.85,
    'info': {
        'method': 'greedy',
        'max_factors': 20,
        'selected_count': 15
    }
}
```

##### create_ensemble()

Create an ensemble factor.

```python
result = api.create_ensemble(
    factors_df=factors_data,
    returns=returns_series,
    method='ic_weight'
)
```

**Parameters:**
- `factors_df`: Factor-data DataFrame
- `returns`: Returns Series
- `method`: Ensemble method ('equal_weight', 'ic_weight', 'ml_weight')

**Returns:**
```python
{
    'success': True,
    'ensemble_factor': Series,
    'score': 0.82,
    'info': {
        'method': 'ic_weight',
        'total_factors': 50
    }
}
```

##### run_complete_analysis()

Run the complete analysis.

```python
result = api.run_complete_analysis(
    symbol='BTC_USDT',
    timeframe='1h',
    factor_types=['technical', 'statistical'],
    start_date='2023-01-01',
    end_date='2023-12-31'
)
```

**Parameters:**
- `symbol`: Trading pair name
- `timeframe`: Timeframe
- `factor_types`: List of factor types
- `start_date`: Start date (optional)
- `end_date`: End date (optional)

**Returns:**
```python
{
    'success': True,
    'data_info': {...},
    'factors_info': {...},
    'evaluation': {...},
    'optimization': {...},
    'ensemble': {...},
    'report': 'Detailed report text'
}
```

##### get_factor_info()

Get factor information.

```python
result = api.get_factor_info()
```

**Returns:**
```python
{
    'success': True,
    'factor_info': {
        'technical': ['SMA', 'EMA', 'RSI', ...],
        'statistical': ['rolling_mean', 'volatility', ...],
        'advanced': ['interaction', 'ratio', ...],
        'ml': ['random_forest', 'gradient_boosting', ...]
    }
}
```

## Core Module APIs

### DataLoader

Data loader.

```python
from factor_miner.core import DataLoader

loader = DataLoader()

# Load data
data = loader.get_data(
    symbol='BTC_USDT',
    interval='1h',
    start_date='2023-01-01',
    end_date='2023-12-31',
    data_source='binance'
)
```

### FactorBuilder

Factor builder.

```python
from factor_miner.core import FactorBuilder

builder = FactorBuilder()

# Build all factors
factors_df = builder.build_all_factors(
    data=market_data,
    factor_types=['technical', 'statistical']
)

# Build specific factor types
technical_factors = builder.build_technical_factors(data)
statistical_factors = builder.build_statistical_factors(data)
```

### FactorEvaluator

Factor evaluator.

```python
from factor_miner.core import FactorEvaluator

evaluator = FactorEvaluator()

# Evaluate multiple factors
results = evaluator.evaluate_multiple_factors(
    factors_df=factors_data,
    returns=returns_series
)

# Evaluate one factor
result = evaluator.evaluate_single_factor(
    factor=factor_series,
    returns=returns_series
)
```

### FactorOptimizer

Factor optimizer.

```python
from factor_miner.core import FactorOptimizer

optimizer = FactorOptimizer()

# Set the data
optimizer.set_data(factors_df, returns)

# Optimize the factor combination
selected_factors, score = optimizer.optimize_factor_combination(
    factors_df,
    max_factors=20,
    method='greedy'
)

# Create an ensemble factor
ensemble_factor = optimizer.create_ensemble_factor(
    factors_df,
    method='ic_weight'
)
```

## Utility Functions

### Data Utilities

```python
from factor_miner.utils import *

# Save results
save_results(results, 'output.json', format='json')

# Load results
results = load_results('output.json', format='json')

# Calculate returns
returns = calculate_returns(data, method='pct_change')

# Align data
aligned_data = align_data(data_dict)

# Validate data
is_valid = validate_data(data, required_columns=['open', 'high', 'low', 'close', 'volume'])

# Get data information
info = get_data_info(data)

# Format a number
formatted = format_number(0.12345, decimals=4)

# Create a report
report = create_summary_report(results, "Analysis Report")
```

### Visualization Utilities

```python
from factor_miner.utils import *

# Plot the factor distribution
plot_factor_distribution(factor_series, title="Factor Distribution")

# Plot IC over time
plot_ic_timeseries(ic_series, title="IC Time Series")

# Plot the correlation heatmap
plot_correlation_heatmap(correlation_matrix, title="Correlation Heatmap")

# Compare factor performance
plot_factor_performance_comparison(factors_data, returns, title="Factor Performance Comparison")
```

## Error Handling

### Common Errors

1. **Data loading error**
```python
{
    'success': False,
    'error': 'Data loading failed or the data is empty'
}
```

2. **Factor construction error**
```python
{
    'success': False,
    'error': 'Factor construction failed: incorrect data format'
}
```

3. **Evaluation error**
```python
{
    'success': False,
    'error': 'Evaluation failed: data lengths do not match'
}
```

### Error Handling Example

```python
try:
    result = api.run_complete_analysis(symbol='BTC_USDT')
    if result['success']:
        print("Analysis succeeded")
        print(result['report'])
    else:
        print(f"Analysis failed: {result['error']}")
except Exception as e:
    print(f"An exception occurred: {e}")
```

## Performance Optimization

### Processing Large Datasets

```python
# Process data in chunks
chunk_size = 10000
for chunk in pd.read_csv('large_data.csv', chunksize=chunk_size):
    # Process the data chunk
    pass

# Use multiple processes
from multiprocessing import Pool

def process_factor(data_chunk):
    # Process the factor
    return result

with Pool(processes=4) as pool:
    results = pool.map(process_factor, data_chunks)
```

### Memory Optimization

```python
# Use appropriate data types
data = data.astype({
    'open': 'float32',
    'high': 'float32',
    'low': 'float32',
    'close': 'float32',
    'volume': 'int32'
})

# Release memory promptly
import gc
del large_dataframe
gc.collect()
```

## Code Examples

### Complete Workflow

```python
from factor_miner import FactorAPI
import pandas as pd

# Initialize the API
api = FactorAPI()

# 1. Load data
data_result = api.load_data('BTC_USDT', '1h')
if not data_result['success']:
    print(f"Data loading failed: {data_result['error']}")
    exit()

data = data_result['data']

# 2. Build factors
factors_result = api.build_factors(
    data,
    factor_types=['technical', 'statistical']
)
if not factors_result['success']:
    print(f"Factor construction failed: {factors_result['error']}")
    exit()

factors_df = factors_result['factors']

# 3. Calculate returns
returns = data['close'].pct_change().shift(-1).dropna()

# 4. Evaluate factors
evaluation_result = api.evaluate_factors(factors_df, returns)
if not evaluation_result['success']:
    print(f"Factor evaluation failed: {evaluation_result['error']}")
    exit()

# 5. Optimize the factor combination
optimization_result = api.optimize_factors(factors_df, returns)
if optimization_result['success']:
    print(f"Selected factors: {optimization_result['selected_factors']}")
    print(f"Optimization score: {optimization_result['score']}")

# 6. Create an ensemble factor
ensemble_result = api.create_ensemble(factors_df, returns)
if ensemble_result['success']:
    print(f"Ensemble score: {ensemble_result['score']}")

# 7. Save results
api.save_analysis_results({
    'evaluation': evaluation_result['evaluation'],
    'optimization': optimization_result,
    'ensemble': ensemble_result
}, 'analysis_results.json')

print("Analysis complete!")
```

### Batch Processing

```python
import pandas as pd
from factor_miner import FactorAPI

api = FactorAPI()

# Define parameters
symbols = ['BTC_USDT', 'ETH_USDT', 'BNB_USDT', 'SOL_USDT']
timeframes = ['1h', '4h', '1d']
factor_types = ['technical', 'statistical']

results = {}

for symbol in symbols:
    for timeframe in timeframes:
        print(f"Processing {symbol} {timeframe}")

        try:
            result = api.run_complete_analysis(
                symbol=symbol,
                timeframe=timeframe,
                factor_types=factor_types
            )

            if result['success']:
                results[f"{symbol}_{timeframe}"] = result
                print(f"✅ {symbol} {timeframe} complete")
            else:
                print(f"❌ {symbol} {timeframe} failed: {result['error']}")

        except Exception as e:
            print(f"❌ {symbol} {timeframe} exception: {e}")

# Save all results
api.save_analysis_results(results, 'batch_results.json')
print("Batch processing complete!")
```

## Data Management Module

### WebUI Data Management API

FactorMiner provides WebUI data management, including dynamic retrieval of exchange data through CCXT.

#### List Exchanges

```http
GET /api/data/exchanges
```

**Returns:**
```json
{
  "success": true,
  "data": [
    {
      "id": "binance",
      "name": "Binance",
      "type": "cryptocurrency",
      "description": "The world's largest cryptocurrency exchange"
    },
    {
      "id": "okx",
      "name": "OKX",
      "type": "cryptocurrency",
      "description": "A leading cryptocurrency trading platform"
    },
    {
      "id": "bybit",
      "name": "Bybit",
      "type": "cryptocurrency",
      "description": "A specialized cryptocurrency derivatives exchange"
    }
  ]
}
```

#### Get Trading Pairs

```http
GET /api/data/symbols/{exchange}
```

**Parameters:**
- `exchange`: Exchange ID (binance, okx, bybit)

**Returns:**
```json
{
  "success": true,
  "data": {
    "spot": [
      {
        "symbol": "BTC/USDT",
        "name": "Bitcoin/USDT",
        "type": "spot",
        "active": true
      }
    ],
    "futures": [
      {
        "symbol": "BTC/USDT",
        "name": "Bitcoin/USDT",
        "type": "futures",
        "active": true
      }
    ]
  },
  "note": "Using preset trading pairs (network connection failed)"
}
```

#### Get Timeframes

```http
GET /api/data/timeframes
```

**Returns:**
```json
{
  "success": true,
  "data": [
    {
      "value": "1m",
      "name": "1 minute",
      "description": "1-minute candlestick data"
    },
    {
      "value": "5m",
      "name": "5 minutes",
      "description": "5-minute candlestick data"
    }
  ]
}
```

#### Start a Data Download

```http
POST /api/data/download
```

**Request body:**
```json
{
  "exchange": "binance",
  "trade_type": "futures",
  "symbol": "BTC_USDT",
  "timeframe": "1h",
  "start_date": "2025-08-01",
  "end_date": "2025-08-08",
  "merge_with_existing": true,
  "merge_strategy": "overlap"
}
```

**Parameter details:**
- `merge_with_existing`: Whether to merge with existing data; defaults to true
- `merge_strategy`: Merge strategy: `append`, `overlap`, or `replace`

**Returns:**
```json
{
  "success": true,
  "data": {
    "id": "binance_BTC_USDT_1h_20250808_204322",
    "exchange": "binance",
    "symbol": "BTC_USDT",
    "timeframe": "1h",
    "trade_type": "futures",
    "start_date": "2025-08-01",
    "end_date": "2025-08-08",
    "merge_with_existing": true,
    "merge_strategy": "overlap",
    "merge_info": {
      "existing_file": "/path/to/existing/file.feather",
      "existing_start_date": "2022-06-01 00:00",
      "existing_end_date": "2025-07-31 06:00",
      "existing_data_points": 27751,
      "merge_strategy": "append"
    },
    "status": "started",
    "progress": 0,
    "message": "Starting download..."
  }
}
```

#### Merge Data Files

```http
POST /api/data/merge-data
```

**Request body:**
```json
{
  "exchange": "binance",
  "trade_type": "futures",
  "symbol": "BTC_USDT",
  "timeframe": "1h",
  "new_data_file": "/path/to/new_data.feather",
  "merge_strategy": "overlap"
}
```

**Merge strategies:**
- `append`: Append new data after existing data; suitable for nonoverlapping periods
- `overlap`: Handle overlapping periods by retaining the latest data; suitable for updating existing data
- `replace`: Replace existing data completely; suitable for downloading it again

**Returns:**
```json
{
  "success": true,
  "message": "Data merged successfully; strategy: overlap",
  "merged_file": "/path/to/merged/file.feather",
  "backup_file": "/path/to/backup/file.feather.backup",
  "original_data_points": 27751,
  "new_data_points": 168,
  "merged_data_points": 27919,
  "merged_date_range": {
    "start": "2022-06-01 00:00",
    "end": "2025-08-08 23:00"
  }
}
```

#### Get Download Suggestions

```http
POST /api/data/download-suggestions
```

**Request body:**
```json
{
  "exchange": "binance",
  "trade_type": "futures",
  "symbol": "ETH/USDT",
  "timeframe": "1h"
}
```

**Returns:**
```json
{
  "success": true,
  "data": {
    "exchange": "binance",
    "trade_type": "futures",
    "symbol": "ETH/USDT",
    "timeframe": "1h",
    "existing_data": [
      {
        "data_type": "futures",
        "start_date": "2024-01-01 00:00",
        "end_date": "2024-08-08 00:00",
        "data_points": 5000,
        "file_size": "2.5 MB"
      }
    ],
    "recommended_downloads": [
      {
        "data_type": "futures",
        "start_date": "2024-08-08 00:00",
        "end_date": "2025-08-08 00:00",
        "reason": "Fill futures data from 2024-08-08 00:00 to the latest available time"
      }
    ]
  }
}
```

### CCXT Integration

FactorMiner uses CCXT to retrieve real exchange data. CCXT is a unified cryptocurrency trading API library supporting over 100 exchanges.

#### Network Requirements

Network restrictions may require a VPN to access some exchange APIs:

1. **Binance**: A VPN may be needed in some regions
2. **OKX**: A VPN may be needed in some regions
3. **Bybit**: A VPN may be needed in some regions

#### Using a VPN to Retrieve Real Data

When connectivity is available, the system automatically retrieves the latest trading-pair list from the exchange:

```python
# Example: retrieve real Binance trading pairs
import ccxt

exchange = ccxt.binance({
    'enableRateLimit': True,
    'timeout': 10000,
})

markets = exchange.load_markets()
usdt_markets = [s for s in markets.keys() if s.endswith('/USDT')]
print(f"Number of Binance USDT trading pairs: {len(usdt_markets)}")
```

#### Fallback Mechanism

If the network connection fails, the system automatically uses a preset list of common trading pairs:

```json
{
  "note": "Using preset trading pairs (network connection failed)"
}
```

This keeps data-management features available even in environments with restricted connectivity.

#### Supported Exchanges

- **Binance**: The world's largest cryptocurrency exchange
- **OKX**: A leading cryptocurrency trading platform
- **Bybit**: A specialized cryptocurrency derivatives exchange

#### Trading Types

- **Spot**: Immediate buying and selling of cryptocurrencies
- **Futures**: Leveraged trading and derivatives

#### Data Format

All trading pairs use the standard `BASE/QUOTE` format, for example:
- `BTC/USDT`: Bitcoin against USDT
- `ETH/USDT`: Ethereum against USDT
- `BNB/USDT`: BNB against USDT

### Data Download Features

#### Start a Data Download

```http
POST /api/data/download
```

**Request body:**
```json
{
  "exchange": "binance",
  "trade_type": "futures",
  "symbol": "ETH/USDT",
  "timeframe": "1h",
  "start_date": "2024-01-01",
  "end_date": "2024-08-08"
}
```

#### Query Download Status

```http
GET /api/data/download-status/{download_id}
```

#### Delete Data

```http
POST /api/data/delete-data
```

**Request body:**
```json
{
  "file_path": "data/binance/futures/ETH_USDT_USDT-1h-futures.feather"
}
```

### Local Data Management

#### List Local Data

```http
GET /api/data/local-data
```

**Returns:**
```json
{
  "success": true,
  "data": [
    {
      "symbol": "BTC_USDT",
      "timeframe": "1h",
      "data_type": "futures",
      "start_date": "2024-01-01 00:00",
      "end_date": "2024-08-08 00:00",
      "data_points": 5000,
      "file_size": "2.5 MB"
    }
  ]
}
```

#### Data Coverage

```http
GET /api/data/data-coverage
```

**Returns:**
```json
{
  "success": true,
  "data": {
    "BTC_USDT": {
      "1h": {
        "futures": {
          "start_date": "2024-01-01",
          "end_date": "2024-08-08",
          "data_points": 5000
        }
      }
    }
  }
}
```

### Usage Recommendations

1. **Network access**: A VPN is recommended to ensure access to all exchange APIs
2. **Data selection**: Choose trading pairs and timeframes that suit the analysis
3. **Storage management**: Regularly remove unneeded data files to save space
4. **Data validation**: Check completeness and accuracy after downloading
