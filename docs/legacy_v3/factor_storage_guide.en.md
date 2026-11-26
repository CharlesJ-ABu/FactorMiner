# 📁 FactorMiner factor storage architecture

[简体中文](factor_storage_guide.md) · [English documentation](../README.en.md)

> Archived V3 documentation. For the current interface, see the [V4 product guide](../FactorMiner_Product_Report_EN.md).

## 🎯 Overview

V3 introduced a storage architecture built around `FactorEngine` and `TransparentFactorStorage`. This guide describes how that release organized and managed factors.

## 🏗️ V3 architecture

### Core components

- **`FactorEngine`** provides a common interface for computing factors.
- **`TransparentFactorStorage`** manages the underlying files.
- **`FactorDefinition`** describes a factor through metadata.
- **`FactorCalculator`** calculates factor values.

### Storage layout

```
factorlib/
├── definitions/          # Factor definitions (JSON)
├── evaluations/          # Factor evaluation results
├── temp/                 # Temporary files
└── mining_history/       # Mining history
```

## 📊 Factor types

### 1. Technical factors
- **Computation type** — `formula` or `function`
- **Location** — `factorlib/definitions/`
- **Examples** — SMA, RSI, MACD and Bollinger Bands

### 2. Statistical factors
- **Computation type** — `function`
- **Location** — `factorlib/definitions/`
- **Examples** — rolling statistics, distribution features and correlation factors

### 3. Machine learning factors
- **Computation type** — `ml_model` (loads a .pkl file) or `function`
- **Location** — `factorlib/definitions/` and a `.pkl` model file
- **Examples** — ensembles, PCA and feature selection

### 4. Advanced factors
- **Computation type** — `function`
- **Location** — `factorlib/definitions/`
- **Examples** — interactions, ratios and composite factors

## 📂 Factor definitions

### 1. JSON definition file

Each factor had a JSON definition under `factorlib/definitions/`.

```json
{
  "factor_id": "sma_20",
  "name": "20-period simple moving average",
  "description": "Simple moving average of the closing price over 20 periods",
  "category": "technical",
  "subcategory": "trend",
  "computation_type": "formula",
  "computation_data": {
    "formula": "close.rolling(window=20).mean()"
  },
  "parameters": {
    "window": 20
  },
  "dependencies": [],
  "output_type": "series",
  "metadata": {
    "created_at": "2024-01-01T00:00:00Z",
    "version": "1.0.0"
  }
}
```

### 2. Computation types

#### Formula
- **Use** — simple mathematical formulas
- **Example** — `close.rolling(window=20).mean()`
- **Advantage** — fast to compute and easy to read

#### Function
- **Use** — complex Python functions
- **Examples** — custom technical indicators and statistical calculations
- **Advantage** — accommodates more complex logic

#### ML model
- **Use** — machine learning models
- **Example** — a trained model stored in a .pkl file
- **Advantage** — supports complex ML algorithms

#### Pipeline
- **Use** — multi-step computations
- **Example** — feature engineering, model inference and post-processing
- **Advantage** — supports multi-stage ML workflows
- `ic_pearson` — Pearson information coefficient
- `ic_spearman` — Spearman information coefficient
- `ic_kendall` — Kendall information coefficient
- `mutual_information` — mutual information
- `long_short_return` — long–short return
- `sharpe_ratio` — Sharpe ratio
- `win_rate` — win rate
- `turnover` — turnover

**Example data**
```csv
factor_name,data_length,ic_pearson,ic_spearman,sharpe_ratio,win_rate
atr_14,27751,0.009149,0.012722,0.022176,0.507513
bb_width,27751,0.008706,0.012049,0.016136,0.507513
```

### 4. Deep Alpha factors (batch files)

**Location** — `factorlib/deep_alpha/values/`

**File layout**
```
deep_alpha_factors/
├── factor_records.json           # Factor records (8.3 KB)
├── factors_batch_001.csv         # Factor batch 1 (1.2 MB)
├── factors_batch_002.csv         # Factor batch 2 (948 KB)
├── factors_batch_003.csv         # Factor batch 3 (1.0 MB)
├── factors_batch_004.csv         # Factor batch 4 (883 KB)
├── factors_batch_005.csv         # Factor batch 5 (964 KB)
├── best_factors_analysis.csv     # Analysis of the best factors (7.1 KB)
├── best_factor_portfolio.csv     # Best factor portfolio (1.1 MB)
├── comprehensive_factor_report.txt # Comprehensive factor report (2.2 KB)
├── portfolio_report.txt          # Portfolio report (489 B)
├── mining_progress.json          # Mining progress (159 B)
├── factors/                      # Factor subdirectory
├── evaluations/                  # Evaluation results
├── formulas/                     # Formulas
└── principles/                   # Principles
```

**Example record** (`factor_records.json`)
```json
{
  "deep_alpha_0001": {
    "formula": "(high.rolling(10).max() - low.rolling(10).min()) / close.rolling(10).mean()",
    "symbol": "BNB",
    "generation_time": "2025-08-07T22:07:14.041165"
  },
  "deep_alpha_0002": {
    "formula": "Momentum interaction: the product of 10-period price momentum and volume momentum",
    "symbol": "BTC",
    "generation_time": "2025-08-07T22:07:14.042531"
  }
}
```

**Batch factor files**
- Each batch file contains about 6,939 rows.
- Each contains time series for several Deep Alpha factors.
- The format supports different trading pairs and timeframes.

## 🔍 Scanning the factor library

### Scan procedure

1. **Scan Alpha101 factors**
   ```python
   alpha101_dir = FACTOR_LIBRARY_DIR / "alpha101"
   for file in alpha101_dir.glob("*.pkl"):
       symbol, timeframe = parse_alpha101_filename(file.name)
       # Create a factor record
   ```

2. **Scan ML factors**
   ```python
   ml_factors_file = FACTOR_LIBRARY_DIR / "ml_factors.csv"
   df = pd.read_csv(ml_factors_file)
   for _, row in df.iterrows():
       # Create an ML factor record
   ```

3. **Scan traditional factors**
   ```python
   traditional_factors_file = FACTOR_LIBRARY_DIR / "best_traditional_factors.csv"
   df = pd.read_csv(traditional_factors_file)
   for _, row in df.iterrows():
       # Create a traditional factor record
   ```

4. **Scan Deep Alpha factors**
   ```python
   factor_records_file = deep_alpha_dir / "factor_records.json"
   with open(factor_records_file, 'r') as f:
       factor_records = json.load(f)
   for record in factor_records.get('factors', []):
       # Create a Deep Alpha factor record
   ```

### Factor ID conventions

- **Alpha101**: `alpha101_{symbol}_{timeframe}`
- **ML**: `ml_{factor_name}`
- **Traditional**: `traditional_{factor_name}`
- **Deep Alpha**: `deep_alpha_{id}`

## 📈 Factor data formats

### 1. Pickle (.pkl)
- **Use** — Alpha101 factor storage
- **Advantages** — preserves data types and supports compressed storage
- **Read with** `pd.read_pickle(file_path)`

### 2. CSV (.csv)
- **Use** — ML, traditional and Deep Alpha factors
- **Advantages** — human-readable and easy to process
- **Read with** `pd.read_csv(file_path)`

### 3. JSON (.json)
- **Use** — factor metadata, records and configuration
- **Advantages** — structured and easy to parse
- **Read with** `json.load(file)`

## 🛠️ Managing factors

### List factors
```bash
# Retrieve via the API
curl "http://localhost:8080/api/factors/list"

# Inspect the files directly
ls -la factorlib/alpha101/values/
head -5 factorlib/ml_factors/values/ml_factors.csv
```

### Evaluate factors
```bash
# Evaluate a specific factor
curl -X POST "http://localhost:8080/api/factors/evaluate" \
  -H "Content-Type: application/json" \
  -d '{
    "factor_id": "alpha101_BTC_USDT_1h",
    "symbol": "BTC/USDT",
    "timeframe": "1h",
    "start_date": "2024-01-01",
    "end_date": "2024-12-31"
  }'
```

### Export factors
```bash
# Export factor data
curl "http://localhost:8080/api/factors/export/alpha101_BTC_USDT_1h"
```

## 💡 Usage guidelines

### 1. Choosing a factor family
- **Alpha101** for basic quantitative strategies
- **ML factors** for recognizing complex patterns
- **Traditional factors** for technical analysis
- **Deep Alpha** for deep learning strategies

### 2. Managing data
- Back up factor data regularly.
- Monitor factor file sizes.
- Remove obsolete evaluation results.

### 3. Performance
- Choose a format appropriate to the data.
- Evaluate factors in batches.
- Cache frequently used factor data.

## 🔧 Technical details

### File sizes in the V3 example
```
results/
├── alpha101/                    # 14.4 MB (8 files)
├── ml_factors.csv              # 11MB
├── best_traditional_factors.csv # 6.4KB
├── deep_alpha_factors/         # ~5MB
└── Total: ~30 MB
```

### Memory use
- **Alpha101** — about 1.8 MB per file
- **ML factors** — about 11 MB (35,042 rows × 16 columns)
- **Traditional factors** — about 6.4 KB (10 factors)
- **Deep Alpha** — about 5 MB of batch data

### Access times reported for V3
- **API requests** — milliseconds
- **File reads** — seconds
- **Factor evaluation** — minutes

## 📞 Troubleshooting

If you encounter problems storing or accessing factors in V3, check the following.

1. **File permissions.** The Web UI needs read access.
2. **File integrity.** Check for corruption.
3. **Logs.** Inspect the error log.
4. **Service.** Restart the Web UI service.

## 🎯 Summary

The V3 factor library supported several factor types and storage formats.

- **Central management.** The example kept all factors under `results/`.
- **Multiple formats.** Pickle, CSV and JSON were supported.
- **Access.** Factors could be read through the API or the filesystem.
- **Extensions.** New factor types could be added.

The V3 collection described here contained 35,059 factors for quantitative strategy research.
