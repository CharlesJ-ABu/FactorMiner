# FactorMiner API 文档

## 概述

FactorMiner 提供完整的API接口，支持程序化调用因子挖掘功能。

## 核心API类

### FactorAPI

主要的API接口类，提供因子挖掘的完整功能。

#### 初始化

```python
from factor_miner import FactorAPI

api = FactorAPI(config=None)
```

**参数：**
- `config`: 配置字典，可选

#### 方法

##### load_data()

加载市场数据。

```python
result = api.load_data(
    symbol='BTC_USDT',
    timeframe='1h',
    start_date='2023-01-01',
    end_date='2023-12-31'
)
```

**参数：**
- `symbol`: 交易对名称
- `timeframe`: 时间框架
- `start_date`: 开始日期（可选）
- `end_date`: 结束日期（可选）

**返回：**
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

构建因子。

```python
result = api.build_factors(
    data=market_data,
    factor_types=['technical', 'statistical'],
    windows=[5, 10, 20, 50]
)
```

**参数：**
- `data`: 市场数据DataFrame
- `factor_types`: 因子类型列表
- `**kwargs`: 其他参数

**返回：**
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

评估因子。

```python
result = api.evaluate_factors(
    factors_df=factors_data,
    returns=returns_series,
    metrics=['ic', 'ir', 'win_rate', 'effectiveness_score']
)
```

**参数：**
- `factors_df`: 因子数据DataFrame
- `returns`: 收益率Series
- `metrics`: 评估指标列表

**返回：**
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

优化因子组合。

```python
result = api.optimize_factors(
    factors_df=factors_data,
    returns=returns_series,
    method='greedy',
    max_factors=20
)
```

**参数：**
- `factors_df`: 因子数据DataFrame
- `returns`: 收益率Series
- `method`: 优化方法（'greedy', 'genetic', 'lasso'）
- `max_factors`: 最大因子数量

**返回：**
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

创建集成因子。

```python
result = api.create_ensemble(
    factors_df=factors_data,
    returns=returns_series,
    method='ic_weight'
)
```

**参数：**
- `factors_df`: 因子数据DataFrame
- `returns`: 收益率Series
- `method`: 集成方法（'equal_weight', 'ic_weight', 'ml_weight'）

**返回：**
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

运行完整分析。

```python
result = api.run_complete_analysis(
    symbol='BTC_USDT',
    timeframe='1h',
    factor_types=['technical', 'statistical'],
    start_date='2023-01-01',
    end_date='2023-12-31'
)
```

**参数：**
- `symbol`: 交易对名称
- `timeframe`: 时间框架
- `factor_types`: 因子类型列表
- `start_date`: 开始日期（可选）
- `end_date`: 结束日期（可选）

**返回：**
```python
{
    'success': True,
    'data_info': {...},
    'factors_info': {...},
    'evaluation': {...},
    'optimization': {...},
    'ensemble': {...},
    'report': '详细报告文本'
}
```

##### get_factor_info()

获取因子信息。

```python
result = api.get_factor_info()
```

**返回：**
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

## 核心模块API

### DataLoader

数据加载器。

```python
from factor_miner.core import DataLoader

loader = DataLoader()

# 加载数据
data = loader.get_data(
    symbol='BTC_USDT',
    interval='1h',
    start_date='2023-01-01',
    end_date='2023-12-31',
    data_source='binance'
)
```

### FactorBuilder

因子构建器。

```python
from factor_miner.core import FactorBuilder

builder = FactorBuilder()

# 构建所有因子
factors_df = builder.build_all_factors(
    data=market_data,
    factor_types=['technical', 'statistical']
)

# 构建特定类型因子
technical_factors = builder.build_technical_factors(data)
statistical_factors = builder.build_statistical_factors(data)
```

### FactorEvaluator

因子评估器。

```python
from factor_miner.core import FactorEvaluator

evaluator = FactorEvaluator()

# 评估多个因子
results = evaluator.evaluate_multiple_factors(
    factors_df=factors_data,
    returns=returns_series
)

# 评估单个因子
result = evaluator.evaluate_single_factor(
    factor=factor_series,
    returns=returns_series
)
```

### FactorOptimizer

因子优化器。

```python
from factor_miner.core import FactorOptimizer

optimizer = FactorOptimizer()

# 设置数据
optimizer.set_data(factors_df, returns)

# 优化因子组合
selected_factors, score = optimizer.optimize_factor_combination(
    factors_df,
    max_factors=20,
    method='greedy'
)

# 创建集成因子
ensemble_factor = optimizer.create_ensemble_factor(
    factors_df,
    method='ic_weight'
)
```

## 工具函数

### 数据工具

```python
from factor_miner.utils import *

# 保存结果
save_results(results, 'output.json', format='json')

# 加载结果
results = load_results('output.json', format='json')

# 计算收益率
returns = calculate_returns(data, method='pct_change')

# 对齐数据
aligned_data = align_data(data_dict)

# 验证数据
is_valid = validate_data(data, required_columns=['open', 'high', 'low', 'close', 'volume'])

# 获取数据信息
info = get_data_info(data)

# 格式化数字
formatted = format_number(0.12345, decimals=4)

# 创建报告
report = create_summary_report(results, "分析报告")
```

### 可视化工具

```python
from factor_miner.utils import *

# 绘制因子分布
plot_factor_distribution(factor_series, title="因子分布")

# 绘制IC时序图
plot_ic_timeseries(ic_series, title="IC时序图")

# 绘制相关性热图
plot_correlation_heatmap(correlation_matrix, title="相关性热图")

# 绘制因子表现对比
plot_factor_performance_comparison(factors_data, returns, title="因子表现对比")
```

## 错误处理

### 常见错误

1. **数据加载错误**
```python
{
    'success': False,
    'error': '数据加载失败或数据为空'
}
```

2. **因子构建错误**
```python
{
    'success': False,
    'error': '因子构建失败：数据格式不正确'
}
```

3. **评估错误**
```python
{
    'success': False,
    'error': '评估失败：数据长度不匹配'
}
```

### 错误处理示例

```python
try:
    result = api.run_complete_analysis(symbol='BTC_USDT')
    if result['success']:
        print("分析成功")
        print(result['report'])
    else:
        print(f"分析失败：{result['error']}")
except Exception as e:
    print(f"发生异常：{e}")
```

## 性能优化

### 大规模数据处理

```python
# 使用分块处理
chunk_size = 10000
for chunk in pd.read_csv('large_data.csv', chunksize=chunk_size):
    # 处理数据块
    pass

# 使用多进程
from multiprocessing import Pool

def process_factor(data_chunk):
    # 处理因子
    return result

with Pool(processes=4) as pool:
    results = pool.map(process_factor, data_chunks)
```

### 内存优化

```python
# 使用适当的数据类型
data = data.astype({
    'open': 'float32',
    'high': 'float32',
    'low': 'float32',
    'close': 'float32',
    'volume': 'int32'
})

# 及时释放内存
import gc
del large_dataframe
gc.collect()
```

## 示例代码

### 完整工作流程

```python
from factor_miner import FactorAPI
import pandas as pd

# 初始化API
api = FactorAPI()

# 1. 加载数据
data_result = api.load_data('BTC_USDT', '1h')
if not data_result['success']:
    print(f"数据加载失败：{data_result['error']}")
    exit()

data = data_result['data']

# 2. 构建因子
factors_result = api.build_factors(
    data, 
    factor_types=['technical', 'statistical']
)
if not factors_result['success']:
    print(f"因子构建失败：{factors_result['error']}")
    exit()

factors_df = factors_result['factors']

# 3. 计算收益率
returns = data['close'].pct_change().shift(-1).dropna()

# 4. 评估因子
evaluation_result = api.evaluate_factors(factors_df, returns)
if not evaluation_result['success']:
    print(f"因子评估失败：{evaluation_result['error']}")
    exit()

# 5. 优化因子组合
optimization_result = api.optimize_factors(factors_df, returns)
if optimization_result['success']:
    print(f"选择因子：{optimization_result['selected_factors']}")
    print(f"优化得分：{optimization_result['score']}")

# 6. 创建集成因子
ensemble_result = api.create_ensemble(factors_df, returns)
if ensemble_result['success']:
    print(f"集成得分：{ensemble_result['score']}")

# 7. 保存结果
api.save_analysis_results({
    'evaluation': evaluation_result['evaluation'],
    'optimization': optimization_result,
    'ensemble': ensemble_result
}, 'analysis_results.json')

print("分析完成！")
```

### 批量处理

```python
import pandas as pd
from factor_miner import FactorAPI

api = FactorAPI()

# 定义参数
symbols = ['BTC_USDT', 'ETH_USDT', 'BNB_USDT', 'SOL_USDT']
timeframes = ['1h', '4h', '1d']
factor_types = ['technical', 'statistical']

results = {}

for symbol in symbols:
    for timeframe in timeframes:
        print(f"处理 {symbol} {timeframe}")
        
        try:
            result = api.run_complete_analysis(
                symbol=symbol,
                timeframe=timeframe,
                factor_types=factor_types
            )
            
            if result['success']:
                results[f"{symbol}_{timeframe}"] = result
                print(f"✅ {symbol} {timeframe} 完成")
            else:
                print(f"❌ {symbol} {timeframe} 失败：{result['error']}")
                
        except Exception as e:
            print(f"❌ {symbol} {timeframe} 异常：{e}")

# 保存所有结果
api.save_analysis_results(results, 'batch_results.json')
print("批量处理完成！")
```

## 数据管理模块

### WebUI数据管理API

FactorMiner提供了完整的WebUI数据管理功能，支持通过CCXT库动态获取交易所数据。

#### 交易所列表

```http
GET /api/data/exchanges
```

**返回：**
```json
{
  "success": true,
  "data": [
    {
      "id": "binance",
      "name": "Binance",
      "type": "cryptocurrency",
      "description": "全球最大的加密货币交易所"
    },
    {
      "id": "okx",
      "name": "OKX",
      "type": "cryptocurrency", 
      "description": "领先的加密货币交易平台"
    },
    {
      "id": "bybit",
      "name": "Bybit",
      "type": "cryptocurrency",
      "description": "专业的加密货币衍生品交易所"
    }
  ]
}
```

#### 获取交易对列表

```http
GET /api/data/symbols/{exchange}
```

**参数：**
- `exchange`: 交易所ID (binance, okx, bybit)

**返回：**
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
  "note": "使用预设交易对（网络连接失败）"
}
```

#### 获取时间框架

```http
GET /api/data/timeframes
```

**返回：**
```json
{
  "success": true,
  "data": [
    {
      "value": "1m",
      "name": "1分钟",
      "description": "1分钟K线数据"
    },
    {
      "value": "5m", 
      "name": "5分钟",
      "description": "5分钟K线数据"
    }
  ]
}
```

#### 开始下载数据

```http
POST /api/data/download
```

**请求体：**
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

**参数说明：**
- `merge_with_existing`: 是否与现有数据合并，默认true
- `merge_strategy`: 合并策略，可选值：`append`、`overlap`、`replace`

**返回：**
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
    "message": "开始下载..."
  }
}
```

#### 合并数据文件

```http
POST /api/data/merge-data
```

**请求体：**
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

**合并策略说明：**
- `append`: 追加模式 - 将新数据追加到现有数据后面，适用于无重叠时间段
- `overlap`: 重叠模式 - 处理重叠时间段，保留最新数据，适用于更新现有数据
- `replace`: 替换模式 - 完全替换现有数据，适用于重新下载

**返回：**
```json
{
  "success": true,
  "message": "数据合并成功，策略: overlap",
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

#### 获取下载建议

```http
POST /api/data/download-suggestions
```

**请求体：**
```json
{
  "exchange": "binance",
  "trade_type": "futures",
  "symbol": "ETH/USDT",
  "timeframe": "1h"
}
```

**返回：**
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
        "reason": "补充futures数据从 2024-08-08 00:00 到最新"
      }
    ]
  }
}
```

### CCXT集成

FactorMiner使用CCXT库来获取真实的交易所数据。CCXT是一个统一的加密货币交易API库，支持100多个交易所。

#### 网络连接要求

由于网络限制，某些交易所的API可能需要VPN才能正常访问：

1. **Binance**: 在某些地区需要VPN访问
2. **OKX**: 在某些地区需要VPN访问  
3. **Bybit**: 在某些地区需要VPN访问

#### 使用VPN获取真实数据

当网络连接正常时，系统会自动从交易所获取最新的交易对列表：

```python
# 示例：获取Binance的真实交易对
import ccxt

exchange = ccxt.binance({
    'enableRateLimit': True,
    'timeout': 10000,
})

markets = exchange.load_markets()
usdt_markets = [s for s in markets.keys() if s.endswith('/USDT')]
print(f"Binance USDT交易对数量: {len(usdt_markets)}")
```

#### 回退机制

当网络连接失败时，系统会自动使用预设的常用交易对列表：

```json
{
  "note": "使用预设交易对（网络连接失败）"
}
```

这确保了即使在网络受限的环境中，用户仍然可以使用数据管理功能。

#### 支持的交易所

- **Binance**: 全球最大的加密货币交易所
- **OKX**: 领先的加密货币交易平台
- **Bybit**: 专业的加密货币衍生品交易所

#### 交易类型

- **现货 (Spot)**: 即时买卖加密货币
- **期货 (Futures)**: 杠杆交易和衍生品

#### 数据格式

所有交易对都使用标准格式：`BASE/QUOTE`，例如：
- `BTC/USDT`: 比特币兑USDT
- `ETH/USDT`: 以太坊兑USDT
- `BNB/USDT`: BNB兑USDT

### 数据下载功能

#### 启动数据下载

```http
POST /api/data/download
```

**请求体：**
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

#### 查询下载状态

```http
GET /api/data/download-status/{download_id}
```

#### 删除数据

```http
POST /api/data/delete-data
```

**请求体：**
```json
{
  "file_path": "data/binance/futures/ETH_USDT_USDT-1h-futures.feather"
}
```

### 本地数据管理

#### 获取本地数据列表

```http
GET /api/data/local-data
```

**返回：**
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

#### 数据覆盖情况

```http
GET /api/data/data-coverage
```

**返回：**
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

### 使用建议

1. **网络环境**: 建议使用VPN以确保能够访问所有交易所API
2. **数据选择**: 根据分析需求选择合适的交易对和时间框架
3. **存储管理**: 定期清理不需要的数据文件以节省存储空间
4. **数据验证**: 下载完成后验证数据完整性和准确性 