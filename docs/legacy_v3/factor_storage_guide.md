# 📁 FactorMiner 因子存储架构详解

[English](factor_storage_guide.en.md)

## 🎯 概述

FactorMiner 现在使用全新的V3因子存储架构，基于 `FactorEngine` 和 `TransparentFactorStorage` 系统。本文档详细说明了当前的因子存储结构和管理方式。

## 🏗️ 当前架构

### 核心组件

- **`FactorEngine`**: 统一的因子计算接口
- **`TransparentFactorStorage`**: 透明的因子存储管理
- **`FactorDefinition`**: 因子定义元数据
- **`FactorCalculator`**: 因子计算器

### 存储结构

```
factorlib/
├── definitions/          # 因子定义 (JSON格式)
├── evaluations/          # 因子评估结果
├── temp/                 # 临时文件
└── mining_history/       # 挖掘历史记录
```

## 📊 因子类型

### 1. 技术因子 (Technical Factors)
- **计算类型**: `formula` 或 `function`
- **存储位置**: `factorlib/definitions/`
- **示例**: SMA、RSI、MACD、布林带等

### 2. 统计因子 (Statistical Factors)
- **计算类型**: `function`
- **存储位置**: `factorlib/definitions/`
- **示例**: 滚动统计、分布特征、相关性因子等

### 3. 机器学习因子 (ML Factors)
- **计算类型**: `ml_model` (加载.pkl文件) 或 `function`
- **存储位置**: `factorlib/definitions/` + `.pkl` 模型文件
- **示例**: 集成学习、PCA、特征选择等

### 4. 高级因子 (Advanced Factors)
- **计算类型**: `function`
- **存储位置**: `factorlib/definitions/`
- **示例**: 交互因子、比率因子、复合因子等

## 📂 因子定义结构

### 1. 因子定义文件 (JSON格式)

每个因子都有一个JSON定义文件，存储在 `factorlib/definitions/` 目录中：

```json
{
  "factor_id": "sma_20",
  "name": "20期简单移动平均",
  "description": "计算20期收盘价的简单移动平均",
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

### 2. 计算类型说明

#### Formula类型
- **用途**: 简单的数学公式
- **示例**: `close.rolling(window=20).mean()`
- **优势**: 计算快速，易于理解

#### Function类型
- **用途**: 复杂的Python函数
- **示例**: 自定义技术指标、统计计算
- **优势**: 灵活性高，支持复杂逻辑

#### ML Model类型
- **用途**: 机器学习模型
- **示例**: 训练好的.pkl模型文件
- **优势**: 支持复杂的ML算法

#### Pipeline类型
- **用途**: 多步骤计算流程
- **示例**: 特征工程 + 模型 + 后处理
- **优势**: 支持复杂的ML流水线
- `ic_pearson`: 皮尔逊信息系数
- `ic_spearman`: 斯皮尔曼信息系数
- `ic_kendall`: 肯德尔信息系数
- `mutual_information`: 互信息
- `long_short_return`: 多空收益
- `sharpe_ratio`: 夏普比率
- `win_rate`: 胜率
- `turnover`: 换手率

**数据示例**:
```csv
factor_name,data_length,ic_pearson,ic_spearman,sharpe_ratio,win_rate
atr_14,27751,0.009149,0.012722,0.022176,0.507513
bb_width,27751,0.008706,0.012049,0.016136,0.507513
```

### 4. Deep Alpha因子 (批量)

**存储位置**: `factorlib/deep_alpha/values/`

**文件结构**:
```
deep_alpha_factors/
├── factor_records.json           # 因子记录 (8.3KB)
├── factors_batch_001.csv         # 第1批因子 (1.2MB)
├── factors_batch_002.csv         # 第2批因子 (948KB)
├── factors_batch_003.csv         # 第3批因子 (1.0MB)
├── factors_batch_004.csv         # 第4批因子 (883KB)
├── factors_batch_005.csv         # 第5批因子 (964KB)
├── best_factors_analysis.csv     # 最佳因子分析 (7.1KB)
├── best_factor_portfolio.csv     # 最佳因子组合 (1.1MB)
├── comprehensive_factor_report.txt # 综合因子报告 (2.2KB)
├── portfolio_report.txt          # 组合报告 (489B)
├── mining_progress.json          # 挖掘进度 (159B)
├── factors/                      # 因子子目录
├── evaluations/                  # 评估结果目录
├── formulas/                     # 公式目录
└── principles/                   # 原则目录
```

**因子记录示例** (`factor_records.json`):
```json
{
  "deep_alpha_0001": {
    "formula": "(high.rolling(10).max() - low.rolling(10).min()) / close.rolling(10).mean()",
    "symbol": "BNB",
    "generation_time": "2025-08-07T22:07:14.041165"
  },
  "deep_alpha_0002": {
    "formula": "动量交互因子，10期价格动量与成交量动量的乘积",
    "symbol": "BTC",
    "generation_time": "2025-08-07T22:07:14.042531"
  }
}
```

**批量因子文件**:
- 每个batch文件包含约6,939行数据
- 包含多个Deep Alpha因子的时间序列数据
- 支持不同交易对和时间框架

## 🔍 因子库扫描逻辑

### 扫描流程

1. **Alpha101因子扫描**:
   ```python
   alpha101_dir = FACTOR_LIBRARY_DIR / "alpha101"
   for file in alpha101_dir.glob("*.pkl"):
       symbol, timeframe = parse_alpha101_filename(file.name)
       # 创建因子记录
   ```

2. **ML因子扫描**:
   ```python
   ml_factors_file = FACTOR_LIBRARY_DIR / "ml_factors.csv"
   df = pd.read_csv(ml_factors_file)
   for _, row in df.iterrows():
       # 创建ML因子记录
   ```

3. **传统因子扫描**:
   ```python
   traditional_factors_file = FACTOR_LIBRARY_DIR / "best_traditional_factors.csv"
   df = pd.read_csv(traditional_factors_file)
   for _, row in df.iterrows():
       # 创建传统因子记录
   ```

4. **Deep Alpha因子扫描**:
   ```python
   factor_records_file = deep_alpha_dir / "factor_records.json"
   with open(factor_records_file, 'r') as f:
       factor_records = json.load(f)
   for record in factor_records.get('factors', []):
       # 创建Deep Alpha因子记录
   ```

### 因子ID生成规则

- **Alpha101**: `alpha101_{symbol}_{timeframe}`
- **ML**: `ml_{factor_name}`
- **Traditional**: `traditional_{factor_name}`
- **Deep Alpha**: `deep_alpha_{id}`

## 📈 因子数据格式

### 1. Pickle格式 (.pkl)
- **用途**: Alpha101因子存储
- **优势**: 保持数据类型，压缩存储
- **读取**: `pd.read_pickle(file_path)`

### 2. CSV格式 (.csv)
- **用途**: ML因子、传统因子、Deep Alpha因子
- **优势**: 可读性好，易于处理
- **读取**: `pd.read_csv(file_path)`

### 3. JSON格式 (.json)
- **用途**: 因子元数据、记录、配置
- **优势**: 结构化数据，易于解析
- **读取**: `json.load(file)`

## 🛠️ 因子管理操作

### 查看因子列表
```bash
# 通过API获取
curl "http://localhost:8080/api/factors/list"

# 直接查看文件
ls -la factorlib/alpha101/values/
head -5 factorlib/ml_factors/values/ml_factors.csv
```

### 因子评估
```bash
# 评估特定因子
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

### 导出因子
```bash
# 导出因子数据
curl "http://localhost:8080/api/factors/export/alpha101_BTC_USDT_1h"
```

## 💡 使用建议

### 1. 因子选择
- **Alpha101**: 适合基础量化策略
- **ML因子**: 适合复杂模式识别
- **传统因子**: 适合技术分析策略
- **Deep Alpha**: 适合深度学习策略

### 2. 数据管理
- 定期备份因子数据
- 监控因子文件大小
- 清理过期的评估结果

### 3. 性能优化
- 使用适当的数据格式
- 批量处理因子评估
- 缓存常用因子数据

## 🔧 技术细节

### 文件大小统计
```
results/
├── alpha101/                    # 14.4MB (8个文件)
├── ml_factors.csv              # 11MB
├── best_traditional_factors.csv # 6.4KB
├── deep_alpha_factors/         # ~5MB
└── 总计: ~30MB
```

### 内存使用
- **Alpha101**: 每个文件约1.8MB
- **ML因子**: 约11MB (35,042行 × 16列)
- **传统因子**: 约6.4KB (10个因子)
- **Deep Alpha**: 约5MB (批量数据)

### 访问性能
- **API访问**: 毫秒级响应
- **文件读取**: 秒级加载
- **因子评估**: 分钟级计算

## 📞 技术支持

如果在因子存储或访问方面遇到问题：

1. **检查文件权限**: 确保WebUI有读取权限
2. **验证文件完整性**: 检查文件是否损坏
3. **查看日志**: 检查错误日志
4. **重启服务**: 重启WebUI服务

## 🎯 总结

FactorMiner 的因子存储系统设计合理，支持多种因子类型和格式：

- **集中管理**: 所有因子统一存储在 `results/` 目录
- **格式多样**: 支持Pickle、CSV、JSON等多种格式
- **易于访问**: 提供API接口和文件系统访问
- **扩展性强**: 支持新因子类型的添加

通过因子库，你可以高效地管理和访问所有35,059个因子，为量化策略构建提供强大的数据支持！🎊 