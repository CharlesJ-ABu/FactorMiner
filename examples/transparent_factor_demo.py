#!/usr/bin/env python3
"""
透明因子存储演示 v3.0
展示如何以完全透明的方式存储和管理因子
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# 添加项目路径
sys.path.append('/Users/charles/FactorMiner')

from factor_miner.core.factor_storage import TransparentFactorStorage


def create_sample_data(periods=200):
    """创建示例数据"""
    start_time = datetime.now() - timedelta(hours=periods)
    times = pd.date_range(start_time, periods=periods, freq='H')
    
    np.random.seed(42)
    base_price = 50000
    returns = np.random.normal(0, 0.02, periods)
    prices = [base_price]
    
    for i in range(1, periods):
        new_price = prices[-1] * (1 + returns[i])
        prices.append(new_price)
    
    data = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': np.random.uniform(100, 1000, periods)
    }, index=times)
    
    # 确保OHLC逻辑
    data['high'] = data[['open', 'close', 'high']].max(axis=1)
    data['low'] = data[['open', 'close', 'low']].min(axis=1)
    
    return data


def demo_formula_factors():
    """演示公式类因子的透明存储"""
    print("📊 公式类因子演示")
    print("=" * 50)
    
    storage = TransparentFactorStorage()
    
    # 1. 简单移动平均
    sma_formula = """
# 计算简单移动平均线
# 参数:
#   - period: 周期
close.rolling(window=period).mean()
"""
    success = storage.save_formula_factor(
        factor_id="sma_v3",
        name="透明SMA",
        formula=sma_formula.strip(),
        description="完全透明的简单移动平均线实现",
        category="trend",
        parameters={"period": 20}
    )
    print(f"保存SMA公式: {'✅' if success else '❌'}")
    
    # 2. RSI指标
    rsi_formula = """
# 计算RSI指标
# 参数:
#   - period: 周期

# 1. 计算价格变化
delta = close.diff()

# 2. 分离上涨和下跌
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)

# 3. 计算平均值
avg_gain = gain.rolling(window=period).mean()
avg_loss = loss.rolling(window=period).mean()

# 4. 计算相对强度
rs = avg_gain / avg_loss

# 5. 计算RSI
100 - (100 / (1 + rs))
"""
    success = storage.save_formula_factor(
        factor_id="rsi_v3",
        name="透明RSI",
        formula=rsi_formula.strip(),
        description="完全透明的RSI指标实现",
        category="momentum",
        parameters={"period": 14}
    )
    print(f"保存RSI公式: {'✅' if success else '❌'}")
    
    # 测试计算
    data = create_sample_data()
    print(f"\n📈 测试数据: {len(data)} 条记录")
    
    try:
        sma_result = storage.compute_factor("sma_v3", data)
        print(f"SMA计算结果: {sma_result.iloc[-1]:.2f}")
        
        rsi_result = storage.compute_factor("rsi_v3", data)
        print(f"RSI计算结果: {rsi_result.iloc[-1]:.2f}")
        
    except Exception as e:
        print(f"❌ 计算失败: {e}")


def demo_function_factors():
    """演示函数类因子的透明存储"""
    print("\n\n🔧 函数类因子演示")
    print("=" * 50)
    
    storage = TransparentFactorStorage()
    
    # 1. MACD函数
    macd_code = """
def calculate(data, fast_period=12, slow_period=26, signal_period=9):
    \"\"\"
    计算MACD指标
    
    参数:
        - fast_period: 快线周期
        - slow_period: 慢线周期
        - signal_period: 信号线周期
    \"\"\"
    # 1. 计算快线和慢线
    fast_ema = data['close'].ewm(span=fast_period).mean()
    slow_ema = data['close'].ewm(span=slow_period).mean()
    
    # 2. 计算MACD线
    macd_line = fast_ema - slow_ema
    
    # 3. 计算信号线
    signal_line = macd_line.ewm(span=signal_period).mean()
    
    # 4. 计算MACD柱状图
    histogram = macd_line - signal_line
    
    return macd_line  # 返回MACD线
"""
    
    success = storage.save_function_factor(
        factor_id="macd_v3",
        name="透明MACD",
        function_code=macd_code,
        entry_point="calculate",
        description="完全透明的MACD指标实现",
        category="momentum",
        parameters={
            "fast_period": 12,
            "slow_period": 26,
            "signal_period": 9
        }
    )
    print(f"保存MACD函数: {'✅' if success else '❌'}")
    
    # 2. 布林带函数
    bb_code = """
def calculate(data, period=20, std_dev=2):
    \"\"\"
    计算布林带指标
    
    参数:
        - period: 周期
        - std_dev: 标准差倍数
    \"\"\"
    # 1. 计算中轨(SMA)
    middle = data['close'].rolling(window=period).mean()
    
    # 2. 计算标准差
    std = data['close'].rolling(window=period).std()
    
    # 3. 计算上轨和下轨
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    
    # 4. 计算带宽
    bandwidth = (upper - lower) / middle
    
    return bandwidth  # 返回带宽
"""
    
    success = storage.save_function_factor(
        factor_id="bb_bandwidth_v3",
        name="透明布林带宽度",
        function_code=bb_code,
        entry_point="calculate",
        description="完全透明的布林带宽度实现",
        category="volatility",
        parameters={
            "period": 20,
            "std_dev": 2
        }
    )
    print(f"保存布林带函数: {'✅' if success else '❌'}")
    
    # 测试计算
    data = create_sample_data()
    
    try:
        macd_result = storage.compute_factor("macd_v3", data)
        print(f"MACD计算结果: {macd_result.iloc[-1]:.4f}")
        
        bb_result = storage.compute_factor("bb_bandwidth_v3", data)
        print(f"布林带宽度: {bb_result.iloc[-1]:.4f}")
        
    except Exception as e:
        print(f"❌ 计算失败: {e}")


def demo_ml_pipeline():
    """演示ML流水线因子的透明存储"""
    print("\n\n🤖 ML流水线演示")
    print("=" * 50)
    
    storage = TransparentFactorStorage()
    
    # 定义ML流水线
    pipeline_steps = [
        # 步骤1: 特征工程
        {
            "type": "feature_engineering",
            "code": """
# 创建特征DataFrame
features = pd.DataFrame(index=data.index)

# 1. 价格动量特征
features['price_momentum'] = data['close'].pct_change(5)
features['price_volatility'] = data['close'].rolling(10).std() / data['close']

# 2. 成交量特征
features['volume_momentum'] = data['volume'].pct_change(5)
features['volume_ma_ratio'] = data['volume'] / data['volume'].rolling(20).mean()

# 3. 趋势特征
sma_fast = data['close'].rolling(5).mean()
sma_slow = data['close'].rolling(20).mean()
features['trend_strength'] = (sma_fast - sma_slow) / sma_slow

# 4. 波动率特征
high_low_ratio = data['high'] / data['low']
features['volatility_ratio'] = high_low_ratio.rolling(10).mean()
""",
            "outputs": [
                "price_momentum", "price_volatility",
                "volume_momentum", "volume_ma_ratio",
                "trend_strength", "volatility_ratio"
            ]
        },
        
        # 步骤2: 模型
        {
            "type": "model",
            "algorithm": "LinearRegression",
            "parameters": {
                "fit_intercept": True
            },
            "features": [
                "price_momentum", "price_volatility",
                "volume_momentum", "volume_ma_ratio",
                "trend_strength", "volatility_ratio"
            ]
        },
        
        # 步骤3: 后处理
        {
            "type": "postprocess",
            "code": """
# 将预测值转换为交易信号
signals = predictions.copy()

# 1. 标准化信号
signals = signals / signals.abs().max()

# 2. 设置信号阈值
signals = signals.where(abs(signals) > 0.2, 0)

# 3. 离散化为三档
signals = np.where(signals > 0.5, 1,
                  np.where(signals < -0.5, -1, 0))
"""
        }
    ]
    
    success = storage.save_pipeline_factor(
        factor_id="ml_trend_v3",
        name="ML趋势预测器",
        pipeline_steps=pipeline_steps,
        description="完全透明的ML趋势预测流水线",
        category="ml",
        parameters={"lookback": 20}
    )
    print(f"保存ML流水线: {'✅' if success else '❌'}")
    
    # 测试计算
    data = create_sample_data(500)  # 更多数据用于ML
    
    try:
        ml_result = storage.compute_factor("ml_trend_v3", data)
        signals = ml_result[ml_result != 0]
        print(f"ML信号数量: {len(signals)}")
        print(f"信号分布: {pd.value_counts(ml_result)}")
        
    except Exception as e:
        print(f"❌ ML计算失败: {e}")


def show_factor_storage():
    """展示因子存储结构"""
    print("\n\n📁 因子存储结构展示")
    print("=" * 50)
    
    storage = TransparentFactorStorage()
    
    # 展示目录结构
    print("目录结构:")
    print(f"  📂 {storage.storage_dir}")
    print(f"  ├── 📂 definitions/  (因子定义)")
    print(f"  ├── 📂 formulas/     (公式文本)")
    print(f"  ├── 📂 functions/    (函数代码)")
    print(f"  ├── 📂 pipelines/    (ML流水线)")
    print(f"  └── 📂 temp/         (临时缓存)")
    
    # 统计文件
    def count_files(path):
        return len([f for f in path.glob("*") if f.is_file()])
    
    print("\n文件统计:")
    print(f"  📊 定义文件: {count_files(storage.definitions_dir)} 个")
    print(f"  📊 公式文件: {count_files(storage.formulas_dir)} 个")
    print(f"  📊 函数文件: {count_files(storage.functions_dir)} 个")
    print(f"  📊 流水线文件: {count_files(storage.pipelines_dir)} 个")
    
    # 展示示例文件内容
    print("\n📄 文件内容示例:")
    
    # 1. 公式文件
    formula_files = list(storage.formulas_dir.glob("*.txt"))
    if formula_files:
        print("\n公式文件示例:")
        with open(formula_files[0], 'r') as f:
            content = f.read()
            print(f"  {formula_files[0].name}:")
            print("  " + "\n  ".join(content.split("\n")[:5]))
    
    # 2. 函数文件
    function_files = list(storage.functions_dir.glob("*.py"))
    if function_files:
        print("\n函数文件示例:")
        with open(function_files[0], 'r') as f:
            content = f.read()
            print(f"  {function_files[0].name}:")
            print("  " + "\n  ".join(content.split("\n")[:5]))
    
    # 3. 流水线文件
    pipeline_files = list(storage.pipelines_dir.glob("*.json"))
    if pipeline_files:
        print("\n流水线文件示例:")
        with open(pipeline_files[0], 'r') as f:
            content = f.read()
            print(f"  {pipeline_files[0].name}:")
            print("  " + "\n  ".join(content.split("\n")[:5]))


def main():
    """主函数"""
    print("🚀 透明因子存储系统演示 v3.0")
    print("=" * 60)
    print("💡 核心特性:")
    print("  ✅ 完全透明的因子存储")
    print("  ✅ 可读的公式和代码文件")
    print("  ✅ ML流水线的完整记录")
    print("  ✅ 无序列化的二进制数据")
    
    try:
        # 演示各种类型的因子
        demo_formula_factors()
        demo_function_factors()
        demo_ml_pipeline()
        show_factor_storage()
        
        print("\n\n🎉 演示完成!")
        print("=" * 60)
        print("🎯 V3系统的优势:")
        print("  ✅ 因子逻辑完全透明")
        print("  ✅ 支持复杂的计算过程")
        print("  ✅ ML流水线完整记录")
        print("  ✅ 便于审计和维护")
        print("  ✅ 适合团队协作")
        
    except Exception as e:
        print(f"❌ 演示过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
