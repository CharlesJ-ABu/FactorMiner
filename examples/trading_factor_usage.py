#!/usr/bin/env python3
"""
实际交易中的因子使用示例
演示如何在实时交易系统中调用因子
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from factor_miner.core.factor_engine import factor_engine
from factor_miner.core.factor_trading_api import trading_api
from factor_miner.core.factor_registry import register_factor


def create_custom_factors():
    """创建一些自定义的交易因子"""
    
    @register_factor(
        factor_id='trend_strength',
        name='趋势强度因子',
        description='结合价格趋势和成交量的趋势强度指标',
        category='custom',
        subcategory='trend',
        parameters={'short_period': 5, 'long_period': 20, 'volume_period': 10}
    )
    def calculate_trend_strength(data, short_period=5, long_period=20, volume_period=10):
        """计算趋势强度因子"""
        # 价格趋势
        price_ma_short = data['close'].rolling(short_period).mean()
        price_ma_long = data['close'].rolling(long_period).mean()
        price_trend = (price_ma_short / price_ma_long - 1) * 100
        
        # 成交量趋势
        volume_ma = data['volume'].rolling(volume_period).mean()
        volume_strength = data['volume'] / volume_ma
        
        # 趋势强度 = 价格趋势 * 成交量强度
        trend_strength = price_trend * np.log(volume_strength)
        
        return trend_strength
    
    @register_factor(
        factor_id='volatility_adjusted_momentum',
        name='波动率调整动量',
        description='根据波动率调整的动量因子',
        category='custom',
        subcategory='momentum',
        parameters={'momentum_period': 10, 'volatility_period': 20}
    )
    def calculate_volatility_adjusted_momentum(data, momentum_period=10, volatility_period=20):
        """计算波动率调整的动量因子"""
        # 价格动量
        momentum = data['close'] / data['close'].shift(momentum_period) - 1
        
        # 价格波动率
        returns = data['close'].pct_change()
        volatility = returns.rolling(volatility_period).std()
        
        # 波动率调整动量 = 动量 / 波动率
        adj_momentum = momentum / volatility
        
        return adj_momentum
    
    @register_factor(
        factor_id='support_resistance_strength',
        name='支撑阻力强度',
        description='基于历史价格的支撑阻力强度',
        category='custom',
        subcategory='pattern',
        parameters={'window': 50}
    )
    def calculate_support_resistance_strength(data, window=50):
        """计算支撑阻力强度"""
        def calculate_strength(prices):
            if len(prices) < window:
                return 0
            
            current_price = prices.iloc[-1]
            historical_prices = prices.iloc[-window:-1]
            
            # 计算当前价格与历史价格的接近程度
            price_distances = np.abs(historical_prices - current_price) / current_price
            
            # 支撑阻力强度 = 1 / (1 + 最小距离的平均值)
            min_distances = np.sort(price_distances)[:5]  # 最近的5个价格点
            avg_min_distance = np.mean(min_distances)
            
            strength = 1 / (1 + avg_min_distance * 100)
            return strength
        
        sr_strength = data['close'].rolling(window=window).apply(
            lambda x: calculate_strength(x), raw=False
        )
        
        return sr_strength


def simulate_trading_session():
    """模拟交易会话中的因子使用"""
    print("=" * 60)
    print("模拟交易会话 - 因子调用演示")
    print("=" * 60)
    
    # 1. 创建模拟的实时数据
    print("\n1. 准备实时市场数据...")
    dates = pd.date_range(start='2024-07-01', periods=1000, freq='H')
    np.random.seed(123)
    
    base_price = 65000
    returns = np.random.normal(0, 0.015, len(dates))
    prices = base_price * np.exp(np.cumsum(returns))
    
    market_data = pd.DataFrame({
        'open': prices + np.random.normal(0, 20, len(dates)),
        'high': prices + np.abs(np.random.normal(0, 80, len(dates))),
        'low': prices - np.abs(np.random.normal(0, 80, len(dates))),
        'close': prices,
        'volume': np.random.exponential(2000, len(dates))
    }, index=dates)
    
    # 修正OHLC关系
    market_data['high'] = np.maximum.reduce([
        market_data['open'], market_data['high'], 
        market_data['low'], market_data['close']
    ])
    market_data['low'] = np.minimum.reduce([
        market_data['open'], market_data['high'], 
        market_data['low'], market_data['close']
    ])
    
    print(f"市场数据准备完成: {market_data.shape}")
    print(f"时间范围: {market_data.index.min()} - {market_data.index.max()}")
    print(f"当前价格: {market_data['close'].iloc[-1]:.2f}")
    
    # 2. 批量计算所有技术因子
    print("\n2. 批量计算技术因子...")
    technical_factors = factor_engine.compute_factor_category(
        category='technical',
        data=market_data,
        symbol='BTC_USDT',
        timeframe='1h',
        save_results=False
    )
    
    print(f"技术因子计算完成: {technical_factors.shape}")
    print("技术因子列表:", list(technical_factors.columns)[:5], "...")
    
    # 3. 计算自定义因子
    print("\n3. 计算自定义因子...")
    custom_factors = factor_engine.compute_multiple_factors(
        factor_ids=['trend_strength', 'volatility_adjusted_momentum', 'support_resistance_strength'],
        data=market_data,
        symbol='BTC_USDT',
        timeframe='1h',
        save_results=False
    )
    
    print(f"自定义因子计算完成: {custom_factors.shape}")
    
    # 4. 模拟交易决策过程
    print("\n4. 模拟交易决策...")
    
    # 获取最新的因子值
    latest_factors = {}
    
    # 传统技术因子
    for factor_id in ['rsi', 'sma', 'ema', 'atr', 'volatility']:
        value = factor_engine.compute_single_factor(
            factor_id=factor_id,
            data=market_data,
            symbol='BTC_USDT',
            timeframe='1h',
            save_result=False
        )
        if value is not None:
            latest_factors[factor_id] = value.iloc[-1]
    
    # 自定义因子
    if not custom_factors.empty:
        for col in custom_factors.columns:
            latest_factors[col] = custom_factors[col].iloc[-1]
    
    print("\n当前因子值:")
    for factor_name, value in latest_factors.items():
        if not pd.isna(value):
            print(f"  {factor_name}: {value:.4f}")
    
    # 5. 生成交易信号
    print("\n5. 生成交易信号...")
    
    # 定义交易规则
    trading_rules = {
        'rsi': {'buy_below': 30, 'sell_above': 70, 'weight': 1.0},
        'trend_strength': {'buy_above': 2, 'sell_below': -2, 'weight': 1.5},
        'volatility_adjusted_momentum': {'buy_above': 0.5, 'sell_below': -0.5, 'weight': 1.2}
    }
    
    signals = {}
    total_signal = 0
    total_weight = 0
    
    for factor_name, rules in trading_rules.items():
        if factor_name in latest_factors:
            value = latest_factors[factor_name]
            weight = rules['weight']
            
            signal = 0
            if 'buy_below' in rules and value < rules['buy_below']:
                signal = 1  # 买入信号
            elif 'buy_above' in rules and value > rules['buy_above']:
                signal = 1  # 买入信号
            elif 'sell_above' in rules and value > rules['sell_above']:
                signal = -1  # 卖出信号
            elif 'sell_below' in rules and value < rules['sell_below']:
                signal = -1  # 卖出信号
            
            signals[factor_name] = {
                'value': value,
                'signal': signal,
                'weight': weight
            }
            
            total_signal += signal * weight
            total_weight += weight
    
    # 计算综合信号
    final_signal = total_signal / total_weight if total_weight > 0 else 0
    
    print("\n交易信号分析:")
    for factor_name, info in signals.items():
        signal_desc = "买入" if info['signal'] > 0 else "卖出" if info['signal'] < 0 else "持有"
        print(f"  {factor_name}: {info['value']:.4f} -> {signal_desc} (权重: {info['weight']})")
    
    print(f"\n综合信号: {final_signal:.3f}")
    
    if final_signal > 0.3:
        decision = "强烈买入"
    elif final_signal > 0.1:
        decision = "买入"
    elif final_signal < -0.3:
        decision = "强烈卖出"
    elif final_signal < -0.1:
        decision = "卖出"
    else:
        decision = "持有"
    
    print(f"交易决策: {decision}")
    
    # 6. 性能统计
    print("\n6. 系统性能统计...")
    stats = factor_engine.get_factor_statistics()
    cache_stats = trading_api.get_cache_stats()
    
    print(f"注册因子总数: {stats['total_factors']}")
    print("因子分类分布:")
    for category, subcats in stats['categories'].items():
        total = sum(subcats.values())
        print(f"  {category}: {total} 个")
    
    print(f"\n缓存统计:")
    print(f"  数据缓存: {cache_stats['data_cache_size']} 项")
    print(f"  因子缓存: {cache_stats['factor_cache_size']} 项")
    print(f"  缓存TTL: {cache_stats['cache_ttl_minutes']} 分钟")


def performance_benchmark():
    """性能基准测试"""
    print("\n" + "=" * 60)
    print("性能基准测试")
    print("=" * 60)
    
    # 创建大数据集
    dates = pd.date_range(start='2023-01-01', end='2024-08-01', freq='H')
    np.random.seed(42)
    
    n_periods = len(dates)
    base_price = 50000
    returns = np.random.normal(0, 0.01, n_periods)
    prices = base_price * np.exp(np.cumsum(returns))
    
    big_data = pd.DataFrame({
        'open': prices + np.random.normal(0, 10, n_periods),
        'high': prices + np.abs(np.random.normal(0, 50, n_periods)),
        'low': prices - np.abs(np.random.normal(0, 50, n_periods)),
        'close': prices,
        'volume': np.random.exponential(1000, n_periods)
    }, index=dates)
    
    big_data['high'] = np.maximum.reduce([big_data['open'], big_data['high'], big_data['low'], big_data['close']])
    big_data['low'] = np.minimum.reduce([big_data['open'], big_data['high'], big_data['low'], big_data['close']])
    
    print(f"测试数据集大小: {big_data.shape} ({len(big_data) / 1000:.1f}K 数据点)")
    
    # 测试批量计算性能
    import time
    
    start_time = time.time()
    
    all_factors = factor_engine.compute_multiple_factors(
        factor_ids=['rsi', 'sma', 'ema', 'macd', 'atr', 'volatility', 'bollinger_bands'],
        data=big_data,
        symbol='BTC_USDT',
        timeframe='1h',
        parallel=True,
        save_results=False
    )
    
    end_time = time.time()
    
    print(f"批量计算耗时: {end_time - start_time:.2f} 秒")
    print(f"计算的因子: {all_factors.shape[1]} 个")
    print(f"平均每个因子耗时: {(end_time - start_time) / all_factors.shape[1]:.3f} 秒")
    print(f"数据处理速度: {len(big_data) / (end_time - start_time):.0f} 行/秒")


def main():
    """主演示函数"""
    print("FactorMiner 实际交易因子使用演示")
    print("=" * 60)
    
    # 1. 创建自定义因子
    create_custom_factors()
    print(f"已创建自定义因子，总注册因子数: {len(factor_engine.registry.registered_factors)}")
    
    # 2. 模拟交易会话
    simulate_trading_session()
    
    # 3. 性能基准测试
    performance_benchmark()
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)
    
    print("\n💡 核心特性:")
    print("✅ 装饰器注册 - 简单创建新因子")
    print("✅ 并行计算 - 高效批量处理")
    print("✅ 智能缓存 - 避免重复计算")
    print("✅ 实时API - 毫秒级因子获取")
    print("✅ 类型安全 - 完整类型注解")
    print("✅ 错误处理 - 健壮的异常处理")
    
    print("\n🎯 适用场景:")
    print("- 实时量化交易系统")
    print("- 因子研究和回测")
    print("- 算法交易策略开发")
    print("- 风险管理系统")
    print("- 投资组合优化")


if __name__ == "__main__":
    main()
