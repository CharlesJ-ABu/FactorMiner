#!/usr/bin/env python3
"""
因子存储系统演示
展示新的因子注册、计算和存储功能
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from factor_miner.core.factor_registry import factor_registry, register_factor
from factor_miner.core.factor_engine import factor_engine
from factor_miner.core.factor_trading_api import trading_api


def demo_factor_registration():
    """演示因子注册功能"""
    print("=" * 60)
    print("1. 因子注册演示")
    print("=" * 60)
    
    # 注册一个自定义因子
    @register_factor(
        factor_id='custom_momentum',
        name='自定义动量因子',
        description='结合价格和成交量的动量指标',
        category='custom',
        subcategory='momentum',
        parameters={'price_period': 10, 'volume_period': 20}
    )
    def calculate_custom_momentum(data, price_period=10, volume_period=20):
        """计算自定义动量因子"""
        price_momentum = data['close'] / data['close'].shift(price_period) - 1
        volume_momentum = data['volume'] / data['volume'].rolling(volume_period).mean() - 1
        return price_momentum * volume_momentum
    
    # 注册另一个因子
    @register_factor(
        factor_id='price_volume_correlation',
        name='价量相关性',
        description='价格和成交量的滚动相关性',
        category='statistical',
        subcategory='correlation',
        parameters={'window': 20}
    )
    def calculate_price_volume_corr(data, window=20):
        """计算价格和成交量的相关性"""
        price_returns = data['close'].pct_change()
        volume_changes = data['volume'].pct_change()
        correlation = price_returns.rolling(window=window).corr(volume_changes)
        return correlation
    
    print(f"已注册因子: {len(factor_registry.registered_factors)} 个")
    print("注册的因子列表:")
    for factor_id, factor_def in factor_registry.registered_factors.items():
        print(f"  - {factor_id}: {factor_def.name} ({factor_def.category}.{factor_def.subcategory})")


def demo_factor_computation():
    """演示因子计算功能"""
    print("\n" + "=" * 60)
    print("2. 因子计算演示")
    print("=" * 60)
    
    # 创建模拟数据
    print("创建模拟数据...")
    dates = pd.date_range(start='2024-01-01', end='2024-08-01', freq='H')
    n_periods = len(dates)
    
    # 生成模拟的OHLCV数据
    np.random.seed(42)
    base_price = 50000
    returns = np.random.normal(0, 0.02, n_periods)
    prices = base_price * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        'open': prices + np.random.normal(0, 10, n_periods),
        'high': prices + np.abs(np.random.normal(0, 50, n_periods)),
        'low': prices - np.abs(np.random.normal(0, 50, n_periods)),
        'close': prices,
        'volume': np.random.exponential(1000, n_periods)
    }, index=dates)
    
    # 确保OHLC数据的逻辑关系
    data['high'] = np.maximum.reduce([data['open'], data['high'], data['low'], data['close']])
    data['low'] = np.minimum.reduce([data['open'], data['high'], data['low'], data['close']])
    
    print(f"数据形状: {data.shape}")
    print(f"数据时间范围: {data.index.min()} 到 {data.index.max()}")
    
    # 计算单个因子
    print("\n计算单个因子 (RSI)...")
    rsi_values = factor_engine.compute_single_factor(
        factor_id='rsi',
        data=data,
        symbol='BTC_USDT',
        timeframe='1h',
        period=14
    )
    
    if rsi_values is not None:
        print(f"RSI计算成功，形状: {rsi_values.shape}")
        print(f"最近5个值: {rsi_values.tail().tolist()}")
    
    # 批量计算多个因子
    print("\n批量计算技术因子...")
    technical_factors = factor_engine.compute_factor_category(
        category='technical',
        data=data,
        symbol='BTC_USDT',
        timeframe='1h'
    )
    
    print(f"技术因子计算完成，形状: {technical_factors.shape}")
    print(f"计算的因子: {list(technical_factors.columns)}")
    
    # 计算自定义因子
    print("\n计算自定义因子...")
    custom_factors = factor_engine.compute_multiple_factors(
        factor_ids=['custom_momentum', 'price_volume_correlation'],
        data=data,
        symbol='BTC_USDT',
        timeframe='1h'
    )
    
    print(f"自定义因子计算完成，形状: {custom_factors.shape}")
    print(f"自定义因子: {list(custom_factors.columns)}")
    
    return data


def demo_trading_api(sample_data):
    """演示交易API功能"""
    print("\n" + "=" * 60)
    print("3. 交易API演示")
    print("=" * 60)
    
    # 获取单个因子值
    print("获取RSI实时值...")
    rsi_value = trading_api.get_factor_value(
        factor_id='rsi',
        symbol='BTC_USDT',
        timeframe='1h',
        lookback_periods=100,
        period=14
    )
    
    if rsi_value is not None:
        print(f"当前RSI值: {rsi_value:.2f}")
    
    # 获取多个因子值
    print("\n获取多个因子值...")
    factor_values = trading_api.get_multiple_factor_values(
        factor_ids=['rsi', 'sma', 'ema', 'atr'],
        symbol='BTC_USDT',
        timeframe='1h',
        lookback_periods=100
    )
    
    print("当前因子值:")
    for factor_id, value in factor_values.items():
        if value is not None:
            print(f"  {factor_id}: {value:.4f}")
    
    # 获取交易信号
    print("\n获取交易信号...")
    factor_configs = [
        {
            'factor_id': 'rsi',
            'params': {'period': 14},
            'signal_rules': {
                'buy_threshold': 30,
                'sell_threshold': 70,
                'signal_type': 'threshold'
            },
            'weight': 1.0
        },
        {
            'factor_id': 'price_momentum',
            'params': {'period': 10},
            'signal_rules': {
                'buy_threshold': -0.05,
                'sell_threshold': 0.05,
                'signal_type': 'threshold'
            },
            'weight': 0.8
        }
    ]
    
    signals = trading_api.get_factor_signals(
        factor_configs=factor_configs,
        symbol='BTC_USDT',
        timeframe='1h',
        lookback_periods=100
    )
    
    print("交易信号:")
    print(f"  综合信号: {signals['combined_signal']:.3f}")
    print(f"  信号强度: {signals['signal_strength']:.3f}")
    print("  因子信号详情:")
    for factor_id, signal_info in signals['factor_signals'].items():
        print(f"    {factor_id}: 值={signal_info['value']:.4f}, 信号={signal_info['signal']:.2f}")


def demo_storage_stats():
    """演示存储统计功能"""
    print("\n" + "=" * 60)
    print("4. 存储统计演示")
    print("=" * 60)
    
    # 获取因子统计信息
    factor_stats = factor_engine.get_factor_statistics()
    
    print("因子注册统计:")
    print(f"  总因子数: {factor_stats['total_factors']}")
    print("  分类统计:")
    for category, subcategories in factor_stats['categories'].items():
        print(f"    {category}:")
        for subcategory, count in subcategories.items():
            print(f"      {subcategory}: {count} 个")
    
    print(f"\n存储路径: {factor_stats['storage_path']}")
    print(f"最后更新: {factor_stats['last_updated']}")
    
    # 获取缓存统计
    cache_stats = trading_api.get_cache_stats()
    print(f"\n缓存统计:")
    print(f"  数据缓存: {cache_stats['data_cache_size']} 项")
    print(f"  因子缓存: {cache_stats['factor_cache_size']} 项")
    print(f"  缓存TTL: {cache_stats['cache_ttl_minutes']} 分钟")
    print(f"  缓存启用: {cache_stats['cache_enabled']}")


def demo_factor_list():
    """演示因子列表功能"""
    print("\n" + "=" * 60)
    print("5. 可用因子列表")
    print("=" * 60)
    
    available_factors = factor_engine.get_available_factors()
    
    print(f"总共可用因子: {len(available_factors)} 个\n")
    
    # 按分类显示
    categories = {}
    for factor in available_factors:
        category = factor['category']
        if category not in categories:
            categories[category] = []
        categories[category].append(factor)
    
    for category, factors in categories.items():
        print(f"{category.upper()} 因子 ({len(factors)} 个):")
        for factor in factors:
            print(f"  {factor['factor_id']}: {factor['name']}")
            print(f"    描述: {factor['description']}")
            print(f"    子类: {factor['subcategory']}")
            if factor['parameters']:
                print(f"    参数: {factor['parameters']}")
            print()


def main():
    """主演示函数"""
    print("FactorMiner 新因子存储系统演示")
    print("=" * 60)
    
    try:
        # 1. 因子注册演示
        demo_factor_registration()
        
        # 2. 因子计算演示
        sample_data = demo_factor_computation()
        
        # 3. 交易API演示
        demo_trading_api(sample_data)
        
        # 4. 存储统计演示
        demo_storage_stats()
        
        # 5. 因子列表演示
        demo_factor_list()
        
        print("\n" + "=" * 60)
        print("演示完成！")
        print("=" * 60)
        
        print("\n新因子存储系统的特点:")
        print("✅ 统一的因子注册机制")
        print("✅ 高效的因子计算引擎")
        print("✅ 智能缓存系统")
        print("✅ 专为交易设计的API")
        print("✅ 灵活的因子存储格式")
        print("✅ 完整的元数据管理")
        
        print("\n使用方法:")
        print("1. 使用 @register_factor 装饰器注册新因子")
        print("2. 使用 factor_engine 进行批量计算和存储")
        print("3. 使用 trading_api 在交易中实时获取因子值")
        print("4. 所有因子数据自动缓存和管理")
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
