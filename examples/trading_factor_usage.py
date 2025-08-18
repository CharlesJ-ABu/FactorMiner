#!/usr/bin/env python3
"""
交易因子使用示例
展示如何在交易策略中使用因子
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from factor_miner.core.factor_storage import TransparentFactorStorage


def create_sample_data():
    """创建示例市场数据"""
    print("=" * 60)
    print("1. 创建示例市场数据")
    print("=" * 60)
    
    # 生成一年的小时级数据
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='H')
    n_periods = len(dates)
    
    # 生成模拟的BTC价格数据
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
    data['low'] = np.minimum.reduce([data['open'], data['high'], data['low'], data['close'])
    
    print(f"数据形状: {data.shape}")
    print(f"时间范围: {data.index.min()} 到 {data.index.max()}")
    print(f"价格范围: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
    
    return data


def calculate_basic_factors(data):
    """计算基本技术因子"""
    print("\n" + "=" * 60)
    print("2. 计算基本技术因子")
    print("=" * 60)
    
    factors = pd.DataFrame(index=data.index)
    
    # 移动平均线
    factors['ma_20'] = data['close'].rolling(window=20).mean()
    factors['ma_50'] = data['close'].rolling(window=50).mean()
    
    # 价格动量
    factors['momentum_5'] = data['close'] / data['close'].shift(5) - 1
    factors['momentum_20'] = data['close'] / data['close'].shift(20) - 1
    
    # RSI
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-8)
    factors['rsi'] = 100 - (100 / (1 + rs))
    
    # 布林带
    factors['bb_middle'] = data['close'].rolling(window=20).mean()
    bb_std = data['close'].rolling(window=20).std()
    factors['bb_upper'] = factors['bb_middle'] + (bb_std * 2)
    factors['bb_lower'] = factors['bb_middle'] - (bb_std * 2)
    
    # 成交量指标
    factors['volume_ma'] = data['volume'].rolling(window=20).mean()
    factors['volume_ratio'] = data['volume'] / factors['volume_ma']
    
    print("✅ 基本因子计算完成")
    print(f"因子数量: {len(factors.columns)}")
    print("因子列表:", list(factors.columns))
    
    return factors


def generate_trading_signals(factors, data):
    """生成交易信号"""
    print("\n" + "=" * 60)
    print("3. 生成交易信号")
    print("=" * 60)
    
    signals = pd.DataFrame(index=data.index)
    
    # 趋势跟踪信号
    signals['trend_signal'] = 0
    signals.loc[factors['ma_20'] > factors['ma_50'], 'trend_signal'] = 1  # 上升趋势
    signals.loc[factors['ma_20'] < factors['ma_50'], 'trend_signal'] = -1  # 下降趋势
    
    # 动量信号
    signals['momentum_signal'] = 0
    signals.loc[factors['momentum_20'] > 0.05, 'momentum_signal'] = 1  # 强动量
    signals.loc[factors['momentum_20'] < -0.05, 'momentum_signal'] = -1  # 负动量
    
    # RSI信号
    signals['rsi_signal'] = 0
    signals.loc[factors['rsi'] < 30, 'rsi_signal'] = 1  # 超卖
    signals.loc[factors['rsi'] > 70, 'rsi_signal'] = -1  # 超买
    
    # 布林带信号
    signals['bb_signal'] = 0
    signals.loc[data['close'] < factors['bb_lower'], 'bb_signal'] = 1  # 价格触及下轨
    signals.loc[data['close'] > factors['bb_upper'], 'bb_signal'] = -1  # 价格触及上轨
    
    # 成交量确认信号
    signals['volume_signal'] = 0
    signals.loc[factors['volume_ratio'] > 1.5, 'volume_signal'] = 1  # 放量
    signals.loc[factors['volume_ratio'] < 0.5, 'volume_signal'] = -1  # 缩量
    
    # 综合信号
    signals['combined_signal'] = (
        signals['trend_signal'] * 0.3 +
        signals['momentum_signal'] * 0.25 +
        signals['rsi_signal'] * 0.2 +
        signals['bb_signal'] * 0.15 +
        signals['volume_signal'] * 0.1
    )
    
    # 信号强度分类
    signals['signal_strength'] = 'neutral'
    signals.loc[signals['combined_signal'] > 0.5, 'signal_strength'] = 'strong_buy'
    signals.loc[signals['combined_signal'] > 0.2, 'signal_strength'] = 'buy'
    signals.loc[signals['combined_signal'] < -0.5, 'signal_strength'] = 'strong_sell'
    signals.loc[signals['combined_signal'] < -0.2, 'signal_strength'] = 'sell'
    
    print("✅ 交易信号生成完成")
    print("信号类型:", list(signals.columns))
    
    return signals


def backtest_strategy(signals, data, initial_capital=100000):
    """回测策略"""
    print("\n" + "=" * 60)
    print("4. 策略回测")
    print("=" * 60)
    
    # 创建回测结果DataFrame
    backtest = pd.DataFrame(index=data.index)
    backtest['price'] = data['close']
    backtest['signal'] = signals['combined_signal']
    backtest['position'] = 0
    
    # 根据信号确定仓位
    backtest.loc[backtest['signal'] > 0.3, 'position'] = 1  # 买入信号
    backtest.loc[backtest['signal'] < -0.3, 'position'] = -1  # 卖出信号
    
    # 计算收益率
    backtest['returns'] = backtest['price'].pct_change()
    backtest['strategy_returns'] = backtest['position'].shift(1) * backtest['returns']
    
    # 计算累积收益
    backtest['cumulative_returns'] = (1 + backtest['returns']).cumprod()
    backtest['strategy_cumulative_returns'] = (1 + backtest['strategy_returns']).cumprod()
    
    # 计算策略表现指标
    total_return = backtest['strategy_cumulative_returns'].iloc[-1] - 1
    annual_return = (1 + total_return) ** (365 / len(backtest)) - 1
    volatility = backtest['strategy_returns'].std() * np.sqrt(365 * 24)  # 年化波动率
    sharpe_ratio = annual_return / volatility if volatility > 0 else 0
    
    # 计算最大回撤
    cumulative = backtest['strategy_cumulative_returns']
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    print("✅ 回测完成")
    print(f"总收益率: {total_return:.2%}")
    print(f"年化收益率: {annual_return:.2%}")
    print(f"年化波动率: {volatility:.2%}")
    print(f"夏普比率: {sharpe_ratio:.2f}")
    print(f"最大回撤: {max_drawdown:.2%}")
    
    return backtest


def analyze_factor_contribution(factors, signals, data):
    """分析因子贡献度"""
    print("\n" + "=" * 60)
    print("5. 因子贡献度分析")
    print("=" * 60)
    
    # 计算各因子与价格的相关性
    correlations = {}
    for col in factors.columns:
        if not factors[col].isna().all():
            corr = factors[col].corr(data['close'])
            correlations[col] = corr
    
    # 排序显示相关性
    sorted_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    
    print("因子与价格的相关性:")
    for factor, corr in sorted_correlations[:10]:
        print(f"  {factor}: {corr:.4f}")
    
    # 分析信号质量
    signal_quality = {}
    for col in signals.columns:
        if col.endswith('_signal') and col != 'combined_signal':
            # 计算信号与未来收益的相关性
            future_returns = data['close'].pct_change().shift(-1)
            signal_corr = signals[col].corr(future_returns)
            signal_quality[col] = signal_corr
    
    print("\n信号预测质量 (与未来收益的相关性):")
    sorted_signals = sorted(signal_quality.items(), key=lambda x: abs(x[1]), reverse=True)
    for signal, corr in sorted_signals:
        print(f"  {signal}: {corr:.4f}")
    
    return correlations, signal_quality


def main():
    """主函数"""
    print("🚀 FactorMiner 交易因子使用示例")
    print("=" * 60)
    
    try:
        # 创建示例数据
        data = create_sample_data()
        
        # 计算因子
        factors = calculate_basic_factors(data)
        
        # 生成交易信号
        signals = generate_trading_signals(factors, data)
        
        # 回测策略
        backtest = backtest_strategy(signals, data)
        
        # 分析因子贡献
        correlations, signal_quality = analyze_factor_contribution(factors, signals, data)
        
        print("\n" + "=" * 60)
        print("✅ 所有演示完成！")
        print("=" * 60)
        
        # 保存结果
        output_dir = Path(__file__).parent / "output"
        output_dir.mkdir(exist_ok=True)
        
        # 保存因子数据
        factors.to_csv(output_dir / "factors.csv")
        signals.to_csv(output_dir / "signals.csv")
        backtest.to_csv(output_dir / "backtest.csv")
        
        print(f"结果已保存到: {output_dir}")
        
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
