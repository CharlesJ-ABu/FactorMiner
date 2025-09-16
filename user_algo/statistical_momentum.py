"""
统计动量因子挖掘算法
基于统计方法挖掘动量相关的预测因子
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def calculate_factors(data, **kwargs):
    """
    挖掘统计动量因子
    
    Args:
        data: DataFrame，包含 'open', 'high', 'low', 'close', 'volume' 列
        **kwargs: 算法参数
    
    Returns:
        Dict[str, pd.Series]: 因子名称 -> 因子值序列
    """
    factors = {}
    
    # 获取参数
    periods = kwargs.get('periods', [5, 10, 20, 50])
    
    try:
        # 计算收益率
        returns = data['close'].pct_change()
        
        # 1. 动量因子 - 基于价格变化
        for period in periods:
            # 简单动量
            momentum = data['close'] / data['close'].shift(period) - 1
            factors[f'momentum_{period}'] = momentum
            
            # 标准化动量（Z-score）
            momentum_mean = momentum.rolling(window=period*2).mean()
            momentum_std = momentum.rolling(window=period*2).std()
            factors[f'momentum_zscore_{period}'] = (momentum - momentum_mean) / momentum_std
            
            # 动量强度
            momentum_strength = momentum.rolling(window=period).rank(pct=True)
            factors[f'momentum_strength_{period}'] = momentum_strength
        
        # 2. 收益率动量因子
        for period in periods:
            # 收益率累积
            cum_returns = returns.rolling(window=period).sum()
            factors[f'cum_returns_{period}'] = cum_returns
            
            # 收益率波动率调整动量
            vol = returns.rolling(window=period).std()
            vol_adj_momentum = cum_returns / (vol + 1e-8)
            factors[f'vol_adj_momentum_{period}'] = vol_adj_momentum
            
            # 收益率偏度
            returns_skew = returns.rolling(window=period).skew()
            factors[f'returns_skew_{period}'] = returns_skew
        
        # 3. 价格位置动量因子
        for period in periods:
            # 价格在区间中的位置
            high_max = data['high'].rolling(window=period).max()
            low_min = data['low'].rolling(window=period).min()
            price_position = (data['close'] - low_min) / (high_max - low_min + 1e-8)
            factors[f'price_position_{period}'] = price_position
            
            # 价格位置变化率
            price_position_change = price_position.diff()
            factors[f'price_position_change_{period}'] = price_position_change
        
        # 4. 成交量动量因子
        for period in periods:
            # 成交量变化率
            volume_change = data['volume'].pct_change(period)
            factors[f'volume_momentum_{period}'] = volume_change
            
            # 成交量与价格动量的一致性
            price_momentum = data['close'].pct_change(period)
            volume_momentum = data['volume'].pct_change(period)
            momentum_consistency = (price_momentum * volume_momentum).rolling(window=period).mean()
            factors[f'momentum_consistency_{period}'] = momentum_consistency
        
        # 5. 复合动量因子
        for period in periods:
            # 多时间框架动量组合
            short_momentum = data['close'].pct_change(period//2)
            long_momentum = data['close'].pct_change(period)
            momentum_ratio = short_momentum / (long_momentum + 1e-8)
            factors[f'momentum_ratio_{period}'] = momentum_ratio
            
            # 动量加速度
            momentum_acceleration = momentum.diff()
            factors[f'momentum_acceleration_{period}'] = momentum_acceleration
        
        print(f"✅ 统计动量因子挖掘完成，生成 {len(factors)} 个因子")
        
    except Exception as e:
        print(f"❌ 统计动量因子挖掘失败: {e}")
    
    return factors

# 算法元信息
ALGORITHM_INFO = {
    'name': '统计动量因子挖掘',
    'description': '基于统计方法挖掘动量相关的预测因子',
    'category': 'statistical',
    'subcategory': 'momentum',
    'parameters': {
        'periods': {
            'type': 'list',
            'default': [5, 10, 20, 50],
            'description': '计算周期列表'
        }
    }
}
