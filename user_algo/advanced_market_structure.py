"""
高级市场结构因子挖掘算法
基于市场微观结构挖掘预测因子
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def calculate_factors(data, **kwargs):
    """
    挖掘市场结构因子
    
    Args:
        data: DataFrame，包含 'open', 'high', 'low', 'close', 'volume' 列
        **kwargs: 算法参数
    
    Returns:
        Dict[str, pd.Series]: 因子名称 -> 因子值序列
    """
    factors = {}
    
    # 获取参数
    periods = kwargs.get('periods', [10, 20, 50])
    
    try:
        # 1. 价格效率因子
        for period in periods:
            # 价格效率 = 净价格变化 / 总价格变化
            net_change = abs(data['close'] - data['close'].shift(period))
            total_change = data['close'].diff().abs().rolling(window=period).sum()
            price_efficiency = net_change / (total_change + 1e-8)
            factors[f'price_efficiency_{period}'] = price_efficiency
            
            # 价格效率变化率
            efficiency_change = price_efficiency.diff()
            factors[f'price_efficiency_change_{period}'] = efficiency_change
        
        # 2. 市场冲击因子
        for period in periods:
            # 价格冲击 = 价格变化 / 成交量
            price_change = data['close'].diff()
            volume_ma = data['volume'].rolling(window=period).mean()
            price_impact = price_change / (volume_ma + 1e-8)
            factors[f'price_impact_{period}'] = price_impact
            
            # 成交量冲击
            volume_change = data['volume'].diff()
            price_ma = data['close'].rolling(window=period).mean()
            volume_impact = volume_change / (price_ma + 1e-8)
            factors[f'volume_impact_{period}'] = volume_impact
        
        # 3. 支撑阻力强度因子
        for period in periods:
            # 支撑位强度
            support_level = data['low'].rolling(window=period).min()
            support_strength = (data['close'] - support_level) / (data['close'] + 1e-8)
            factors[f'support_strength_{period}'] = support_strength
            
            # 阻力位强度
            resistance_level = data['high'].rolling(window=period).max()
            resistance_strength = (resistance_level - data['close']) / (data['close'] + 1e-8)
            factors[f'resistance_strength_{period}'] = resistance_strength
            
            # 支撑阻力比率
            support_resistance_ratio = support_strength / (resistance_strength + 1e-8)
            factors[f'support_resistance_ratio_{period}'] = support_resistance_ratio
        
        # 4. 趋势强度因子
        for period in periods:
            # ADX趋势强度
            adx = _calculate_adx(data, period)
            factors[f'adx_{period}'] = adx
            
            # 趋势一致性
            price_trend = data['close'].rolling(window=period).apply(
                lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1, raw=False
            )
            volume_trend = data['volume'].rolling(window=period).apply(
                lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1, raw=False
            )
            trend_consistency = (price_trend * volume_trend).rolling(window=period).mean()
            factors[f'trend_consistency_{period}'] = trend_consistency
        
        # 5. 波动率结构因子
        for period in periods:
            # 波动率聚类
            returns = data['close'].pct_change()
            volatility = returns.rolling(window=period).std()
            vol_clustering = volatility.rolling(window=period).std()
            factors[f'vol_clustering_{period}'] = vol_clustering
            
            # 波动率均值回归
            vol_mean = volatility.rolling(window=period*2).mean()
            vol_mean_reversion = (volatility - vol_mean) / (vol_mean + 1e-8)
            factors[f'vol_mean_reversion_{period}'] = vol_mean_reversion
        
        # 6. 市场情绪因子
        for period in periods:
            # 价格成交量背离
            price_change = data['close'].pct_change(period)
            volume_change = data['volume'].pct_change(period)
            divergence = price_change - volume_change
            factors[f'price_volume_divergence_{period}'] = divergence
            
            # 市场恐慌指数
            returns = data['close'].pct_change()
            negative_returns = returns.where(returns < 0, 0)
            fear_index = negative_returns.rolling(window=period).sum()
            factors[f'fear_index_{period}'] = fear_index
        
        print(f"✅ 市场结构因子挖掘完成，生成 {len(factors)} 个因子")
        
    except Exception as e:
        print(f"❌ 市场结构因子挖掘失败: {e}")
    
    return factors

def _calculate_adx(data, period):
    """计算ADX指标"""
    high_diff = data['high'].diff()
    low_diff = data['low'].diff()
    
    plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
    minus_dm = -low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
    
    # 计算真实波幅
    high_low = data['high'] - data['low']
    high_close = np.abs(data['high'] - data['close'].shift())
    low_close = np.abs(data['low'] - data['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    
    # 计算DI
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / true_range.rolling(window=period).mean())
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / true_range.rolling(window=period).mean())
    
    # 计算DX和ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()
    
    return adx

# 算法元信息
ALGORITHM_INFO = {
    'name': '市场结构因子挖掘',
    'description': '基于市场微观结构挖掘预测因子',
    'category': 'advanced',
    'subcategory': 'market_structure',
    'parameters': {
        'periods': {
            'type': 'list',
            'default': [10, 20, 50],
            'description': '计算周期列表'
        }
    }
}
