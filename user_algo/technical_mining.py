"""
技术指标因子挖掘算法
基于技术分析指标的因子发现和优化
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# 算法信息
ALGORITHM_INFO = {
    # 必填字段
    'id': 'technical_mining',
    'name': '技术指标因子挖掘',
    'description': '基于技术分析指标的因子发现和优化算法',
    'category': 'technical',
    'version': '1.0.0',
    'author': 'FactorMiner Team',
    
    # 可选字段
    'parameters': {
        'max_factors': {
            'type': 'int',
            'default': 30,
            'description': '最大因子数量',
            'min': 1,
            'max': 200,
            'required': False
        },
        'min_ic': {
            'type': 'float',
            'default': 0.05,
            'description': '最小IC阈值',
            'min': 0.0,
            'max': 1.0,
            'required': False
        },
        'max_correlation': {
            'type': 'float',
            'default': 0.8,
            'description': '最大相关性阈值',
            'min': 0.0,
            'max': 1.0,
            'required': False
        }
    },
    'requirements': ['pandas', 'numpy'],
    'tags': ['technical', 'factor_mining'],
    'created_at': '2024-01-01',
    'updated_at': '2024-01-01'
},
        'min_ic': {'type': 'float', 'default': 0.04, 'description': '最小IC阈值'},
        'min_ir': {'type': 'float', 'default': 0.09, 'description': '最小IR阈值'},
        'max_correlation': {'type': 'float', 'default': 0.8, 'description': '最大相关性阈值'},
        'indicator_windows': {'type': 'list', 'default': [5, 10, 20, 50], 'description': '技术指标窗口列表'},
        'evaluation_periods': {'type': 'list', 'default': [1, 3, 5, 10], 'description': '评估周期列表'}
    }
}


def validate_data(data: pd.DataFrame) -> bool:
    """
    数据验证
    
    Args:
        data: 输入数据
        
    Returns:
        bool: 验证是否通过
    """
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    
    # 检查必要列
    if not all(col in data.columns for col in required_columns):
        print("❌ 缺少必要的列")
        return False
    
    # 检查数据长度
    if len(data) < 100:  # 最少需要100个数据点
        print("❌ 数据长度不足")
        return False
    
    # 检查数据质量
    if data[required_columns].isnull().all().any():
        print("❌ 数据包含全空列")
        return False
    
    # 检查价格数据合理性
    if (data['high'] < data['low']).any():
        print("❌ 价格数据不合理（high < low）")
        return False
    
    return True

def get_factor_info(factor_name: str) -> Dict:
    """
    获取因子信息
    
    Args:
        factor_name: 因子名称
        
    Returns:
        Dict: 因子信息
    """
    return {
        'name': f'因子_{factor_name}',
        'description': f'因子_{factor_name}',
        'type': 'technical',
        'parameters': {}
    }

def calculate_factors(data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
    """
    因子挖掘主函数
    
    Args:
        data: DataFrame，包含 'open', 'high', 'low', 'close', 'volume' 列
        **kwargs: 算法参数，来自ALGORITHM_INFO['parameters']
    
    Returns:
        Dict[str, pd.Series]: 挖掘出的因子字典
    """
    print("📈 开始技术指标因子挖掘...")
    
    # 获取参数
    max_factors = kwargs.get('max_factors', 35)
    min_ic = kwargs.get('min_ic', 0.04)
    min_ir = kwargs.get('min_ir', 0.09)
    max_correlation = kwargs.get('max_correlation', 0.8)
    indicator_windows = kwargs.get('indicator_windows', [5, 10, 20, 50])
    evaluation_periods = kwargs.get('evaluation_periods', [1, 3, 5, 10])
    
    # 1. 生成趋势因子
    print("📊 生成趋势因子...")
    trend_factors = _generate_trend_factors(data, indicator_windows)
    print(f"生成了 {len(trend_factors)} 个趋势因子")
    
    # 2. 生成动量因子
    print("⚡ 生成动量因子...")
    momentum_factors = _generate_momentum_factors(data, indicator_windows)
    print(f"生成了 {len(momentum_factors)} 个动量因子")
    
    # 3. 生成波动率因子
    print("📉 生成波动率因子...")
    volatility_factors = _generate_volatility_factors(data, indicator_windows)
    print(f"生成了 {len(volatility_factors)} 个波动率因子")
    
    # 4. 生成成交量因子
    print("📊 生成成交量因子...")
    volume_factors = _generate_volume_factors(data, indicator_windows)
    print(f"生成了 {len(volume_factors)} 个成交量因子")
    
    # 5. 生成价格形态因子
    print("🔍 生成价格形态因子...")
    pattern_factors = _generate_pattern_factors(data, indicator_windows)
    print(f"生成了 {len(pattern_factors)} 个价格形态因子")
    
    # 6. 合并所有因子
    all_factors = {**trend_factors, **momentum_factors, **volatility_factors, 
                   **volume_factors, **pattern_factors}
    print(f"总共生成了 {len(all_factors)} 个候选因子")
    
    # 7. 计算未来收益率
    print("📈 计算未来收益率...")
    future_returns = _calculate_future_returns(data, evaluation_periods)
    
    # 8. 评估因子质量
    print("🎯 评估因子质量...")
    factor_scores = _evaluate_factors(all_factors, future_returns, min_ic, min_ir)
    
    # 9. 因子去重和筛选
    print("🔧 因子去重和筛选...")
    selected_factors = _select_factors(
        all_factors, factor_scores, max_factors, max_correlation
    )
    
    print(f"✅ 技术指标因子挖掘完成，选择了 {len(selected_factors)} 个因子")
    return selected_factors

def _generate_trend_factors(data: pd.DataFrame, windows: List[int]) -> Dict[str, pd.Series]:
    """生成趋势因子"""
    factors = {}
    
    # 获取正确的列名
    close_col = 'close' if 'close' in data.columns else 'S_DQ_CLOSE'
    high_col = 'high' if 'high' in data.columns else 'S_DQ_HIGH'
    low_col = 'low' if 'low' in data.columns else 'S_DQ_LOW'
    open_col = 'open' if 'open' in data.columns else 'S_DQ_OPEN'
    
    for window in windows:
        # 移动平均线
        sma = data[close_col].rolling(window).mean()
        factors[f'sma_{window}'] = sma
        
        # 价格与移动平均线的关系
        factors[f'price_sma_ratio_{window}'] = data[close_col] / sma
        
        # 移动平均线斜率
        factors[f'sma_slope_{window}'] = sma.diff()
        
        # 移动平均线交叉
        if window > 5:
            short_sma = data[close_col].rolling(window//2).mean()
            factors[f'ma_cross_{window}'] = (short_sma - sma) / sma
        
        # 价格位置（在高低点之间的位置）
        high_max = data[high_col].rolling(window).max()
        low_min = data[low_col].rolling(window).min()
        factors[f'price_position_{window}'] = (data[close_col] - low_min) / (high_max - low_min + 1e-8)
        
        # 趋势强度
        price_change = data[close_col].diff()
        trend_strength = price_change.rolling(window).apply(
            lambda x: np.sum(x > 0) / len(x) if len(x) > 0 else 0.5
        )
        factors[f'trend_strength_{window}'] = trend_strength
    
    return factors

def _generate_momentum_factors(data: pd.DataFrame, windows: List[int]) -> Dict[str, pd.Series]:
    """生成动量因子"""
    factors = {}
    
    # 获取正确的列名
    close_col = 'close' if 'close' in data.columns else 'S_DQ_CLOSE'
    high_col = 'high' if 'high' in data.columns else 'S_DQ_HIGH'
    low_col = 'low' if 'low' in data.columns else 'S_DQ_LOW'
    
    for window in windows:
        # 简单动量
        momentum = data[close_col] / data[close_col].shift(window) - 1
        factors[f'momentum_{window}'] = momentum
        
        # 标准化动量
        momentum_mean = momentum.rolling(window*2).mean()
        momentum_std = momentum.rolling(window*2).std()
        factors[f'momentum_zscore_{window}'] = (momentum - momentum_mean) / (momentum_std + 1e-8)
        
        # RSI
        delta = data[close_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
        rs = gain / (loss + 1e-8)
        factors[f'rsi_{window}'] = 100 - (100 / (1 + rs))
        
        # MACD
        if window >= 12:
            ema_12 = data[close_col].ewm(span=12).mean()
            ema_26 = data[close_col].ewm(span=26).mean()
            macd_line = ema_12 - ema_26
            signal_line = macd_line.ewm(span=9).mean()
            factors[f'macd_{window}'] = macd_line
            factors[f'macd_signal_{window}'] = signal_line
            factors[f'macd_histogram_{window}'] = macd_line - signal_line
        
        # 随机指标
        high_max = data[high_col].rolling(window).max()
        low_min = data[low_col].rolling(window).min()
        k_percent = 100 * (data[close_col] - low_min) / (high_max - low_min + 1e-8)
        factors[f'stoch_k_{window}'] = k_percent
        factors[f'stoch_d_{window}'] = k_percent.rolling(3).mean()
        
        # 威廉指标
        factors[f'williams_r_{window}'] = -100 * (high_max - data[close_col]) / (high_max - low_min + 1e-8)
    
    return factors

def _generate_volatility_factors(data: pd.DataFrame, windows: List[int]) -> Dict[str, pd.Series]:
    """生成波动率因子"""
    factors = {}
    
    # 获取正确的列名
    close_col = 'close' if 'close' in data.columns else 'S_DQ_CLOSE'
    high_col = 'high' if 'high' in data.columns else 'S_DQ_HIGH'
    low_col = 'low' if 'low' in data.columns else 'S_DQ_LOW'
    
    for window in windows:
        returns = data[close_col].pct_change()
        
        # 历史波动率
        factors[f'volatility_{window}'] = returns.rolling(window).std()
        
        # 波动率变化
        factors[f'volatility_change_{window}'] = factors[f'volatility_{window}'].diff()
        
        # 相对波动率
        long_vol = returns.rolling(window*2).std()
        factors[f'relative_volatility_{window}'] = factors[f'volatility_{window}'] / (long_vol + 1e-8)
        
        # ATR (平均真实波幅)
        high_low = data[high_col] - data[low_col]
        high_close = np.abs(data[high_col] - data[close_col].shift(1))
        low_close = np.abs(data[low_col] - data[close_col].shift(1))
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        factors[f'atr_{window}'] = true_range.rolling(window).mean()
        
        # 布林带
        bb_mean = data[close_col].rolling(window).mean()
        bb_std = data[close_col].rolling(window).std()
        factors[f'bb_upper_{window}'] = bb_mean + 2 * bb_std
        factors[f'bb_lower_{window}'] = bb_mean - 2 * bb_std
        factors[f'bb_position_{window}'] = (data[close_col] - bb_mean) / (2 * bb_std + 1e-8)
        factors[f'bb_width_{window}'] = (factors[f'bb_upper_{window}'] - factors[f'bb_lower_{window}']) / bb_mean
        
        # 价格通道
        high_max = data[high_col].rolling(window).max()
        low_min = data[low_col].rolling(window).min()
        factors[f'price_channel_position_{window}'] = (data[close_col] - low_min) / (high_max - low_min + 1e-8)
        factors[f'price_channel_width_{window}'] = (high_max - low_min) / data[close_col]
    
    return factors

def _generate_volume_factors(data: pd.DataFrame, windows: List[int]) -> Dict[str, pd.Series]:
    """生成成交量因子"""
    factors = {}
    
    # 获取正确的列名
    close_col = 'close' if 'close' in data.columns else 'S_DQ_CLOSE'
    volume_col = 'volume' if 'volume' in data.columns else 'S_DQ_VOLUME'
    
    for window in windows:
        # 成交量移动平均
        volume_ma = data[volume_col].rolling(window).mean()
        factors[f'volume_ma_{window}'] = volume_ma
        
        # 成交量比率
        factors[f'volume_ratio_{window}'] = data[volume_col] / (volume_ma + 1e-8)
        
        # 成交量变化
        factors[f'volume_change_{window}'] = data[volume_col].pct_change()
        
        # 成交量加权平均价格 (VWAP)
        vwap = (data[close_col] * data[volume_col]).rolling(window).sum() / data[volume_col].rolling(window).sum()
        factors[f'vwap_{window}'] = vwap
        factors[f'price_vwap_ratio_{window}'] = data[close_col] / (vwap + 1e-8)
        
        # 成交量价格趋势 (VPT)
        vpt = (data[close_col].pct_change() * data[volume_col]).rolling(window).sum()
        factors[f'vpt_{window}'] = vpt
        
        # 能量潮 (OBV)
        obv = (data[close_col].diff() > 0).astype(int) * data[volume_col] - \
              (data[close_col].diff() < 0).astype(int) * data[volume_col]
        factors[f'obv_{window}'] = obv.rolling(window).sum()
        
        # 成交量动量
        factors[f'volume_momentum_{window}'] = data[volume_col] / data[volume_col].shift(window) - 1
        
        # 成交量波动率
        factors[f'volume_volatility_{window}'] = data[volume_col].pct_change().rolling(window).std()
    
    return factors

def _generate_pattern_factors(data: pd.DataFrame, windows: List[int]) -> Dict[str, pd.Series]:
    """生成价格形态因子"""
    factors = {}
    
    # 获取正确的列名
    close_col = 'close' if 'close' in data.columns else 'S_DQ_CLOSE'
    high_col = 'high' if 'high' in data.columns else 'S_DQ_HIGH'
    low_col = 'low' if 'low' in data.columns else 'S_DQ_LOW'
    open_col = 'open' if 'open' in data.columns else 'S_DQ_OPEN'
    
    for window in windows:
        # 价格形态识别
        # 锤子线
        body = np.abs(data[close_col] - data[open_col])
        upper_shadow = data[high_col] - np.maximum(data[close_col], data[open_col])
        lower_shadow = np.minimum(data[close_col], data[open_col]) - data[low_col]
        factors[f'hammer_{window}'] = ((lower_shadow > 2 * body) & (upper_shadow < body)).astype(float)
        
        # 十字星
        factors[f'doji_{window}'] = (body < (data[high_col] - data[low_col]) * 0.1).astype(float)
        
        # 长上影线
        factors[f'long_upper_shadow_{window}'] = (upper_shadow > 2 * body).astype(float)
        
        # 长下影线
        factors[f'long_lower_shadow_{window}'] = (lower_shadow > 2 * body).astype(float)
        
        # 价格突破
        high_max = data[high_col].rolling(window).max()
        low_min = data[low_col].rolling(window).min()
        factors[f'breakout_high_{window}'] = (data[close_col] > high_max.shift(1)).astype(float)
        factors[f'breakout_low_{window}'] = (data[close_col] < low_min.shift(1)).astype(float)
        
        # 支撑阻力位
        factors[f'support_level_{window}'] = low_min
        factors[f'resistance_level_{window}'] = high_max
        factors[f'support_distance_{window}'] = (data[close_col] - low_min) / data[close_col]
        factors[f'resistance_distance_{window}'] = (high_max - data[close_col]) / data[close_col]
        
        # 价格通道突破
        channel_high = data[high_col].rolling(window).max()
        channel_low = data[low_col].rolling(window).min()
        factors[f'channel_breakout_{window}'] = (
            (data[close_col] > channel_high.shift(1)) | 
            (data[close_col] < channel_low.shift(1))
        ).astype(float)
    
    return factors

def _calculate_future_returns(data: pd.DataFrame, periods: List[int]) -> Dict[str, pd.Series]:
    """计算未来收益率"""
    close_col = 'close' if 'close' in data.columns else 'S_DQ_CLOSE'
    returns = data[close_col].pct_change()
    
    future_returns = {}
    for period in periods:
        future_returns[f'future_return_{period}'] = returns.shift(-period)
    
    return future_returns

def _evaluate_factors(factors: Dict[str, pd.Series], 
                     future_returns: Dict[str, pd.Series],
                     min_ic: float, min_ir: float) -> Dict[str, Dict]:
    """评估因子质量"""
    scores = {}
    
    for factor_name, factor_series in factors.items():
        if factor_series.isna().all():
            continue
            
        factor_scores = {
            'ic_mean': 0,
            'ir_mean': 0,
            'ic_std': 0,
            'valid_periods': 0,
            'score': 0
        }
        
        valid_periods = 0
        ic_values = []
        
        for period_name, future_return in future_returns.items():
            # 计算IC
            ic = factor_series.corr(future_return)
            if not np.isnan(ic):
                ic_values.append(ic)
                valid_periods += 1
        
        if valid_periods > 0:
            ic_values = np.array(ic_values)
            factor_scores['ic_mean'] = np.mean(ic_values)
            factor_scores['ic_std'] = np.std(ic_values)
            factor_scores['ir_mean'] = factor_scores['ic_mean'] / factor_scores['ic_std'] if factor_scores['ic_std'] > 0 else 0
            factor_scores['valid_periods'] = valid_periods
            
            # 综合评分
            ic_score = abs(factor_scores['ic_mean']) if abs(factor_scores['ic_mean']) >= min_ic else 0
            ir_score = abs(factor_scores['ir_mean']) if abs(factor_scores['ir_mean']) >= min_ir else 0
            factor_scores['score'] = ic_score * 0.5 + ir_score * 0.5
        
        scores[factor_name] = factor_scores
    
    return scores

def _select_factors(factors: Dict[str, pd.Series], 
                   scores: Dict[str, Dict],
                   max_factors: int,
                   max_correlation: float) -> Dict[str, pd.Series]:
    """选择最优因子"""
    # 按评分排序
    sorted_factors = sorted(scores.items(), key=lambda x: x[1]['score'], reverse=True)
    
    selected_factors = {}
    selected_names = []
    
    for factor_name, score_info in sorted_factors:
        if len(selected_factors) >= max_factors:
            break
            
        if score_info['score'] <= 0:
            continue
            
        # 检查相关性
        factor_series = factors[factor_name]
        is_highly_correlated = False
        
        for selected_name in selected_names:
            selected_series = factors[selected_name]
            correlation = factor_series.corr(selected_series)
            if not np.isnan(correlation) and abs(correlation) > max_correlation:
                is_highly_correlated = True
                break
        
        if not is_highly_correlated:
            selected_factors[factor_name] = factor_series
            selected_names.append(factor_name)
    
    return selected_factors

def calculate_single_factor(data: pd.DataFrame, factor_name: str, **kwargs) -> pd.Series:
    """
    计算单个因子
    
    Args:
        data: DataFrame，包含 'open', 'high', 'low', 'close', 'volume' 列
        factor_name: 因子名称
        **kwargs: 算法参数，来自ALGORITHM_INFO['parameters']
        
    Returns:
        pd.Series: 因子值序列，索引与data相同
    """
    try:
        # 数据验证
        if not validate_data(data):
            print(f"❌ 数据验证失败，无法计算因子 {factor_name}")
            return pd.Series(index=data.index, dtype=float)
        
        # 这里应该实现具体的因子计算逻辑
        # 作为示例，返回随机值
        print(f"⚠️ 注意：{factor_name} 需要实现具体计算逻辑，当前返回随机值")
        
        np.random.seed(42)
        factor_values = np.random.normal(0, 1, len(data))
        
        return pd.Series(factor_values, index=data.index, name=factor_name)
        
    except Exception as e:
        print(f"❌ 计算因子 {factor_name} 时出错: {str(e)}")
        return pd.Series(index=data.index, dtype=float)
