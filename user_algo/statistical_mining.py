"""
统计因子挖掘算法
基于统计分析进行因子发现、评估和优化
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 算法信息
ALGORITHM_INFO = {
    # 必填字段
    'id': 'statistical_mining',
    'name': '统计因子挖掘',
    'description': '基于统计分析的因子发现、评估和优化算法',
    'category': 'statistical',
    'version': '1.0.0',
    'author': 'FactorMiner Team',
    
    # 可选字段
    'parameters': {
        'max_factors': {
            'type': 'int',
            'default': 50,
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
        'min_ir': {
            'type': 'float',
            'default': 0.1,
            'description': '最小IR阈值',
            'min': 0.0,
            'max': 2.0,
            'required': False
        },
        'max_correlation': {
            'type': 'float',
            'default': 0.8,
            'description': '最大相关性阈值',
            'min': 0.0,
            'max': 1.0,
            'required': False
        },
        'windows': {
            'type': 'list',
            'default': [5, 10, 20, 50],
            'description': '滚动窗口列表',
            'required': False
        },
        'evaluation_periods': {
            'type': 'list',
            'default': [1, 5, 10],
            'description': '评估周期列表',
            'required': False
        }
    },
    'requirements': ['pandas', 'numpy', 'scipy'],
    'tags': ['statistical', 'momentum', 'mean_reversion'],
    'created_at': '2024-01-01',
    'updated_at': '2024-01-01'
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
    # 根据因子名称返回相关信息
    if 'momentum' in factor_name:
        return {
            'name': f'动量因子_{factor_name}',
            'description': f'基于动量的{factor_name}因子',
            'type': 'statistical',
            'parameters': {'type': 'momentum'}
        }
    elif 'mean_reversion' in factor_name:
        return {
            'name': f'均值回归因子_{factor_name}',
            'description': f'基于均值回归的{factor_name}因子',
            'type': 'statistical',
            'parameters': {'type': 'mean_reversion'}
        }
    elif 'volatility' in factor_name:
        return {
            'name': f'波动率因子_{factor_name}',
            'description': f'基于波动率的{factor_name}因子',
            'type': 'statistical',
            'parameters': {'type': 'volatility'}
        }
    elif 'volume' in factor_name:
        return {
            'name': f'成交量因子_{factor_name}',
            'description': f'基于成交量的{factor_name}因子',
            'type': 'statistical',
            'parameters': {'type': 'volume'}
        }
    else:
        return {
            'name': f'统计因子_{factor_name}',
            'description': f'统计因子_{factor_name}',
            'type': 'statistical',
            'parameters': {}
        }

def calculate_factors(data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
    """
    统计因子挖掘主函数
    
    Args:
        data: DataFrame，包含 'open', 'high', 'low', 'close', 'volume' 列
        **kwargs: 算法参数，来自ALGORITHM_INFO['parameters']
    
    Returns:
        Dict[str, pd.Series]: 挖掘出的因子字典
    """
    print("🔍 开始统计因子挖掘...")
    
    # 1. 数据验证
    if not validate_data(data):
        raise ValueError("数据验证失败")
    
    print(f"✅ 数据验证通过，共 {len(data)} 条记录")
    
    # 2. 获取参数
    max_factors = kwargs.get('max_factors', ALGORITHM_INFO['parameters']['max_factors']['default'])
    min_ic = kwargs.get('min_ic', 0.05)
    min_ir = kwargs.get('min_ir', 0.1)
    max_correlation = kwargs.get('max_correlation', 0.8)
    windows = kwargs.get('windows', [5, 10, 20, 50])
    evaluation_periods = kwargs.get('evaluation_periods', [1, 5, 10])
    
    # 1. 生成候选因子
    print("📊 生成候选因子...")
    candidate_factors = _generate_candidate_factors(data, windows)
    print(f"生成了 {len(candidate_factors)} 个候选因子")
    
    # 2. 计算未来收益率（用于评估）
    print("📈 计算未来收益率...")
    future_returns = _calculate_future_returns(data, evaluation_periods)
    
    # 3. 评估因子质量
    print("🎯 评估因子质量...")
    factor_scores = _evaluate_factors(candidate_factors, future_returns, min_ic, min_ir)
    
    # 4. 因子去重和筛选
    print("🔧 因子去重和筛选...")
    selected_factors = _select_factors(
        candidate_factors, factor_scores, max_factors, max_correlation
    )
    
    print(f"✅ 统计因子挖掘完成，选择了 {len(selected_factors)} 个因子")
    return selected_factors

def _generate_candidate_factors(data: pd.DataFrame, windows: List[int]) -> Dict[str, pd.Series]:
    """生成候选因子"""
    factors = {}
    
    # 获取正确的列名
    close_col = 'close' if 'close' in data.columns else 'S_DQ_CLOSE'
    volume_col = 'volume' if 'volume' in data.columns else 'S_DQ_VOLUME'
    high_col = 'high' if 'high' in data.columns else 'S_DQ_HIGH'
    low_col = 'low' if 'low' in data.columns else 'S_DQ_LOW'
    open_col = 'open' if 'open' in data.columns else 'S_DQ_OPEN'
    
    # 计算基础数据
    returns = data[close_col].pct_change()
    log_returns = np.log(data[close_col] / data[close_col].shift(1))
    volume_change = data[volume_col].pct_change()
    
    # 1. 价格动量因子
    for window in windows:
        # 简单动量
        momentum = data[close_col] / data[close_col].shift(window) - 1
        factors[f'momentum_{window}'] = momentum
        
        # 标准化动量
        momentum_mean = momentum.rolling(window=window*2).mean()
        momentum_std = momentum.rolling(window=window*2).std()
        factors[f'momentum_zscore_{window}'] = (momentum - momentum_mean) / momentum_std
        
        # 动量强度
        factors[f'momentum_strength_{window}'] = momentum.rolling(window=window).rank(pct=True)
        
        # 累积收益率
        factors[f'cum_returns_{window}'] = returns.rolling(window=window).sum()
        
        # 波动率调整动量
        vol = returns.rolling(window=window).std()
        factors[f'vol_adj_momentum_{window}'] = momentum / vol
        
        # 收益率偏度
        factors[f'returns_skew_{window}'] = returns.rolling(window=window).skew()
        
        # 收益率峰度
        factors[f'returns_kurt_{window}'] = returns.rolling(window=window).kurt()
    
    # 2. 价格位置因子
    for window in windows:
        # 价格在窗口内的位置
        high_max = data[high_col].rolling(window=window).max()
        low_min = data[low_col].rolling(window=window).min()
        factors[f'price_position_{window}'] = (data[close_col] - low_min) / (high_max - low_min)
        
        # 价格位置变化
        factors[f'price_position_change_{window}'] = factors[f'price_position_{window}'].diff()
    
    # 3. 成交量因子
    for window in windows:
        # 成交量动量
        factors[f'volume_momentum_{window}'] = data[volume_col] / data[volume_col].rolling(window=window).mean() - 1
        
        # 成交量加权价格
        vwap = (data[close_col] * data[volume_col]).rolling(window=window).sum() / data[volume_col].rolling(window=window).sum()
        factors[f'price_vwap_ratio_{window}'] = data[close_col] / vwap
    
    # 4. 技术指标因子
    for window in windows:
        # RSI
        delta = data[close_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        factors[f'rsi_{window}'] = 100 - (100 / (1 + rs))
        
        # 布林带位置
        bb_mean = data[close_col].rolling(window=window).mean()
        bb_std = data[close_col].rolling(window=window).std()
        factors[f'bb_position_{window}'] = (data[close_col] - bb_mean) / (2 * bb_std)
    
    # 5. 相关性因子
    for window in windows:
        # 价格-成交量相关性
        factors[f'price_volume_corr_{window}'] = returns.rolling(window=window).corr(volume_change)
        
        # 价格自相关性
        factors[f'price_autocorr_{window}'] = returns.rolling(window=window).apply(
            lambda x: x.autocorr() if len(x) > 1 else np.nan
        )
    
    # 6. 波动率因子
    for window in windows:
        # 滚动波动率
        factors[f'volatility_{window}'] = returns.rolling(window=window).std()
        
        # 波动率变化
        factors[f'volatility_change_{window}'] = factors[f'volatility_{window}'].diff()
        
        # 相对波动率
        long_vol = returns.rolling(window=window*2).std()
        factors[f'relative_volatility_{window}'] = factors[f'volatility_{window}'] / long_vol
    
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
            factor_scores['score'] = ic_score * 0.7 + ir_score * 0.3
        
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
        
        # 根据因子名称计算对应的统计因子
        if 'momentum' in factor_name:
            # 动量因子
            window = 20  # 默认窗口
            if '_' in factor_name:
                try:
                    window = int(factor_name.split('_')[-1])
                except:
                    window = 20
            
            returns = data['close'].pct_change()
            factor_values = returns.rolling(window).sum()
            
        elif 'mean_reversion' in factor_name:
            # 均值回归因子
            window = 20  # 默认窗口
            if '_' in factor_name:
                try:
                    window = int(factor_name.split('_')[-1])
                except:
                    window = 20
            
            returns = data['close'].pct_change()
            rolling_mean = returns.rolling(window).mean()
            rolling_std = returns.rolling(window).std()
            factor_values = (returns - rolling_mean) / rolling_std
            
        elif 'volatility' in factor_name:
            # 波动率因子
            window = 20  # 默认窗口
            if '_' in factor_name:
                try:
                    window = int(factor_name.split('_')[-1])
                except:
                    window = 20
            
            returns = data['close'].pct_change()
            factor_values = returns.rolling(window).std()
            
        elif 'volume' in factor_name:
            # 成交量因子
            window = 20  # 默认窗口
            if '_' in factor_name:
                try:
                    window = int(factor_name.split('_')[-1])
                except:
                    window = 20
            
            volume_ma = data['volume'].rolling(window).mean()
            factor_values = data['volume'] / volume_ma - 1
            
        else:
            # 默认返回价格变化率
            factor_values = data['close'].pct_change()
        
        # 处理NaN值
        factor_values = factor_values.fillna(0)
        
        return pd.Series(factor_values, index=data.index, name=factor_name)
        
    except Exception as e:
        print(f"❌ 计算因子 {factor_name} 时出错: {str(e)}")
        return pd.Series(index=data.index, dtype=float)
