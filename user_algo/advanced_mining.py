"""
高级因子挖掘算法
基于复杂特征工程和交互因子的挖掘方法
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# 算法信息
ALGORITHM_INFO = {
    # 必填字段
    'id': 'advanced_mining',
    'name': '高级因子挖掘',
    'description': '基于复杂特征工程和交互因子的高级挖掘算法',
    'category': 'advanced',
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
    'tags': ['advanced', 'factor_mining'],
    'created_at': '2024-01-01',
    'updated_at': '2024-01-01'
},
        'min_ic': {'type': 'float', 'default': 0.03, 'description': '最小IC阈值'},
        'min_ir': {'type': 'float', 'default': 0.08, 'description': '最小IR阈值'},
        'max_correlation': {'type': 'float', 'default': 0.75, 'description': '最大相关性阈值'},
        'interaction_windows': {'type': 'list', 'default': [5, 10, 20, 50], 'description': '交互因子窗口列表'},
        'pca_components': {'type': 'int', 'default': 10, 'description': 'PCA主成分数量'},
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
    print("🚀 开始高级因子挖掘...")
    
    # 获取参数
    max_factors = kwargs.get('max_factors', 40)
    min_ic = kwargs.get('min_ic', 0.03)
    min_ir = kwargs.get('min_ir', 0.08)
    max_correlation = kwargs.get('max_correlation', 0.75)
    interaction_windows = kwargs.get('interaction_windows', [5, 10, 20, 50])
    pca_components = kwargs.get('pca_components', 10)
    evaluation_periods = kwargs.get('evaluation_periods', [1, 3, 5, 10])
    
    # 1. 生成基础特征
    print("🔧 生成基础特征...")
    base_factors = _generate_base_factors(data, interaction_windows)
    print(f"生成了 {len(base_factors)} 个基础因子")
    
    # 2. 生成交互因子
    print("🔗 生成交互因子...")
    interaction_factors = _generate_interaction_factors(data, interaction_windows)
    print(f"生成了 {len(interaction_factors)} 个交互因子")
    
    # 3. 生成比率因子
    print("📊 生成比率因子...")
    ratio_factors = _generate_ratio_factors(data, interaction_windows)
    print(f"生成了 {len(ratio_factors)} 个比率因子")
    
    # 4. 生成PCA因子
    print("🎯 生成PCA因子...")
    pca_factors = _generate_pca_factors(data, pca_components)
    print(f"生成了 {len(pca_factors)} 个PCA因子")
    
    # 5. 合并所有因子
    all_factors = {**base_factors, **interaction_factors, **ratio_factors, **pca_factors}
    print(f"总共生成了 {len(all_factors)} 个候选因子")
    
    # 6. 计算未来收益率
    print("📈 计算未来收益率...")
    future_returns = _calculate_future_returns(data, evaluation_periods)
    
    # 7. 评估因子质量
    print("🎯 评估因子质量...")
    factor_scores = _evaluate_factors(all_factors, future_returns, min_ic, min_ir)
    
    # 8. 因子去重和筛选
    print("🔧 因子去重和筛选...")
    selected_factors = _select_factors(
        all_factors, factor_scores, max_factors, max_correlation
    )
    
    print(f"✅ 高级因子挖掘完成，选择了 {len(selected_factors)} 个因子")
    return selected_factors

def _generate_base_factors(data: pd.DataFrame, windows: List[int]) -> Dict[str, pd.Series]:
    """生成基础因子"""
    factors = {}
    
    # 获取正确的列名
    close_col = 'close' if 'close' in data.columns else 'S_DQ_CLOSE'
    volume_col = 'volume' if 'volume' in data.columns else 'S_DQ_VOLUME'
    high_col = 'high' if 'high' in data.columns else 'S_DQ_HIGH'
    low_col = 'low' if 'low' in data.columns else 'S_DQ_LOW'
    open_col = 'open' if 'open' in data.columns else 'S_DQ_OPEN'
    
    # 基础价格特征
    returns = data[close_col].pct_change()
    log_returns = np.log(data[close_col] / data[close_col].shift(1))
    
    for window in windows:
        # 价格动量
        factors[f'momentum_{window}'] = data[close_col] / data[close_col].shift(window) - 1
        
        # 价格位置
        high_max = data[high_col].rolling(window).max()
        low_min = data[low_col].rolling(window).min()
        factors[f'price_position_{window}'] = (data[close_col] - low_min) / (high_max - low_min + 1e-8)
        
        # 波动率
        factors[f'volatility_{window}'] = returns.rolling(window).std()
        
        # 成交量特征
        factors[f'volume_ratio_{window}'] = data[volume_col] / data[volume_col].rolling(window).mean()
        
        # 价格-成交量关系
        factors[f'price_volume_corr_{window}'] = returns.rolling(window).corr(
            data[volume_col].pct_change()
        )
    
    return factors

def _generate_interaction_factors(data: pd.DataFrame, windows: List[int]) -> Dict[str, pd.Series]:
    """生成交互因子"""
    factors = {}
    
    # 获取正确的列名
    close_col = 'close' if 'close' in data.columns else 'S_DQ_CLOSE'
    volume_col = 'volume' if 'volume' in data.columns else 'S_DQ_VOLUME'
    high_col = 'high' if 'high' in data.columns else 'S_DQ_HIGH'
    low_col = 'low' if 'low' in data.columns else 'S_DQ_LOW'
    open_col = 'open' if 'open' in data.columns else 'S_DQ_OPEN'
    
    returns = data[close_col].pct_change()
    volume_change = data[volume_col].pct_change()
    
    for window in windows:
        # 价格-成交量交互
        factors[f'price_volume_interaction_{window}'] = (
            data[close_col] * data[volume_col]
        ) / data[close_col].rolling(window).mean()
        
        # 波动率-成交量交互
        volatility = returns.rolling(window).std()
        factors[f'vol_volume_interaction_{window}'] = volatility * data[volume_col]
        
        # 价格位置-成交量交互
        price_position = (data[close_col] - data[close_col].rolling(window).min()) / \
                        (data[close_col].rolling(window).max() - data[close_col].rolling(window).min() + 1e-8)
        factors[f'position_volume_interaction_{window}'] = price_position * data[volume_col]
        
        # 动量-成交量交互
        momentum = data[close_col] / data[close_col].shift(window) - 1
        factors[f'momentum_volume_interaction_{window}'] = momentum * data[volume_col]
        
        # 价格-波动率交互
        factors[f'price_vol_interaction_{window}'] = data[close_col] * volatility
        
        # 收益率-成交量交互
        factors[f'returns_volume_interaction_{window}'] = returns * volume_change
    
    return factors

def _generate_ratio_factors(data: pd.DataFrame, windows: List[int]) -> Dict[str, pd.Series]:
    """生成比率因子"""
    factors = {}
    
    # 获取正确的列名
    close_col = 'close' if 'close' in data.columns else 'S_DQ_CLOSE'
    volume_col = 'volume' if 'volume' in data.columns else 'S_DQ_VOLUME'
    high_col = 'high' if 'high' in data.columns else 'S_DQ_HIGH'
    low_col = 'low' if 'low' in data.columns else 'S_DQ_LOW'
    open_col = 'open' if 'open' in data.columns else 'S_DQ_OPEN'
    
    for window in windows:
        # 价格比率
        factors[f'high_low_ratio_{window}'] = data[high_col] / (data[low_col] + 1e-8)
        factors[f'close_open_ratio_{window}'] = data[close_col] / (data[open_col] + 1e-8)
        
        # 成交量比率
        volume_ma = data[volume_col].rolling(window).mean()
        factors[f'volume_ma_ratio_{window}'] = data[volume_col] / (volume_ma + 1e-8)
        
        # 价格动量比率
        short_momentum = data[close_col] / data[close_col].shift(window//2) - 1
        long_momentum = data[close_col] / data[close_col].shift(window) - 1
        factors[f'momentum_ratio_{window}'] = short_momentum / (long_momentum + 1e-8)
        
        # 波动率比率
        short_vol = data[close_col].pct_change().rolling(window//2).std()
        long_vol = data[close_col].pct_change().rolling(window).std()
        factors[f'volatility_ratio_{window}'] = short_vol / (long_vol + 1e-8)
        
        # 价格位置比率
        price_position_short = (data[close_col] - data[close_col].rolling(window//2).min()) / \
                              (data[close_col].rolling(window//2).max() - data[close_col].rolling(window//2).min() + 1e-8)
        price_position_long = (data[close_col] - data[close_col].rolling(window).min()) / \
                             (data[close_col].rolling(window).max() - data[close_col].rolling(window).min() + 1e-8)
        factors[f'position_ratio_{window}'] = price_position_short / (price_position_long + 1e-8)
    
    return factors

def _generate_pca_factors(data: pd.DataFrame, n_components: int) -> Dict[str, pd.Series]:
    """生成PCA因子"""
    factors = {}
    
    # 获取正确的列名
    close_col = 'close' if 'close' in data.columns else 'S_DQ_CLOSE'
    volume_col = 'volume' if 'volume' in data.columns else 'S_DQ_VOLUME'
    high_col = 'high' if 'high' in data.columns else 'S_DQ_HIGH'
    low_col = 'low' if 'low' in data.columns else 'S_DQ_LOW'
    open_col = 'open' if 'open' in data.columns else 'S_DQ_OPEN'
    
    # 准备特征矩阵
    features = []
    feature_names = []
    
    # 价格特征
    returns = data[close_col].pct_change()
    features.append(returns)
    feature_names.append('returns')
    
    # 技术指标特征
    for window in [5, 10, 20, 50]:
        # 移动平均
        sma = data[close_col].rolling(window).mean()
        features.append(sma)
        feature_names.append(f'sma_{window}')
        
        # 波动率
        vol = returns.rolling(window).std()
        features.append(vol)
        feature_names.append(f'vol_{window}')
        
        # 动量
        momentum = data[close_col] / data[close_col].shift(window) - 1
        features.append(momentum)
        feature_names.append(f'momentum_{window}')
        
        # 成交量比率
        volume_ratio = data[volume_col] / data[volume_col].rolling(window).mean()
        features.append(volume_ratio)
        feature_names.append(f'volume_ratio_{window}')
    
    # 组合特征矩阵
    feature_matrix = pd.DataFrame(dict(zip(feature_names, features))).fillna(0)
    
    if len(feature_matrix) == 0 or feature_matrix.isna().all().all():
        return factors
    
    # 标准化
    scaler = StandardScaler()
    feature_matrix_scaled = scaler.fit_transform(feature_matrix)
    
    # PCA
    try:
        pca = PCA(n_components=min(n_components, len(feature_names)))
        pca_result = pca.fit_transform(feature_matrix_scaled)
        
        # 将PCA结果转换为因子
        for i in range(pca_result.shape[1]):
            factors[f'pca_component_{i+1}'] = pd.Series(
                pca_result[:, i], 
                index=data.index
            )
            
        print(f"PCA解释方差比: {pca.explained_variance_ratio_}")
        
    except Exception as e:
        print(f"PCA计算失败: {e}")
    
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
            factor_scores['score'] = ic_score * 0.6 + ir_score * 0.4
        
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
