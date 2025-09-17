"""
机器学习因子挖掘算法
使用机器学习方法进行因子发现、特征工程和模型训练
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import warnings
warnings.filterwarnings('ignore')

# 算法信息
ALGORITHM_INFO = {
    # 必填字段
    'id': 'ml_mining',
    'name': '机器学习因子挖掘',
    'description': '使用机器学习方法进行因子发现、特征工程和模型训练',
    'category': 'ml',
    'version': '1.0.0',
    'author': 'FactorMiner Team',
    
    # 可选字段
    'parameters': {
        'max_factors': {
            'type': 'int',
            'default': 30,
            'description': '最大因子数量',
            'min': 1,
            'max': 100,
            'required': False
        },
        'min_importance': {
            'type': 'float',
            'default': 0.01,
            'description': '最小特征重要性阈值',
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
        },
        'feature_selection_method': {
            'type': 'str',
            'default': 'mutual_info',
            'description': '特征选择方法',
            'choices': ['mutual_info', 'f_regression', 'pca'],
            'required': False
        },
        'n_features': {
            'type': 'int',
            'default': 50,
            'description': '选择的特征数量',
            'min': 1,
            'max': 200,
            'required': False
        },
        'test_size': {
            'type': 'float',
            'default': 0.3,
            'description': '测试集比例',
            'min': 0.1,
            'max': 0.5,
            'required': False
        },
        'random_state': {
            'type': 'int',
            'default': 42,
            'description': '随机种子',
            'required': False
        }
    },
    'requirements': ['pandas', 'numpy', 'sklearn'],
    'tags': ['ml', 'feature_selection', 'regression'],
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
    if 'rf_' in factor_name:
        return {
            'name': f'随机森林因子_{factor_name}',
            'description': f'基于随机森林的{factor_name}因子',
            'type': 'ml',
            'parameters': {'model': 'RandomForestRegressor'}
        }
    elif 'gb_' in factor_name:
        return {
            'name': f'梯度提升因子_{factor_name}',
            'description': f'基于梯度提升的{factor_name}因子',
            'type': 'ml',
            'parameters': {'model': 'GradientBoostingRegressor'}
        }
    elif 'ridge_' in factor_name:
        return {
            'name': f'岭回归因子_{factor_name}',
            'description': f'基于岭回归的{factor_name}因子',
            'type': 'ml',
            'parameters': {'model': 'Ridge'}
        }
    elif 'lasso_' in factor_name:
        return {
            'name': f'Lasso因子_{factor_name}',
            'description': f'基于Lasso的{factor_name}因子',
            'type': 'ml',
            'parameters': {'model': 'Lasso'}
        }
    elif 'pca_' in factor_name:
        return {
            'name': f'PCA因子_{factor_name}',
            'description': f'基于PCA的{factor_name}因子',
            'type': 'ml',
            'parameters': {'model': 'PCA'}
        }
    else:
        return {
            'name': f'ML因子_{factor_name}',
            'description': f'机器学习因子_{factor_name}',
            'type': 'ml',
            'parameters': {}
        }

def calculate_factors(data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
    """
    机器学习因子挖掘主函数
    
    Args:
        data: DataFrame，包含 'open', 'high', 'low', 'close', 'volume' 列
        **kwargs: 算法参数，来自ALGORITHM_INFO['parameters']
    
    Returns:
        Dict[str, pd.Series]: 挖掘出的因子字典
    """
    print("🤖 开始机器学习因子挖掘...")
    
    # 1. 数据验证
    if not validate_data(data):
        raise ValueError("数据验证失败")
    
    print(f"✅ 数据验证通过，共 {len(data)} 条记录")
    
    # 2. 获取参数
    max_factors = kwargs.get('max_factors', ALGORITHM_INFO['parameters']['max_factors']['default'])
    min_importance = kwargs.get('min_importance', 0.01)
    max_correlation = kwargs.get('max_correlation', 0.8)
    feature_selection_method = kwargs.get('feature_selection_method', 'mutual_info')
    n_features = kwargs.get('n_features', 50)
    test_size = kwargs.get('test_size', 0.3)
    random_state = kwargs.get('random_state', 42)
    
    # 1. 生成基础特征
    print("🔧 生成基础特征...")
    features = _generate_base_features(data)
    print(f"生成了 {len(features)} 个基础特征")
    
    # 2. 计算目标变量
    print("🎯 计算目标变量...")
    targets = _calculate_targets(data)
    
    # 3. 特征工程
    print("⚙️ 进行特征工程...")
    engineered_features = _engineer_features(features, targets)
    
    # 4. 特征选择
    print("🔍 进行特征选择...")
    selected_features = _select_features(
        engineered_features, targets, feature_selection_method, n_features
    )
    
    # 5. 训练机器学习模型
    print("🏋️ 训练机器学习模型...")
    ml_factors = _train_ml_models(selected_features, targets, random_state)
    
    # 6. 因子去重和筛选
    print("🔧 因子去重和筛选...")
    final_factors = _select_final_factors(
        ml_factors, max_factors, max_correlation, min_importance
    )
    
    print(f"✅ 机器学习因子挖掘完成，选择了 {len(final_factors)} 个因子")
    return final_factors

def _generate_base_features(data: pd.DataFrame) -> pd.DataFrame:
    """生成基础特征"""
    features = pd.DataFrame(index=data.index)
    
    # 获取正确的列名
    close_col = 'close' if 'close' in data.columns else 'S_DQ_CLOSE'
    volume_col = 'volume' if 'volume' in data.columns else 'S_DQ_VOLUME'
    high_col = 'high' if 'high' in data.columns else 'S_DQ_HIGH'
    low_col = 'low' if 'low' in data.columns else 'S_DQ_LOW'
    open_col = 'open' if 'open' in data.columns else 'S_DQ_OPEN'
    
    # 价格特征
    features['returns'] = data[close_col].pct_change()
    features['log_returns'] = np.log(data[close_col] / data[close_col].shift(1))
    features['high_low_ratio'] = data[high_col] / data[low_col]
    features['close_open_ratio'] = data[close_col] / data[open_col]
    
    # 成交量特征
    features['volume_ratio'] = data[volume_col] / data[volume_col].rolling(20).mean()
    features['volume_change'] = data[volume_col].pct_change()
    
    # 技术指标特征
    for window in [5, 10, 20, 50]:
        # 移动平均
        features[f'sma_{window}'] = data[close_col].rolling(window).mean()
        features[f'price_sma_ratio_{window}'] = data[close_col] / features[f'sma_{window}']
        
        # 波动率
        features[f'volatility_{window}'] = features['returns'].rolling(window).std()
        
        # 动量
        features[f'momentum_{window}'] = data[close_col] / data[close_col].shift(window) - 1
        
        # RSI
        delta = data[close_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
        rs = gain / loss
        features[f'rsi_{window}'] = 100 - (100 / (1 + rs))
        
        # 布林带
        bb_mean = data[close_col].rolling(window).mean()
        bb_std = data[close_col].rolling(window).std()
        features[f'bb_upper_{window}'] = bb_mean + 2 * bb_std
        features[f'bb_lower_{window}'] = bb_mean - 2 * bb_std
        features[f'bb_position_{window}'] = (data[close_col] - bb_mean) / (2 * bb_std)
        
        # 价格位置
        high_max = data[high_col].rolling(window).max()
        low_min = data[low_col].rolling(window).min()
        features[f'price_position_{window}'] = (data[close_col] - low_min) / (high_max - low_min)
    
    # 交互特征
    features['price_volume_interaction'] = features['returns'] * features['volume_change']
    features['volatility_volume_interaction'] = features['volatility_20'] * features['volume_ratio']
    
    return features

def _calculate_targets(data: pd.DataFrame) -> Dict[str, pd.Series]:
    """计算目标变量"""
    close_col = 'close' if 'close' in data.columns else 'S_DQ_CLOSE'
    returns = data[close_col].pct_change()
    
    targets = {}
    for period in [1, 5, 10, 20]:
        targets[f'future_return_{period}'] = returns.shift(-period)
        targets[f'future_volatility_{period}'] = returns.rolling(period).std().shift(-period)
    
    return targets

def _engineer_features(features: pd.DataFrame, targets: Dict[str, pd.Series]) -> pd.DataFrame:
    """特征工程"""
    engineered = features.copy()
    
    # 滞后特征
    for lag in [1, 2, 3, 5]:
        for col in ['returns', 'volume_ratio', 'volatility_20']:
            if col in engineered.columns:
                engineered[f'{col}_lag_{lag}'] = engineered[col].shift(lag)
    
    # 差分特征
    for col in ['returns', 'volume_ratio', 'volatility_20']:
        if col in engineered.columns:
            engineered[f'{col}_diff'] = engineered[col].diff()
    
    # 滚动统计特征
    for window in [5, 10, 20]:
        for col in ['returns', 'volume_ratio']:
            if col in engineered.columns:
                engineered[f'{col}_mean_{window}'] = engineered[col].rolling(window).mean()
                engineered[f'{col}_std_{window}'] = engineered[col].rolling(window).std()
                engineered[f'{col}_skew_{window}'] = engineered[col].rolling(window).skew()
                engineered[f'{col}_kurt_{window}'] = engineered[col].rolling(window).kurt()
    
    return engineered

def _select_features(features: pd.DataFrame, 
                    targets: Dict[str, pd.Series],
                    method: str,
                    n_features: int) -> pd.DataFrame:
    """特征选择"""
    # 使用第一个目标变量进行特征选择
    target_name = list(targets.keys())[0]
    target = targets[target_name]
    
    # 准备数据
    X = features.fillna(0)
    y = target.fillna(0)
    
    # 移除无效数据
    valid_idx = ~(X.isna().any(axis=1) | y.isna())
    X = X[valid_idx]
    y = y[valid_idx]
    
    if len(X) == 0:
        return features.iloc[:, :n_features]
    
    # 特征选择
    if method == 'mutual_info':
        selector = SelectKBest(score_func=mutual_info_regression, k=min(n_features, len(X.columns)))
    else:  # f_regression
        selector = SelectKBest(score_func=f_regression, k=min(n_features, len(X.columns)))
    
    try:
        X_selected = selector.fit_transform(X, y)
        selected_columns = X.columns[selector.get_support()]
        return features[selected_columns]
    except:
        # 如果特征选择失败，返回前n_features个特征
        return features.iloc[:, :n_features]

def _train_ml_models(features: pd.DataFrame, 
                    targets: Dict[str, pd.Series],
                    random_state: int) -> Dict[str, pd.Series]:
    """训练机器学习模型"""
    ml_factors = {}
    
    # 准备特征数据
    X = features.fillna(0)
    
    for target_name, target in targets.items():
        y = target.fillna(0)
        
        # 移除无效数据
        valid_idx = ~(X.isna().any(axis=1) | y.isna())
        X_valid = X[valid_idx]
        y_valid = y[valid_idx]
        
        if len(X_valid) == 0:
            continue
        
        # 标准化特征
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_valid)
        
        # 训练随机森林
        try:
            rf = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)
            rf.fit(X_scaled, y_valid)
            
            # 使用特征重要性作为因子
            feature_importance = pd.Series(rf.feature_importances_, index=features.columns)
            for i, (feature_name, importance) in enumerate(feature_importance.items()):
                if importance > 0.01:  # 只保留重要性较高的特征
                    factor_name = f'ml_rf_{target_name}_{feature_name}'
                    ml_factors[factor_name] = features[feature_name] * importance
        
        except Exception as e:
            print(f"训练随机森林失败: {e}")
            continue
        
        # 训练梯度提升
        try:
            gb = GradientBoostingRegressor(n_estimators=100, random_state=random_state)
            gb.fit(X_scaled, y_valid)
            
            # 使用特征重要性作为因子
            feature_importance = pd.Series(gb.feature_importances_, index=features.columns)
            for i, (feature_name, importance) in enumerate(feature_importance.items()):
                if importance > 0.01:
                    factor_name = f'ml_gb_{target_name}_{feature_name}'
                    ml_factors[factor_name] = features[feature_name] * importance
        
        except Exception as e:
            print(f"训练梯度提升失败: {e}")
            continue
    
    return ml_factors

def _select_final_factors(ml_factors: Dict[str, pd.Series],
                         max_factors: int,
                         max_correlation: float,
                         min_importance: float) -> Dict[str, pd.Series]:
    """选择最终因子"""
    # 按重要性排序
    factor_importance = {}
    for factor_name, factor_series in ml_factors.items():
        if not factor_series.isna().all():
            importance = abs(factor_series.mean()) if not factor_series.isna().all() else 0
            factor_importance[factor_name] = importance
    
    sorted_factors = sorted(factor_importance.items(), key=lambda x: x[1], reverse=True)
    
    selected_factors = {}
    selected_names = []
    
    for factor_name, importance in sorted_factors:
        if len(selected_factors) >= max_factors:
            break
            
        if importance < min_importance:
            continue
            
        # 检查相关性
        factor_series = ml_factors[factor_name]
        is_highly_correlated = False
        
        for selected_name in selected_names:
            selected_series = ml_factors[selected_name]
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
        
        # 特征工程
        features = _create_engineered_features(data)
        
        # 根据因子名称确定模型类型
        if 'rf_' in factor_name:
            model_type = 'RandomForestRegressor'
        elif 'gb_' in factor_name:
            model_type = 'GradientBoostingRegressor'
        elif 'ridge_' in factor_name:
            model_type = 'Ridge'
        elif 'lasso_' in factor_name:
            model_type = 'Lasso'
        elif 'pca_' in factor_name:
            model_type = 'PCA'
        else:
            print(f"❌ 未知的因子类型: {factor_name}")
            return pd.Series(index=data.index, dtype=float)
        
        # 这里应该加载预训练的模型，但由于是示例，我们返回随机值
        # 在实际应用中，应该从存储中加载对应的模型
        print(f"⚠️ 注意：{factor_name} 需要预训练模型，当前返回随机值")
        
        # 返回随机因子值作为示例
        np.random.seed(42)
        factor_values = np.random.normal(0, 1, len(data))
        
        return pd.Series(factor_values, index=data.index, name=factor_name)
        
    except Exception as e:
        print(f"❌ 计算因子 {factor_name} 时出错: {str(e)}")
        return pd.Series(index=data.index, dtype=float)
