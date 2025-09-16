"""
随机森林因子挖掘算法
使用随机森林模型预测未来收益率
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def calculate_factors(data, **kwargs):
    """
    使用随机森林挖掘预测因子
    
    Args:
        data: DataFrame，包含 'open', 'high', 'low', 'close', 'volume' 列
        **kwargs: 算法参数
    
    Returns:
        Dict[str, pd.Series]: 因子名称 -> 因子值序列
    """
    factors = {}
    
    # 获取参数
    n_estimators = kwargs.get('n_estimators', 100)
    max_depth = kwargs.get('max_depth', 10)
    min_samples_split = kwargs.get('min_samples_split', 5)
    random_state = kwargs.get('random_state', 42)
    
    try:
        # 准备特征
        features = _build_features(data)
        target = _build_target(data)
        
        # 数据清理
        valid_idx = ~(features.isna().any(axis=1) | target.isna())
        features_clean = features.loc[valid_idx]
        target_clean = target.loc[valid_idx]
        
        if len(features_clean) < 50:
            print("⚠️ 数据量不足，无法训练随机森林模型")
            return factors
        
        # 标准化特征
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_clean)
        
        # 训练随机森林模型
        rf_model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            n_jobs=-1
        )
        rf_model.fit(features_scaled, target_clean)
        
        # 预测
        predictions = rf_model.predict(features_scaled)
        
        # 创建因子序列
        factor_series = pd.Series(index=data.index, dtype=float)
        factor_series.loc[valid_idx] = predictions
        factors['ml_random_forest'] = factor_series
        
        # 特征重要性因子
        feature_importance = rf_model.feature_importances_
        for i, importance in enumerate(feature_importance):
            if i < len(features.columns):
                factor_series = pd.Series(index=data.index, dtype=float)
                factor_series.loc[valid_idx] = importance
                factors[f'ml_rf_importance_{i}'] = factor_series
        
        print(f"✅ 随机森林因子挖掘完成，生成 {len(factors)} 个因子")
        
    except Exception as e:
        print(f"❌ 随机森林因子挖掘失败: {e}")
    
    return factors

def _build_features(data):
    """构建特征"""
    features = pd.DataFrame(index=data.index)
    
    # 价格特征
    features['price'] = data['close']
    features['price_change'] = data['close'].pct_change()
    features['price_change_2'] = data['close'].pct_change(2)
    features['price_change_5'] = data['close'].pct_change(5)
    
    # 技术指标特征
    features['sma_20'] = data['close'].rolling(window=20).mean()
    features['sma_50'] = data['close'].rolling(window=50).mean()
    features['rsi_14'] = _calculate_rsi(data['close'], 14)
    
    # 成交量特征
    features['volume'] = data['volume']
    features['volume_change'] = data['volume'].pct_change()
    features['volume_ma'] = data['volume'].rolling(window=20).mean()
    
    # 波动率特征
    returns = data['close'].pct_change()
    features['volatility'] = returns.rolling(window=20).std()
    
    # 趋势特征
    features['trend_5'] = (data['close'] - data['close'].shift(5)) / data['close'].shift(5)
    features['trend_10'] = (data['close'] - data['close'].shift(10)) / data['close'].shift(10)
    
    # 价格位置特征
    features['price_position_20'] = (data['close'] - data['low'].rolling(20).min()) / (data['high'].rolling(20).max() - data['low'].rolling(20).min())
    
    return features

def _build_target(data):
    """构建目标变量（未来收益率）"""
    return data['close'].pct_change().shift(-1)

def _calculate_rsi(prices, period):
    """计算RSI"""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

# 算法元信息
ALGORITHM_INFO = {
    'name': '随机森林因子挖掘',
    'description': '使用随机森林模型挖掘预测未来收益率的因子',
    'category': 'ml',
    'subcategory': 'ensemble',
    'parameters': {
        'n_estimators': {
            'type': 'int',
            'default': 100,
            'description': '决策树数量'
        },
        'max_depth': {
            'type': 'int',
            'default': 10,
            'description': '树的最大深度'
        },
        'min_samples_split': {
            'type': 'int',
            'default': 5,
            'description': '分裂所需的最小样本数'
        },
        'random_state': {
            'type': 'int',
            'default': 42,
            'description': '随机种子'
        }
    }
}
