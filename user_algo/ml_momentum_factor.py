"""
ML动量因子算法示例
展示新的代码规范：用户负责完整的因子计算逻辑
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
from typing import Dict, Tuple
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from factor_miner.core.factor_storage import get_global_storage

# 算法元信息
ALGORITHM_INFO = {
    # 必填字段
    'id': 'ml_momentum_factor',
    'name': 'ML动量因子',
    'description': '基于机器学习的动量因子挖掘算法',
    'category': 'ml',
    'version': '1.0.0',
    'author': 'FactorMiner Team',
    
    # 可选字段
    'parameters': {
        'time_windows': {
            'type': 'list',
            'default': ['5d', '10d', '20d'],
            'description': '时间窗口列表',
            'required': True
        },
        'n_estimators': {
            'type': 'int',
            'default': 100,
            'description': '随机森林树的数量',
            'min': 10,
            'max': 1000,
            'required': False
        },
        'max_depth': {
            'type': 'int',
            'default': 10,
            'description': '树的最大深度',
            'min': 1,
            'max': 50,
            'required': False
        }
    },
    'requirements': ['pandas', 'numpy', 'sklearn', 'joblib'],
    'tags': ['ml', 'momentum', 'regression'],
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
    factor_info = {
        'ml_momentum_5d': {
            'name': 'ML动量因子5日',
            'description': '基于机器学习的5日动量因子',
            'type': 'ml',
            'parameters': {
                'time_window': '5d',
                'n_estimators': 100,
                'max_depth': 10
            }
        },
        'ml_momentum_10d': {
            'name': 'ML动量因子10日',
            'description': '基于机器学习的10日动量因子',
            'type': 'ml',
            'parameters': {
                'time_window': '10d',
                'n_estimators': 100,
                'max_depth': 10
            }
        },
        'ml_momentum_20d': {
            'name': 'ML动量因子20日',
            'description': '基于机器学习的20日动量因子',
            'type': 'ml',
            'parameters': {
                'time_window': '20d',
                'n_estimators': 100,
                'max_depth': 10
            }
        }
    }
    
    return factor_info.get(factor_name, {})

def calculate_factors(data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
    """
    挖掘阶段：训练模型并生成所有因子
    系统会调用这个函数进行因子挖掘
    
    Args:
        data: OHLCV数据，包含 'open', 'high', 'low', 'close', 'volume' 列
        **kwargs: 算法参数，来自ALGORITHM_INFO['parameters']
        
    Returns:
        Dict[str, pd.Series]: 因子名称到因子值的映射
    """
    print("🚀 开始ML动量因子挖掘...")
    
    # 1. 数据验证
    if not validate_data(data):
        raise ValueError("数据验证失败")
    
    print(f"✅ 数据验证通过，共 {len(data)} 条记录")
    
    # 2. 获取参数
    time_windows = kwargs.get('time_windows', ALGORITHM_INFO['parameters']['time_windows']['default'])
    n_estimators = kwargs.get('n_estimators', ALGORITHM_INFO['parameters']['n_estimators']['default'])
    max_depth = kwargs.get('max_depth', ALGORITHM_INFO['parameters']['max_depth']['default'])
    
    print(f"📊 使用参数: time_windows={time_windows}, n_estimators={n_estimators}, max_depth={max_depth}")
    
    # 3. 特征工程
    features = create_features(data)
    print(f"✅ 特征工程完成，特征数量: {features.shape[1]}")
    
    # 4. 准备训练数据
    X, y = prepare_training_data(features, data)
    print(f"✅ 训练数据准备完成，样本数量: {len(X)}")
    
    # 5. 训练不同时间窗口的模型
    factors = {}
    
    for window in time_windows:
        print(f"🔄 训练 {window} 动量模型...")
        
        # 准备该时间窗口的目标变量
        y_window = create_target_variable(data, window)
        X_aligned, y_aligned = align_data(features, y_window)
        
        if len(X_aligned) < 100:  # 数据太少，跳过
            print(f"⚠️  {window} 数据量不足，跳过")
            continue
        
        # 训练模型
        model = train_model(X_aligned, y_aligned, window)
        
        # 生成因子
        factor_name = f'ml_momentum_{window}'
        factor_values = predict_factor(X_aligned, model)
        factors[factor_name] = pd.Series(factor_values, index=X_aligned.index)
        
        print(f"✅ {window} 动量因子生成完成")
    
    print(f"🎉 ML动量因子挖掘完成，共生成 {len(factors)} 个因子")
    return factors

def calculate_single_factor(data: pd.DataFrame, factor_name: str, **kwargs) -> pd.Series:
    """
    计算阶段：计算单个因子
    系统会调用这个函数进行实时因子计算
    
    Args:
        data: OHLCV数据，包含 'open', 'high', 'low', 'close', 'volume' 列
        factor_name: 因子名称，如 'ml_momentum_5d'
        **kwargs: 算法参数，来自ALGORITHM_INFO['parameters']
        
    Returns:
        pd.Series: 因子值序列，索引与data相同
    """
    try:
        # 从因子名称中提取时间窗口
        if '_' in factor_name:
            window = factor_name.split('_')[-1]
        else:
            window = '5d'
        
        # 使用新的存储API加载预训练模型
        storage = get_global_storage()
        
        # 加载主模型
        model_data = storage.load_model(f"ml_momentum_{window}", "pkl")
        if model_data is None:
            print(f"❌ 模型文件不存在: ml_momentum_{window}.pkl")
            return pd.Series(index=data.index, dtype=float)
        
        # 加载预处理器
        scaler_data = storage.load_model(f"ml_momentum_{window}_scaler", "pkl")
        if scaler_data is None:
            print(f"❌ 预处理器文件不存在: ml_momentum_{window}_scaler.pkl")
            return pd.Series(index=data.index, dtype=float)
        
        model = joblib.loads(model_data)
        scaler = joblib.loads(scaler_data)
        
        # 特征工程
        features = create_features(data)
        
        # 预处理
        X_scaled = scaler.transform(features.fillna(0))
        
        # 预测
        predictions = model.predict(X_scaled)
        
        return pd.Series(predictions, index=data.index)
        
    except Exception as e:
        print(f"❌ 计算因子 {factor_name} 失败: {e}")
        return pd.Series(index=data.index, dtype=float)

def create_features(data: pd.DataFrame) -> pd.DataFrame:
    """特征工程：从OHLCV数据创建特征"""
    features = pd.DataFrame(index=data.index)
    
    # 价格特征
    features['price_change'] = data['close'].pct_change()
    features['high_low_ratio'] = data['high'] / data['low']
    features['close_open_ratio'] = data['close'] / data['open']
    features['body_size'] = abs(data['close'] - data['open']) / data['open']
    features['upper_shadow'] = (data['high'] - np.maximum(data['open'], data['close'])) / data['open']
    features['lower_shadow'] = (np.minimum(data['open'], data['close']) - data['low']) / data['open']
    
    # 移动平均特征
    for window in [5, 10, 20, 50]:
        features[f'ma_{window}'] = data['close'].rolling(window).mean()
        features[f'ma_ratio_{window}'] = data['close'] / features[f'ma_{window}']
        features[f'ma_slope_{window}'] = features[f'ma_{window}'].diff()
    
    # 波动率特征
    for window in [5, 10, 20]:
        features[f'volatility_{window}'] = data['close'].rolling(window).std()
        features[f'volatility_ratio_{window}'] = features[f'volatility_{window}'] / features[f'volatility_{window}'].rolling(50).mean()
    
    # 成交量特征
    features['volume_ma_5'] = data['volume'].rolling(5).mean()
    features['volume_ma_20'] = data['volume'].rolling(20).mean()
    features['volume_ratio'] = data['volume'] / features['volume_ma_20']
    features['volume_price_trend'] = (data['volume'] * data['close'].pct_change()).rolling(10).sum()
    
    # 技术指标特征
    features['rsi_14'] = calculate_rsi(data['close'], 14)
    features['rsi_21'] = calculate_rsi(data['close'], 21)
    features['macd'] = calculate_macd(data['close'])
    features['bollinger_position'] = calculate_bollinger_position(data['close'])
    
    # 时间特征
    features['hour'] = data.index.hour
    features['day_of_week'] = data.index.dayofweek
    features['is_weekend'] = (features['day_of_week'] >= 5).astype(int)
    
    # 滞后特征
    for lag in [1, 2, 3, 5]:
        features[f'price_change_lag_{lag}'] = features['price_change'].shift(lag)
        features[f'volume_ratio_lag_{lag}'] = features['volume_ratio'].shift(lag)
    
    return features

def prepare_training_data(features: pd.DataFrame, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """准备训练数据"""
    # 创建目标变量（未来5天收益率）
    future_returns = data['close'].pct_change().shift(-5)
    
    # 对齐数据
    aligned_data = features.join(future_returns, how='inner')
    X = aligned_data.drop(columns=['future_return']) if 'future_return' in aligned_data.columns else aligned_data
    y = aligned_data['future_return'] if 'future_return' in aligned_data.columns else pd.Series(index=aligned_data.index, dtype=float)
    
    return X.fillna(0), y.fillna(0)

def create_target_variable(data: pd.DataFrame, window: str) -> pd.Series:
    """创建目标变量"""
    if window == '5d':
        return data['close'].pct_change().shift(-5)
    elif window == '10d':
        return data['close'].pct_change().shift(-10)
    elif window == '20d':
        return data['close'].pct_change().shift(-20)
    else:
        return data['close'].pct_change().shift(-5)

def align_data(features: pd.DataFrame, target: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """对齐特征和目标变量"""
    aligned_data = features.join(target, how='inner')
    X = aligned_data.drop(columns=[target.name])
    y = aligned_data[target.name]
    return X, y

def train_model(X: pd.DataFrame, y: pd.Series, window: str) -> RandomForestRegressor:
    """训练ML模型"""
    from sklearn.model_selection import train_test_split
    
    # 数据预处理
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # 训练模型
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # 使用新的存储API保存模型
    storage = get_global_storage()
    
    # 保存主模型
    model_data = joblib.dumps(model)
    storage.save_model(f"ml_momentum_{window}", model_data, "pkl")
    
    # 保存预处理器
    scaler_data = joblib.dumps(scaler)
    storage.save_model(f"ml_momentum_{window}_scaler", scaler_data, "pkl")
    
    print(f"✅ 模型已保存: ml_momentum_{window}.pkl")
    
    return model

def predict_factor(X: pd.DataFrame, model: RandomForestRegressor) -> np.ndarray:
    """使用模型预测因子值"""
    return model.predict(X)

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """计算RSI指标"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """计算MACD指标"""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    return macd_line

def calculate_bollinger_position(prices: pd.Series, period: int = 20, std_dev: float = 2) -> pd.Series:
    """计算布林带位置"""
    ma = prices.rolling(period).mean()
    std = prices.rolling(period).std()
    upper_band = ma + (std * std_dev)
    lower_band = ma - (std * std_dev)
    return (prices - lower_band) / (upper_band - lower_band)
