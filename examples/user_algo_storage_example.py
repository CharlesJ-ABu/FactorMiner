#!/usr/bin/env python3
"""
用户算法中使用新存储API的示例

展示如何在user_algo文件中使用新的存储API来保存和加载模型、因子等
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from factor_miner.core.factor_storage import get_global_storage

# 算法元信息
ALGORITHM_INFO = {
    'name': '示例ML算法',
    'description': '展示如何在user_algo中使用新存储API',
    'category': 'ml_example',
    'version': '1.0.0',
    'author': 'FactorMiner Team'
}

def calculate_factors(data: pd.DataFrame) -> dict:
    """
    挖掘阶段：训练模型并生成所有因子
    系统会调用这个函数进行因子挖掘
    """
    print("🚀 开始示例ML算法挖掘...")
    
    # 1. 特征工程
    features = create_features(data)
    print(f"✅ 特征工程完成，特征数量: {features.shape[1]}")
    
    # 2. 准备训练数据
    X, y = prepare_training_data(features, data)
    print(f"✅ 训练数据准备完成，样本数量: {len(X)}")
    
    # 3. 训练模型
    model = train_model(X, y)
    
    # 4. 生成因子
    factors = {}
    factors['example_factor_1'] = predict_factor(data, model, 'factor_1')
    factors['example_factor_2'] = predict_factor(data, model, 'factor_2')
    
    print(f"✅ 因子生成完成，共生成 {len(factors)} 个因子")
    return factors

def calculate_single_factor(data: pd.DataFrame, factor_name: str) -> pd.Series:
    """
    计算阶段：计算单个因子
    系统会调用这个函数进行实时因子计算
    """
    print(f"🔄 计算因子: {factor_name}")
    
    # 使用新的存储API加载预训练模型
    storage = get_global_storage()
    
    # 加载模型
    model_data = storage.load_model(f"example_model_{factor_name}", "pkl")
    if model_data is None:
        print(f"❌ 模型文件不存在: example_model_{factor_name}.pkl")
        return pd.Series(index=data.index, dtype=float)
    
    # 加载预处理器
    scaler_data = storage.load_model(f"example_scaler_{factor_name}", "pkl")
    if scaler_data is None:
        print(f"❌ 预处理器文件不存在: example_scaler_{factor_name}.pkl")
        return pd.Series(index=data.index, dtype=float)
    
    model = joblib.loads(model_data)
    scaler = joblib.loads(scaler_data)
    
    # 计算因子
    return predict_factor(data, model, factor_name, scaler)

def create_features(data: pd.DataFrame) -> pd.DataFrame:
    """特征工程"""
    features = pd.DataFrame(index=data.index)
    
    # 价格特征
    features['price_change'] = data['close'].pct_change()
    features['high_low_ratio'] = data['high'] / data['low']
    features['close_open_ratio'] = data['close'] / data['open']
    
    # 技术指标
    features['ma_5'] = data['close'].rolling(5).mean()
    features['ma_20'] = data['close'].rolling(20).mean()
    features['volatility'] = data['close'].rolling(20).std()
    
    # 成交量特征
    features['volume_ma'] = data['volume'].rolling(10).mean()
    features['volume_ratio'] = data['volume'] / features['volume_ma']
    
    return features

def prepare_training_data(features: pd.DataFrame, data: pd.DataFrame) -> tuple:
    """准备训练数据"""
    # 创建目标变量（未来收益率）
    future_returns = data['close'].pct_change().shift(-5)  # 5天后的收益率
    
    # 对齐数据
    aligned_data = features.join(future_returns, how='inner')
    X = aligned_data.drop(columns=['future_return'])
    y = aligned_data['future_return']
    
    return X.fillna(0), y.fillna(0)

def train_model(X: pd.DataFrame, y: pd.Series) -> RandomForestRegressor:
    """训练ML模型"""
    print("🔄 训练ML模型...")
    
    # 数据预处理
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.fillna(0))
    
    # 训练模型
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_scaled, y)
    
    # 使用新的存储API保存模型
    storage = get_global_storage()
    
    # 保存主模型
    model_data = joblib.dumps(model)
    storage.save_model("example_model_factor_1", model_data, "pkl")
    storage.save_model("example_model_factor_2", model_data, "pkl")  # 示例：两个因子使用同一个模型
    
    # 保存预处理器
    scaler_data = joblib.dumps(scaler)
    storage.save_model("example_scaler_factor_1", scaler_data, "pkl")
    storage.save_model("example_scaler_factor_2", scaler_data, "pkl")
    
    print("✅ 模型已保存到存储系统")
    
    return model

def predict_factor(data: pd.DataFrame, model: RandomForestRegressor, 
                  factor_name: str, scaler: StandardScaler = None) -> pd.Series:
    """使用模型预测因子值"""
    # 特征工程
    features = create_features(data)
    
    # 预处理
    if scaler is not None:
        X_scaled = scaler.transform(features.fillna(0))
    else:
        X_scaled = features.fillna(0).values
    
    # 预测
    predictions = model.predict(X_scaled)
    return pd.Series(predictions, index=data.index)

def main():
    """主函数：演示如何使用新的存储API"""
    print("用户算法中使用新存储API的示例")
    print("=" * 50)
    
    # 创建示例数据
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=1000, freq='1H')
    data = pd.DataFrame({
        'open': np.random.randn(1000).cumsum() + 100,
        'high': np.random.randn(1000).cumsum() + 102,
        'low': np.random.randn(1000).cumsum() + 98,
        'close': np.random.randn(1000).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 1000)
    }, index=dates)
    
    try:
        # 演示挖掘阶段
        print("\n=== 挖掘阶段 ===")
        factors = calculate_factors(data)
        print(f"生成的因子: {list(factors.keys())}")
        
        # 演示计算阶段
        print("\n=== 计算阶段 ===")
        factor_1 = calculate_single_factor(data, 'factor_1')
        factor_2 = calculate_single_factor(data, 'factor_2')
        
        print(f"因子1长度: {len(factor_1)}")
        print(f"因子2长度: {len(factor_2)}")
        
        print("\n✅ 示例运行成功！")
        
    except Exception as e:
        print(f"❌ 运行示例时出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
