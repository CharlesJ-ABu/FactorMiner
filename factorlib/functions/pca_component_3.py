
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def calculate(data: pd.DataFrame, **kwargs) -> pd.Series:
    """
    计算pca_component_3因子
    
    智能加载预训练的ML模型，使用pkl文件进行推理
    
    Args:
        data: 市场数据DataFrame，必须包含 OHLCV 列
        **kwargs: 其他参数
        
        Returns:
        因子值Series，预测结果
    """
    try:
        # 检查数据完整性
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if data is None or len(data) == 0:
            return pd.Series(dtype=float)
        
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            print(f"缺少必要的列: {missing_cols}")
            return pd.Series(index=data.index, dtype=float)
        
        # 智能查找模型文件
        model_paths = [
            Path(__file__).parent.parent / "models" / "pca_component_3.pkl",
            Path.cwd() / "factorlib" / "models" / "pca_component_3.pkl",
            Path(__file__).parent.parent.parent / "models" / "pca_component_3.pkl"
        ]
        
        artifact_file = None
        for path in model_paths:
            if path.exists():
                artifact_file = path
                break
        
        if artifact_file is None:
            print(f"未找到模型文件: {factor_name}.pkl")
            print(f"尝试过的路径: {[str(p) for p in model_paths]}")
            return pd.Series(index=data.index, dtype=float)
        
        # 加载预训练的模型
        with open(artifact_file, 'rb') as f:
            artifact = pickle.load(f)
        
        model = artifact.get("model")
        feature_columns = artifact.get("feature_columns", [])
        scaler = artifact.get("scaler")
        
        if model is None:
            print("模型文件损坏：无法加载模型")
            return pd.Series(index=data.index, dtype=float)
        
        # 构建特征（与训练时一致）
        features = _build_features(data)
        
        # 对齐所需列
        missing = [c for c in feature_columns if c not in features.columns]
        if missing:
            print(f"缺少特征列: {missing}")
            # 对缺失列补NaN，保持列齐全
            for c in missing:
                features[c] = np.nan
        
        X = features[feature_columns]
        
        # 清洗与标准化
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # 处理NaN值 - PCA不接受NaN值
        # 方法1：前向填充，然后后向填充
        X_cleaned = X.fillna(method='ffill').fillna(method='bfill')
        
        # 方法2：如果仍有NaN值，用0填充
        if X_cleaned.isna().any().any():
            print(f"警告：仍有NaN值，用0填充")
            X_cleaned = X_cleaned.fillna(0)
        
        # 方法3：如果某行全是NaN，删除该行
        valid_rows = ~X_cleaned.isna().all(axis=1)
        if not valid_rows.all():
            print(f"删除包含全NaN的行，从{len(X_cleaned)}行减少到{valid_rows.sum()}行")
            X_cleaned = X_cleaned[valid_rows]
            # 同时更新索引
            data = data.loc[valid_rows]
        
        if scaler is not None:
            X_scaled = scaler.transform(X_cleaned)
        else:
            X_scaled = X_cleaned.values
        
        # PCA转换（降维）
        # PCA不是预测模型，而是降维工具，使用transform方法
        try:
            X_transformed = model.transform(X_scaled)
            
            # 取第3个主成分（索引为2，因为从0开始）
            # 如果n_components < 3，则取最后一个
            component_index = min(2, model.n_components_ - 1)
            y_pred = X_transformed[:, component_index]
            
        except Exception as e:
            print(f"PCA转换失败: {e}")
            # 如果PCA转换失败，返回NaN序列
            return pd.Series(index=data.index, dtype=float)
        
        return pd.Series(y_pred, index=data.index)
        
    except Exception as e:
        print(f"计算pca_component_3因子时出错: {e}")
        import traceback
        traceback.print_exc()
    return pd.Series(index=data.index, dtype=float)

def _build_features(data: pd.DataFrame) -> pd.DataFrame:
    """构建与训练一致的特征"""
    features = pd.DataFrame(index=data.index)
    
    # 价格动量特征
    features['price_momentum_1'] = data['close'].pct_change(1)
    features['price_momentum_5'] = data['close'].pct_change(5)
    features['price_momentum_10'] = data['close'].pct_change(10)
    
    # 成交量特征
    features['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
    features['volume_momentum'] = data['volume'].pct_change(5)
    
    # 波动率特征
    features['volatility_10'] = data['close'].rolling(10).std() / data['close'].rolling(10).mean()
    features['volatility_20'] = data['close'].rolling(20).std() / data['close'].rolling(20).mean()
    
    # 趋势特征
    features['trend_5'] = (data['close'] - data['close'].shift(5)) / data['close'].shift(5)
    features['trend_10'] = (data['close'] - data['close'].shift(10)) / data['close'].shift(10)
    
    # 价格位置特征
    features['price_position_20'] = (data['close'] - data['low'].rolling(20).min()) / (data['high'].rolling(20).max() - data['low'].rolling(20).min())
    
    # 移动平均特征
    features['ma_5'] = data['close'] / data['close'].rolling(5).mean() - 1
    features['ma_10'] = data['close'] / data['close'].rolling(10).mean() - 1
    features['ma_20'] = data['close'] / data['close'].rolling(20).mean() - 1
    
    return features
