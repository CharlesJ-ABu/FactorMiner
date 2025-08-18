"""
统一的ML特征构建管道
与训练与推理共用，避免特征不一致
"""

from typing import Optional
import pandas as pd
import numpy as np


def _resolve_column(data: pd.DataFrame, primary: str, alternatives: Optional[list] = None) -> str:
    """
    返回数据中可用的列名，优先使用primary，否则在alternatives中查找
    """
    if primary in data.columns:
        return primary
    if alternatives:
        for alt in alternatives:
            if alt in data.columns:
                return alt
    return primary  # 若不存在，返回primary，后续会产生NaN，便于统一处理


def build_ml_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    构建与训练一致的一组通用ML特征

    该实现与 `MLFactorBuilder._prepare_features` 对齐，
    以保证已训练模型（artifact中记录的feature_columns）可以正确推理。
    """
    features = pd.DataFrame(index=data.index)

    close_col = _resolve_column(data, 'close', ['S_DQ_CLOSE'])
    open_col = _resolve_column(data, 'open', ['S_DQ_OPEN'])
    high_col = _resolve_column(data, 'high', ['S_DQ_HIGH'])
    low_col = _resolve_column(data, 'low', ['S_DQ_LOW'])
    volume_col = _resolve_column(data, 'volume', ['S_DQ_VOLUME'])

    # 价格相关
    features['returns'] = data[close_col].pct_change()
    features['log_returns'] = np.log(data[close_col] / data[close_col].shift(1))
    features['high_low_ratio'] = data[high_col] / (data[low_col] + 1e-8)
    features['close_open_ratio'] = data[close_col] / (data[open_col] + 1e-8)

    # 移动平均与价格相对位置
    for window in [5, 10, 20, 50]:
        ma_col = f'ma_{window}'
        features[ma_col] = data[close_col].rolling(window=window).mean()
        features[f'ma_ratio_{window}'] = data[close_col] / (features[ma_col] + 1e-8)

    # 波动率
    for window in [5, 10, 20]:
        features[f'volatility_{window}'] = features['returns'].rolling(window=window).std()

    # 成交量
    features['volume_ratio'] = data[volume_col] / (data[volume_col].rolling(window=20).mean() + 1e-8)
    features['volume_ma_5'] = data[volume_col].rolling(window=5).mean()
    features['volume_ma_20'] = data[volume_col].rolling(window=20).mean()

    # 动量
    for period in [1, 5, 10, 20]:
        features[f'momentum_{period}'] = data[close_col] / data[close_col].shift(period) - 1

    # RSI
    for window in [14, 21]:
        delta = data[close_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / (loss + 1e-8)
        features[f'rsi_{window}'] = 100 - (100 / (1 + rs))

    # 价格区间位置
    for window in [20, 50]:
        min_price = data[close_col].rolling(window=window).min()
        max_price = data[close_col].rolling(window=window).max()
        features[f'price_position_{window}'] = (data[close_col] - min_price) / (max_price - min_price + 1e-8)

    return features


