"""
高级因子构建模块
包含更复杂的特征工程和因子构建方法
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Callable
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class AdvancedFactorBuilder:
    """
    高级因子构建器
    提供更复杂的特征工程和因子构建方法
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化高级因子构建器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        
    def build_interaction_factors(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        构建交互因子
        
        Args:
            data: 市场数据
            **kwargs: 其他参数
            
        Returns:
            交互因子DataFrame
        """
        factors = {}
        
        # 获取正确的列名
        close_col = 'close' if 'close' in data.columns else 'S_DQ_CLOSE'
        volume_col = 'volume' if 'volume' in data.columns else 'S_DQ_VOLUME'
        high_col = 'high' if 'high' in data.columns else 'S_DQ_HIGH'
        low_col = 'low' if 'low' in data.columns else 'S_DQ_LOW'
        open_col = 'open' if 'open' in data.columns else 'S_DQ_OPEN'
        
        # 价格-成交量交互因子
        factors['price_volume_interaction'] = data[close_col] * data[volume_col]
        factors['price_volume_ratio'] = data[close_col] / (data[volume_col] + 1)
        
        # 波动率-成交量交互
        returns = data[close_col].pct_change()
        volatility = returns.rolling(window=20).std()
        factors['vol_volume_interaction'] = volatility * data[volume_col]
        
        # 价格位置-成交量交互
        price_position = (data[close_col] - data[close_col].rolling(window=20).min()) / \
                        (data[close_col].rolling(window=20).max() - data[close_col].rolling(window=20).min())
        factors['position_volume_interaction'] = price_position * data[volume_col]
        
        return pd.DataFrame(factors, index=data.index)
    
    def build_ratio_factors(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        构建比率因子
        
        Args:
            data: 市场数据
            **kwargs: 其他参数
            
        Returns:
            比率因子DataFrame
        """
        factors = {}
        
        # 获取正确的列名
        close_col = 'close' if 'close' in data.columns else 'S_DQ_CLOSE'
        volume_col = 'volume' if 'volume' in data.columns else 'S_DQ_VOLUME'
        high_col = 'high' if 'high' in data.columns else 'S_DQ_HIGH'
        low_col = 'low' if 'low' in data.columns else 'S_DQ_LOW'
        open_col = 'open' if 'open' in data.columns else 'S_DQ_OPEN'
        
        # 价格比率
        factors['high_low_ratio'] = data[high_col] / (data[low_col] + 1e-8)
        factors['close_open_ratio'] = data[close_col] / (data[open_col] + 1e-8)
        
        # 成交量比率
        volume_ma = data[volume_col].rolling(window=20).mean()
        factors['volume_ma_ratio'] = data[volume_col] / (volume_ma + 1e-8)
        
        # 价格动量比率
        factors['momentum_5_20_ratio'] = (data[close_col] / data[close_col].shift(5)) / \
                                       (data[close_col] / data[close_col].shift(20))
        
        # 波动率比率
        returns = data[close_col].pct_change()
        vol_5 = returns.rolling(window=5).std()
        vol_20 = returns.rolling(window=20).std()
        factors['volatility_ratio'] = vol_5 / (vol_20 + 1e-8)
        
        return pd.DataFrame(factors, index=data.index)
    
    def build_lag_factors(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        构建滞后因子
        
        Args:
            data: 市场数据
            **kwargs: 其他参数
            
        Returns:
            滞后因子DataFrame
        """
        factors = {}
        lags = kwargs.get('lags', [1, 2, 3, 5, 10])
        
        # 获取正确的列名
        close_col = 'close' if 'close' in data.columns else 'S_DQ_CLOSE'
        volume_col = 'volume' if 'volume' in data.columns else 'S_DQ_VOLUME'
        
        for lag in lags:
            # 价格滞后
            factors[f'price_lag_{lag}'] = data[close_col].shift(lag)
            
            # 成交量滞后
            factors[f'volume_lag_{lag}'] = data[volume_col].shift(lag)
            
            # 收益率滞后
            returns = data[close_col].pct_change()
            factors[f'returns_lag_{lag}'] = returns.shift(lag)
            
            # 价格变化滞后
            factors[f'price_change_lag_{lag}'] = data[close_col].diff(lag)
        
        return pd.DataFrame(factors, index=data.index)
    
    def build_rolling_statistics_factors(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        构建滚动统计因子
        
        Args:
            data: 市场数据
            **kwargs: 其他参数
            
        Returns:
            滚动统计因子DataFrame
        """
        factors = {}
        windows = kwargs.get('windows', [5, 10, 20, 50])
        
        # 获取正确的列名
        close_col = 'close' if 'close' in data.columns else 'S_DQ_CLOSE'
        
        for window in windows:
            # 滚动偏度
            returns = data[close_col].pct_change()
            factors[f'skewness_{window}'] = returns.rolling(window=window).skew()
            
            # 滚动峰度
            factors[f'kurtosis_{window}'] = returns.rolling(window=window).kurt()
            
            # 滚动分位数
            factors[f'q25_{window}'] = returns.rolling(window=window).quantile(0.25)
            factors[f'q75_{window}'] = returns.rolling(window=window).quantile(0.75)
            factors[f'q90_{window}'] = returns.rolling(window=window).quantile(0.90)
            
            # 滚动变异系数
            mean = returns.rolling(window=window).mean()
            std = returns.rolling(window=window).std()
            factors[f'cv_{window}'] = std / (np.abs(mean) + 1e-8)
        
        return pd.DataFrame(factors, index=data.index)
    
    def build_technical_pattern_factors(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        构建技术形态因子
        
        Args:
            data: 市场数据
            **kwargs: 其他参数
            
        Returns:
            技术形态因子DataFrame
        """
        factors = {}
        
        # 获取正确的列名
        close_col = 'close' if 'close' in data.columns else 'S_DQ_CLOSE'
        volume_col = 'volume' if 'volume' in data.columns else 'S_DQ_VOLUME'
        high_col = 'high' if 'high' in data.columns else 'S_DQ_HIGH'
        low_col = 'low' if 'low' in data.columns else 'S_DQ_LOW'
        open_col = 'open' if 'open' in data.columns else 'S_DQ_OPEN'
        
        # 锤子线形态
        body = np.abs(data[close_col] - data[open_col])
        lower_shadow = np.minimum(data[open_col], data[close_col]) - data[low_col]
        upper_shadow = data[high_col] - np.maximum(data[open_col], data[close_col])
        
        factors['hammer_pattern'] = np.where(
            (lower_shadow > 2 * body) & (upper_shadow < body),
            1, 0
        )
        
        # 十字星形态
        factors['doji_pattern'] = np.where(
            body < 0.1 * (data[high_col] - data[low_col]),
            1, 0
        )
        
        # 吞没形态
        factors['engulfing_bullish'] = np.where(
            (data[close_col] > data[open_col]) &  # 当前为阳线
            (data[close_col].shift(1) < data[open_col].shift(1)) &  # 前一根为阴线
            (data[close_col] > data[open_col].shift(1)) &  # 当前收盘价高于前一根开盘价
            (data[open_col] < data[close_col].shift(1)),  # 当前开盘价低于前一根收盘价
            1, 0
        )
        
        # 支撑阻力位
        window = kwargs.get('support_resistance_window', 20)
        factors['support_level'] = data[low_col].rolling(window=window).min()
        factors['resistance_level'] = data[high_col].rolling(window=window).max()
        factors['price_to_support'] = (data[close_col] - factors['support_level']) / \
                                    (factors['resistance_level'] - factors['support_level'] + 1e-8)
        
        return pd.DataFrame(factors, index=data.index)
    
    def build_market_regime_factors(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        构建市场状态因子
        
        Args:
            data: 市场数据
            **kwargs: 其他参数
            
        Returns:
            市场状态因子DataFrame
        """
        factors = {}
        
        # 获取正确的列名
        close_col = 'close' if 'close' in data.columns else 'S_DQ_CLOSE'
        volume_col = 'volume' if 'volume' in data.columns else 'S_DQ_VOLUME'
        
        # 趋势状态
        ma_short = data[close_col].rolling(window=5).mean()
        ma_long = data[close_col].rolling(window=20).mean()
        factors['trend_state'] = np.where(ma_short > ma_long, 1, -1)
        
        # 波动率状态
        returns = data[close_col].pct_change()
        vol_short = returns.rolling(window=5).std()
        vol_long = returns.rolling(window=20).std()
        factors['volatility_state'] = np.where(vol_short > vol_long, 1, -1)
        
        # 成交量状态
        volume_ma = data[volume_col].rolling(window=20).mean()
        factors['volume_state'] = np.where(data[volume_col] > volume_ma, 1, -1)
        
        # 价格位置状态
        price_position = (data[close_col] - data[close_col].rolling(window=20).min()) / \
                        (data[close_col].rolling(window=20).max() - data[close_col].rolling(window=20).min())
        factors['price_position_state'] = np.where(price_position > 0.8, 1, 
                                                 np.where(price_position < 0.2, -1, 0))
        
        return pd.DataFrame(factors, index=data.index)
    
    def build_adaptive_factors(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        构建自适应因子
        
        Args:
            data: 市场数据
            **kwargs: 其他参数
            
        Returns:
            自适应因子DataFrame
        """
        factors = {}
        
        # 获取正确的列名
        close_col = 'close' if 'close' in data.columns else 'S_DQ_CLOSE'
        
        # 自适应移动平均
        returns = data[close_col].pct_change()
        volatility = returns.rolling(window=20).std()
        
        # 根据波动率调整窗口
        # 处理无穷大和NaN值
        volatility_clean = volatility.replace([np.inf, -np.inf], np.nan).fillna(volatility.mean())
        adaptive_window = np.maximum(5, np.minimum(50, (1 / (volatility_clean + 1e-8)).astype(int)))
        
        # 自适应移动平均
        adaptive_ma = pd.Series(index=data.index, dtype=float)
        for i in tqdm(range(len(data)), desc="计算自适应移动平均", leave=False):
            if i >= 5:
                window = int(adaptive_window.iloc[i])
                adaptive_ma.iloc[i] = data[close_col].iloc[max(0, i-window):i+1].mean()
        
        factors['adaptive_ma'] = adaptive_ma
        factors['adaptive_ma_ratio'] = data[close_col] / (adaptive_ma + 1e-8)
        
        # 自适应波动率
        factors['adaptive_volatility'] = volatility * adaptive_window / 20
        
        return pd.DataFrame(factors, index=data.index)
    
    def build_all_advanced_factors(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        构建所有高级因子
        
        Args:
            data: 市场数据
            **kwargs: 其他参数
            
        Returns:
            所有高级因子DataFrame
        """
        print("构建高级因子...")
        
        all_factors = pd.DataFrame(index=data.index)
        
        # 构建各类因子
        factor_types = [
            'interaction_factors',
            'ratio_factors', 
            'lag_factors',
            'rolling_statistics_factors',
            'technical_pattern_factors',
            'market_regime_factors',
            'adaptive_factors'
        ]
        
        # 使用tqdm显示进度条
        for factor_type in tqdm(factor_types, desc="构建高级因子", unit="类型"):
            try:
                if factor_type == 'interaction_factors':
                    factors = self.build_interaction_factors(data, **kwargs)
                elif factor_type == 'ratio_factors':
                    factors = self.build_ratio_factors(data, **kwargs)
                elif factor_type == 'lag_factors':
                    factors = self.build_lag_factors(data, **kwargs)
                elif factor_type == 'rolling_statistics_factors':
                    factors = self.build_rolling_statistics_factors(data, **kwargs)
                elif factor_type == 'technical_pattern_factors':
                    factors = self.build_technical_pattern_factors(data, **kwargs)
                elif factor_type == 'market_regime_factors':
                    factors = self.build_market_regime_factors(data, **kwargs)
                elif factor_type == 'adaptive_factors':
                    factors = self.build_adaptive_factors(data, **kwargs)
                
                all_factors = pd.concat([all_factors, factors], axis=1)
                tqdm.write(f"✓ 成功构建 {factor_type}: {len(factors.columns)} 个因子")
                
            except Exception as e:
                tqdm.write(f"✗ 构建 {factor_type} 失败: {e}")
                continue
        
        # 处理异常值
        all_factors = all_factors.replace([np.inf, -np.inf], np.nan)
        all_factors = all_factors.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        print(f"总共构建了 {len(all_factors.columns)} 个高级因子")
        
        return all_factors 