"""
统计因子构建器
包含滚动统计、分布特征等统计因子
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class StatisticalFactorBuilder:
    """
    统计因子构建器
    构建基于统计分析的因子
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化统计因子构建器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        
    def build_all_factors(self, data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """
        构建所有统计因子
        
        Args:
            data: 市场数据
            **kwargs: 其他参数
            
        Returns:
            统计因子字典
        """
        factors = {}
        
        # 获取正确的列名
        close_col = 'close' if 'close' in data.columns else 'S_DQ_CLOSE'
        volume_col = 'volume' if 'volume' in data.columns else 'S_DQ_VOLUME'
        
        # 构建滚动统计因子
        rolling_factors = self.build_rolling_statistics_factors(data, **kwargs)
        factors.update(rolling_factors)
        
        # 构建分布特征因子
        distribution_factors = self.build_distribution_factors(data, **kwargs)
        factors.update(distribution_factors)
        
        # 构建波动率因子
        volatility_factors = self.build_volatility_factors(data, **kwargs)
        factors.update(volatility_factors)
        
        # 构建相关性因子
        correlation_factors = self.build_correlation_factors(data, **kwargs)
        factors.update(correlation_factors)
        
        return factors
    
    def build_rolling_statistics_factors(self, data: pd.DataFrame, 
                                       windows: Optional[List[int]] = None,
                                       **kwargs) -> Dict[str, pd.Series]:
        """
        构建滚动统计因子
        
        Args:
            data: 市场数据
            windows: 滚动窗口列表
            **kwargs: 其他参数
            
        Returns:
            滚动统计因子字典
        """
        if windows is None:
            windows = [5, 10, 20, 50, 100, 200]
        
        factors = {}
        close_col = 'close' if 'close' in data.columns else 'S_DQ_CLOSE'
        returns = data[close_col].pct_change()
        
        for window in windows:
            # 滚动均值
            factors[f'rolling_mean_{window}'] = data[close_col].rolling(window=window).mean()
            
            # 滚动标准差
            factors[f'rolling_std_{window}'] = data[close_col].rolling(window=window).std()
            
            # 滚动偏度
            factors[f'skewness_{window}'] = data[close_col].rolling(window=window).skew()
            
            # 滚动峰度
            factors[f'kurtosis_{window}'] = data[close_col].rolling(window=window).kurt()
            
            # 滚动分位数
            factors[f'q25_{window}'] = data[close_col].rolling(window=window).quantile(0.25)
            factors[f'q75_{window}'] = data[close_col].rolling(window=window).quantile(0.75)
            factors[f'q90_{window}'] = data[close_col].rolling(window=window).quantile(0.90)
            
            # 滚动变异系数
            rolling_mean = data[close_col].rolling(window=window).mean()
            rolling_std = data[close_col].rolling(window=window).std()
            factors[f'cv_{window}'] = rolling_std / (rolling_mean + 1e-8)
            
            # 滚动Z-score
            factors[f'zscore_{window}'] = (data[close_col] - rolling_mean) / (rolling_std + 1e-8)
        
        return factors
    
    def build_distribution_factors(self, data: pd.DataFrame, 
                                 windows: Optional[List[int]] = None,
                                 **kwargs) -> Dict[str, pd.Series]:
        """
        构建分布特征因子
        
        Args:
            data: 市场数据
            windows: 滚动窗口列表
            **kwargs: 其他参数
            
        Returns:
            分布特征因子字典
        """
        if windows is None:
            windows = [20, 50, 100]
        
        factors = {}
        close_col = 'close' if 'close' in data.columns else 'S_DQ_CLOSE'
        returns = data[close_col].pct_change()
        
        for window in windows:
            # 价格位置
            rolling_min = data[close_col].rolling(window=window).min()
            rolling_max = data[close_col].rolling(window=window).max()
            factors[f'price_position_{window}'] = (data[close_col] - rolling_min) / (rolling_max - rolling_min + 1e-8)
            
            # 收益率分布特征
            returns_window = returns.rolling(window=window)
            factors[f'returns_skew_{window}'] = returns_window.skew()
            factors[f'returns_kurt_{window}'] = returns_window.kurt()
            
            # 价格动量
            factors[f'price_momentum_{window}'] = data[close_col] / data[close_col].shift(window) - 1
            
            # 价格反转
            factors[f'price_reversal_{window}'] = -factors[f'price_momentum_{window}']
        
        return factors
    
    def build_volatility_factors(self, data: pd.DataFrame,
                               windows: Optional[List[int]] = None,
                               **kwargs) -> Dict[str, pd.Series]:
        """
        构建波动率因子
        
        Args:
            data: 市场数据
            windows: 滚动窗口列表
            **kwargs: 其他参数
            
        Returns:
            波动率因子字典
        """
        if windows is None:
            windows = [5, 10, 20, 50]
        
        factors = {}
        close_col = 'close' if 'close' in data.columns else 'S_DQ_CLOSE'
        returns = data[close_col].pct_change()
        
        for window in windows:
            # 历史波动率
            factors[f'volatility_{window}'] = returns.rolling(window=window).std()
            
            # 波动率变化
            vol = returns.rolling(window=window).std()
            factors[f'volatility_change_{window}'] = vol / vol.shift(1) - 1
            
            # 波动率比率
            if window > 5:
                short_vol = returns.rolling(window=5).std()
                long_vol = returns.rolling(window=window).std()
                factors[f'volatility_ratio_{window}'] = short_vol / (long_vol + 1e-8)
            
            # 波动率状态
            vol_ma = vol.rolling(window=window).mean()
            factors[f'volatility_state_{window}'] = np.where(vol > vol_ma, 1, -1)
        
        return factors
    
    def build_correlation_factors(self, data: pd.DataFrame,
                                windows: Optional[List[int]] = None,
                                **kwargs) -> Dict[str, pd.Series]:
        """
        构建相关性因子
        
        Args:
            data: 市场数据
            windows: 滚动窗口列表
            **kwargs: 其他参数
            
        Returns:
            相关性因子字典
        """
        if windows is None:
            windows = [20, 50, 100]
        
        factors = {}
        close_col = 'close' if 'close' in data.columns else 'S_DQ_CLOSE'
        volume_col = 'volume' if 'volume' in data.columns else 'S_DQ_VOLUME'
        
        returns = data[close_col].pct_change()
        volume_change = data[volume_col].pct_change()
        
        for window in windows:
            # 价格-成交量相关性
            factors[f'price_volume_corr_{window}'] = returns.rolling(window=window).corr(volume_change)
            
            # 价格自相关性
            factors[f'price_autocorr_{window}'] = returns.rolling(window=window).apply(
                lambda x: x.autocorr() if len(x) > 1 else np.nan
            )
            
            # 成交量自相关性
            factors[f'volume_autocorr_{window}'] = volume_change.rolling(window=window).apply(
                lambda x: x.autocorr() if len(x) > 1 else np.nan
            )
        
        return factors
    
    def get_available_factors(self) -> List[str]:
        """
        获取可用的统计因子列表
        
        Returns:
            因子名称列表
        """
        return [
            'rolling_statistics',  # 滚动统计因子
            'distribution_features',  # 分布特征因子
            'volatility_factors',  # 波动率因子
            'correlation_factors'  # 相关性因子
        ] 