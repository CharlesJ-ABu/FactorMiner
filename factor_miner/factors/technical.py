"""
因子计算模块
包含各种技术指标和自定义因子的计算
"""

import pandas as pd
import numpy as np
import ta
from typing import Dict, List, Optional, Union, Callable
import warnings
warnings.filterwarnings('ignore')


class FactorCalculator:
    """
    因子计算器
    计算各种技术指标和自定义因子
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化因子计算器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.factor_registry = self._register_factors()
        
    def _register_factors(self) -> Dict[str, Callable]:
        """
        注册所有可用的因子计算函数
        """
        return {
            # 趋势因子
            'ma_cross': self._calculate_ma_cross,
            'price_position': self._calculate_price_position,
            'trend_strength': self._calculate_trend_strength,
            
            # 动量因子
            'momentum': self._calculate_momentum,
            'rsi': self._calculate_rsi,
            'macd': self._calculate_macd,
            'stoch': self._calculate_stochastic,
            
            # 波动率因子
            'volatility': self._calculate_volatility,
            'atr': self._calculate_atr,
            'bollinger_bands': self._calculate_bollinger_bands,
            
            # 成交量因子
            'volume_indicators': self._calculate_volume_indicators,
            'obv': self._calculate_obv,
            'vwap': self._calculate_vwap,
            
            # 价格形态因子
            'price_patterns': self._calculate_price_patterns,
            'support_resistance': self._calculate_support_resistance,
            
            # 统计因子
            'statistical_factors': self._calculate_statistical_factors,
            'z_score': self._calculate_z_score,
            'percentile_rank': self._calculate_percentile_rank,
            
            # 自定义因子
            'custom_factors': self._calculate_custom_factors
        }
    
    def calculate_all_factors(self,
                             data: pd.DataFrame,
                             factor_types: Optional[List[str]] = None,
                             **kwargs) -> pd.DataFrame:
        """
        计算所有因子
        
        Args:
            data: 市场数据
            factor_types: 因子类型列表，如果为None则计算所有因子
            **kwargs: 其他参数
            
        Returns:
            因子数据DataFrame
        """
        if factor_types is None:
            factor_types = list(self.factor_registry.keys())
        
        factors_df = pd.DataFrame(index=data.index)
        
        for factor_type in factor_types:
            if factor_type in self.factor_registry:
                try:
                    factor_data = self.factor_registry[factor_type](data, **kwargs)
                    if isinstance(factor_data, pd.DataFrame):
                        factors_df = pd.concat([factors_df, factor_data], axis=1)
                    elif isinstance(factor_data, pd.Series):
                        factors_df[factor_type] = factor_data
                except Exception as e:
                    print(f"计算因子 {factor_type} 失败: {e}")
                    continue
        
        return factors_df
    
    def _calculate_ma_cross(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算移动平均交叉因子
        """
        periods = kwargs.get('periods', [5, 10, 20, 50])
        factors = {}
        
        for i, short_period in enumerate(periods[:-1]):
            for long_period in periods[i+1:]:
                ma_short = data['close'].rolling(window=short_period).mean()
                ma_long = data['close'].rolling(window=long_period).mean()
                
                # 交叉信号
                cross_signal = np.where(ma_short > ma_long, 1, -1)
                factors[f'ma_cross_{short_period}_{long_period}'] = cross_signal
                
                # 距离比率
                distance_ratio = (ma_short - ma_long) / ma_long
                factors[f'ma_distance_{short_period}_{long_period}'] = distance_ratio
        
        return pd.DataFrame(factors, index=data.index)
    
    def _calculate_price_position(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算价格位置因子
        """
        periods = kwargs.get('periods', [20, 50, 100])
        factors = {}
        
        for period in periods:
            high = data['high'].rolling(window=period).max()
            low = data['low'].rolling(window=period).min()
            close = data['close']
            
            # 价格位置
            position = (close - low) / (high - low)
            factors[f'price_position_{period}'] = position
            
            # 价格偏离度
            ma = data['close'].rolling(window=period).mean()
            deviation = (close - ma) / ma
            factors[f'price_deviation_{period}'] = deviation
        
        return pd.DataFrame(factors, index=data.index)
    
    def _calculate_trend_strength(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算趋势强度因子
        """
        periods = kwargs.get('periods', [20, 50])
        factors = {}
        
        for period in periods:
            # ADX指标
            adx = ta.trend.ADXIndicator(data['high'], data['low'], data['close'], window=period)
            factors[f'adx_{period}'] = adx.adx()
            factors[f'di_plus_{period}'] = adx.adx_pos()
            factors[f'di_minus_{period}'] = adx.adx_neg()
            
            # 趋势一致性
            ma = data['close'].rolling(window=period).mean()
            trend_consistency = np.where(data['close'] > ma, 1, -1)
            factors[f'trend_consistency_{period}'] = trend_consistency
        
        return pd.DataFrame(factors, index=data.index)
    
    def _calculate_momentum(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算动量因子
        """
        periods = kwargs.get('periods', [5, 10, 20, 50])
        factors = {}
        
        for period in periods:
            # 价格动量
            momentum = data['close'] / data['close'].shift(period) - 1
            factors[f'momentum_{period}'] = momentum
            
            # 收益率动量
            returns = data['close'].pct_change()
            returns_momentum = returns.rolling(window=period).mean()
            factors[f'returns_momentum_{period}'] = returns_momentum
            
            # 相对强弱
            gains = returns.where(returns > 0, 0)
            losses = -returns.where(returns < 0, 0)
            avg_gains = gains.rolling(window=period).mean()
            avg_losses = losses.rolling(window=period).mean()
            rs = avg_gains / avg_losses
            factors[f'rs_{period}'] = rs
        
        return pd.DataFrame(factors, index=data.index)
    
    def _calculate_rsi(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算RSI因子
        """
        periods = kwargs.get('periods', [14, 21])
        factors = {}
        
        for period in periods:
            rsi = ta.momentum.RSIIndicator(data['close'], window=period)
            factors[f'rsi_{period}'] = rsi.rsi()
            
            # RSI衍生指标
            rsi_values = factors[f'rsi_{period}']
            factors[f'rsi_overbought_{period}'] = np.where(rsi_values > 70, 1, 0)
            factors[f'rsi_oversold_{period}'] = np.where(rsi_values < 30, 1, 0)
        
        return pd.DataFrame(factors, index=data.index)
    
    def _calculate_macd(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算MACD因子
        """
        fast = kwargs.get('fast', 12)
        slow = kwargs.get('slow', 26)
        signal = kwargs.get('signal', 9)
        
        macd = ta.trend.MACD(data['close'], window_fast=fast, window_slow=slow, window_sign=signal)
        
        factors = {
            'macd': macd.macd(),
            'macd_signal': macd.macd_signal(),
            'macd_diff': macd.macd_diff(),
            'macd_cross': np.where(macd.macd() > macd.macd_signal(), 1, -1)
        }
        
        return pd.DataFrame(factors, index=data.index)
    
    def _calculate_stochastic(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算随机指标因子
        """
        k_period = kwargs.get('k_period', 14)
        d_period = kwargs.get('d_period', 3)
        
        stoch = ta.momentum.StochasticOscillator(data['high'], data['low'], data['close'], 
                                                window=k_period, smooth_window=d_period)
        
        factors = {
            'stoch_k': stoch.stoch(),
            'stoch_d': stoch.stoch_signal(),
            'stoch_overbought': np.where(stoch.stoch() > 80, 1, 0),
            'stoch_oversold': np.where(stoch.stoch() < 20, 1, 0)
        }
        
        return pd.DataFrame(factors, index=data.index)
    
    def _calculate_volatility(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算波动率因子
        """
        periods = kwargs.get('periods', [5, 10, 20, 50])
        factors = {}
        
        returns = data['close'].pct_change()
        
        for period in periods:
            # 历史波动率
            vol = returns.rolling(window=period).std() * np.sqrt(252)
            factors[f'volatility_{period}'] = vol
            
            # 波动率比率
            if period > 5:
                vol_ratio = vol / returns.rolling(window=5).std() * np.sqrt(252)
                factors[f'vol_ratio_{period}_5'] = vol_ratio
            
            # 波动率变化
            vol_change = vol.pct_change()
            factors[f'vol_change_{period}'] = vol_change
        
        return pd.DataFrame(factors, index=data.index)
    
    def _calculate_atr(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算ATR因子
        """
        periods = kwargs.get('periods', [14, 21])
        factors = {}
        
        for period in periods:
            atr = ta.volatility.AverageTrueRange(data['high'], data['low'], data['close'], window=period)
            factors[f'atr_{period}'] = atr.average_true_range()
            
            # ATR比率
            atr_ratio = factors[f'atr_{period}'] / data['close']
            factors[f'atr_ratio_{period}'] = atr_ratio
        
        return pd.DataFrame(factors, index=data.index)
    
    def _calculate_bollinger_bands(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算布林带因子
        """
        period = kwargs.get('period', 20)
        std_dev = kwargs.get('std_dev', 2)
        
        bb = ta.volatility.BollingerBands(data['close'], window=period, window_dev=std_dev)
        
        factors = {
            'bb_upper': bb.bollinger_hband(),
            'bb_middle': bb.bollinger_mavg(),
            'bb_lower': bb.bollinger_lband(),
            'bb_width': bb.bollinger_wband(),
            'bb_percent': bb.bollinger_pband(),
            'bb_position': (data['close'] - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband())
        }
        
        return pd.DataFrame(factors, index=data.index)
    
    def _calculate_volume_indicators(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算成交量指标因子
        """
        periods = kwargs.get('periods', [20, 50])
        factors = {}
        
        for period in periods:
            # 成交量移动平均
            vol_ma = data['volume'].rolling(window=period).mean()
            factors[f'volume_ma_{period}'] = vol_ma
            
            # 成交量比率
            vol_ratio = data['volume'] / vol_ma
            factors[f'volume_ratio_{period}'] = vol_ratio
            
            # 价量关系
            price_change = data['close'].pct_change()
            vol_price_corr = price_change.rolling(window=period).corr(data['volume'])
            factors[f'vol_price_corr_{period}'] = vol_price_corr
        
        return pd.DataFrame(factors, index=data.index)
    
    def _calculate_obv(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算OBV因子
        """
        obv = ta.volume.OnBalanceVolumeIndicator(data['close'], data['volume'])
        
        factors = {
            'obv': obv.on_balance_volume(),
            'obv_change': obv.on_balance_volume().pct_change(),
            'obv_ma': obv.on_balance_volume().rolling(window=20).mean()
        }
        
        return pd.DataFrame(factors, index=data.index)
    
    def _calculate_vwap(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算VWAP因子
        """
        vwap = ta.volume.VolumeWeightedAveragePrice(data['high'], data['low'], data['close'], data['volume'])
        
        factors = {
            'vwap': vwap.volume_weighted_average_price(),
            'vwap_distance': (data['close'] - vwap.volume_weighted_average_price()) / vwap.volume_weighted_average_price()
        }
        
        return pd.DataFrame(factors, index=data.index)
    
    def _calculate_price_patterns(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算价格形态因子
        """
        factors = {}
        
        # 锤子线
        body = abs(data['close'] - data['open'])
        lower_shadow = np.minimum(data['open'], data['close']) - data['low']
        upper_shadow = data['high'] - np.maximum(data['open'], data['close'])
        
        hammer = (lower_shadow > 2 * body) & (upper_shadow < body)
        factors['hammer'] = hammer.astype(int)
        
        # 十字星
        doji = body <= (data['high'] - data['low']) * 0.1
        factors['doji'] = doji.astype(int)
        
        # 缺口
        gap_up = data['low'] > data['high'].shift(1)
        gap_down = data['high'] < data['low'].shift(1)
        factors['gap_up'] = gap_up.astype(int)
        factors['gap_down'] = gap_down.astype(int)
        
        return pd.DataFrame(factors, index=data.index)
    
    def _calculate_support_resistance(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算支撑阻力因子
        """
        period = kwargs.get('period', 20)
        factors = {}
        
        # 支撑位和阻力位
        support = data['low'].rolling(window=period).min()
        resistance = data['high'].rolling(window=period).max()
        
        # 距离支撑阻力位的距离
        support_distance = (data['close'] - support) / data['close']
        resistance_distance = (resistance - data['close']) / data['close']
        
        factors['support_distance'] = support_distance
        factors['resistance_distance'] = resistance_distance
        factors['price_range'] = (resistance - support) / data['close']
        
        return pd.DataFrame(factors, index=data.index)
    
    def _calculate_statistical_factors(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算统计因子
        """
        periods = kwargs.get('periods', [20, 50])
        factors = {}
        
        returns = data['close'].pct_change()
        
        for period in periods:
            # 偏度
            skewness = returns.rolling(window=period).skew()
            factors[f'skewness_{period}'] = skewness
            
            # 峰度
            kurtosis = returns.rolling(window=period).kurt()
            factors[f'kurtosis_{period}'] = kurtosis
            
            # 分位数
            q25 = returns.rolling(window=period).quantile(0.25)
            q75 = returns.rolling(window=period).quantile(0.75)
            factors[f'q25_{period}'] = q25
            factors[f'q75_{period}'] = q75
        
        return pd.DataFrame(factors, index=data.index)
    
    def _calculate_z_score(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算Z-score因子
        """
        periods = kwargs.get('periods', [20, 50])
        factors = {}
        
        for period in periods:
            # 价格Z-score
            price_mean = data['close'].rolling(window=period).mean()
            price_std = data['close'].rolling(window=period).std()
            price_zscore = (data['close'] - price_mean) / price_std
            factors[f'price_zscore_{period}'] = price_zscore
            
            # 收益率Z-score
            returns = data['close'].pct_change()
            returns_mean = returns.rolling(window=period).mean()
            returns_std = returns.rolling(window=period).std()
            returns_zscore = (returns - returns_mean) / returns_std
            factors[f'returns_zscore_{period}'] = returns_zscore
        
        return pd.DataFrame(factors, index=data.index)
    
    def _calculate_percentile_rank(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算百分位排名因子
        """
        periods = kwargs.get('periods', [20, 50])
        factors = {}
        
        for period in periods:
            # 价格百分位排名
            price_rank = data['close'].rolling(window=period).rank(pct=True)
            factors[f'price_rank_{period}'] = price_rank
            
            # 成交量百分位排名
            volume_rank = data['volume'].rolling(window=period).rank(pct=True)
            factors[f'volume_rank_{period}'] = volume_rank
        
        return pd.DataFrame(factors, index=data.index)
    
    def _calculate_custom_factors(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算自定义因子
        """
        factors = {}
        
        # 这里可以添加自定义的因子计算逻辑
        # 例如：基于特定业务逻辑的因子
        
        return pd.DataFrame(factors, index=data.index)
    
    def add_custom_factor(self, name: str, func: Callable):
        """
        添加自定义因子
        
        Args:
            name: 因子名称
            func: 因子计算函数
        """
        self.factor_registry[name] = func
    
    def get_available_factors(self) -> List[str]:
        """
        获取所有可用的因子列表
        
        Returns:
            因子名称列表
        """
        return list(self.factor_registry.keys()) 