"""
统一因子构建器 V3.0
整合所有因子构建功能，与透明因子存储系统完全兼容
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Callable
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from .factor_storage import TransparentFactorStorage
from .factor_engine import FactorEngine


class FactorBuilder:
    """
    统一因子构建器 V3.0
    整合所有类型的因子构建功能，与透明因子存储系统完全兼容
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化因子构建器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.storage = TransparentFactorStorage()
        self.engine = FactorEngine()
        
        # 注册所有可用的因子构建方法
        self.factor_builders = {
            'technical': self._build_technical_factors,
            'statistical': self._build_statistical_factors,
            'advanced': self._build_advanced_factors,
            'ml': self._build_ml_factors,
            'crypto': self._build_crypto_factors,
            'pattern': self._build_pattern_factors,
            'composite': self._build_composite_factors,
            'sentiment': self._build_sentiment_factors
        }
        
    def build_all_factors(self, data: pd.DataFrame, 
                         factor_types: Optional[List[str]] = None,
                         save_to_storage: bool = True,
                         progress_callback: Optional[callable] = None,
                         **kwargs) -> pd.DataFrame:
        """
        构建所有类型的因子
        
        Args:
            data: 市场数据
            factor_types: 因子类型列表，如果为None则构建所有类型
            save_to_storage: 是否保存到V3存储系统
            **kwargs: 其他参数
            
        Returns:
            因子DataFrame
        """
        if factor_types is None:
            factor_types = list(self.factor_builders.keys())
        
        all_factors = {}
        built_factors = {}
        
        print(f"🚀 开始构建因子，类型: {', '.join(factor_types)}")
        
        for factor_type in factor_types:
            if factor_type in self.factor_builders:
                try:
                    print(f"📊 构建 {factor_type} 类型因子...")
                    factors = self.factor_builders[factor_type](data, **kwargs)
                    
                    if isinstance(factors, dict):
                        all_factors.update(factors)
                        built_factors[factor_type] = factors
                    elif isinstance(factors, pd.DataFrame):
                        for col in factors.columns:
                            all_factors[col] = factors[col]
                        built_factors[factor_type] = factors.to_dict('series')
                        
                    print(f"✅ {factor_type} 类型因子构建完成，共 {len(factors)} 个")
                    
                except Exception as e:
                    print(f"❌ 构建 {factor_type} 类型因子失败: {e}")
                    continue
        
        # 转换为DataFrame
        factors_df = pd.DataFrame(all_factors, index=data.index)
        
        print(f"🎉 因子构建完成，总共 {len(factors_df.columns)} 个因子")
        
        # 保存到V3存储系统
        if save_to_storage:
            self._save_factors_to_storage(built_factors, data, **kwargs)
        
        return factors_df
    
    def _build_technical_factors(self, data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """构建技术指标因子"""
        factors = {}
        
        # 移动平均类因子
        for period in [5, 10, 20, 50, 100]:
            # SMA
            factors[f'sma_{period}'] = data['close'].rolling(window=period).mean()
            
            # EMA
            factors[f'ema_{period}'] = data['close'].ewm(span=period).mean()
            
            # 价格位置
            factors[f'price_position_sma_{period}'] = (data['close'] - factors[f'sma_{period}']) / factors[f'sma_{period}']
        
        # 动量类因子
        for period in [14, 21]:
            # RSI
            delta = data['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            rs = avg_gain / avg_loss
            factors[f'rsi_{period}'] = 100 - (100 / (1 + rs))
            
            # 动量
            factors[f'momentum_{period}'] = data['close'] / data['close'].shift(period) - 1
        
        # 波动率类因子
        for period in [20, 50]:
            # 历史波动率
            returns = data['close'].pct_change()
            factors[f'volatility_{period}'] = returns.rolling(window=period).std() * np.sqrt(period)
            
            # ATR
            high_low = data['high'] - data['low']
            high_close = np.abs(data['high'] - data['close'].shift())
            low_close = np.abs(data['low'] - data['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            factors[f'atr_{period}'] = true_range.rolling(window=period).mean()
        
        # 布林带
        for period in [20, 50]:
            sma = data['close'].rolling(window=period).mean()
            std = data['close'].rolling(window=period).std()
            factors[f'bb_upper_{period}'] = sma + (std * 2)
            factors[f'bb_lower_{period}'] = sma - (std * 2)
            factors[f'bb_width_{period}'] = (factors[f'bb_upper_{period}'] - factors[f'bb_lower_{period}']) / sma
            factors[f'bb_position_{period}'] = (data['close'] - factors[f'bb_lower_{period}']) / (factors[f'bb_upper_{period}'] - factors[f'bb_lower_{period}'])
        
        # 成交量类因子
        for period in [20, 50]:
            # 成交量SMA
            factors[f'volume_sma_{period}'] = data['volume'].rolling(window=period).mean()
            
            # 成交量比率
            factors[f'volume_ratio_{period}'] = data['volume'] / factors[f'volume_sma_{period}']
            
            # OBV
            obv = pd.Series(index=data.index, dtype=float)
            obv.iloc[0] = data['volume'].iloc[0]
            for i in range(1, len(data)):
                if data['close'].iloc[i] > data['close'].iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] + data['volume'].iloc[i]
                elif data['close'].iloc[i] < data['close'].iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] - data['volume'].iloc[i]
                else:
                    obv.iloc[i] = obv.iloc[i-1]
            factors[f'obv_{period}'] = obv
            
            # VWAP
            typical_price = (data['high'] + data['low'] + data['close']) / 3
            vwap = (typical_price * data['volume']).rolling(window=period).sum() / data['volume'].rolling(window=period).sum()
            factors[f'vwap_{period}'] = vwap
        
        return factors
    
    def _build_statistical_factors(self, data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """构建统计因子"""
        factors = {}
        
        # Z-score类因子
        for period in [20, 50, 100]:
            # 价格Z-score
            sma = data['close'].rolling(window=period).mean()
            std = data['close'].rolling(window=period).std()
            factors[f'price_zscore_{period}'] = (data['close'] - sma) / std
            
            # 收益率Z-score
            returns = data['close'].pct_change()
            returns_sma = returns.rolling(window=period).mean()
            returns_std = returns.rolling(window=period).std()
            factors[f'returns_zscore_{period}'] = (returns - returns_sma) / returns_std
        
        # 百分位排名
        for period in [20, 50]:
            factors[f'price_percentile_{period}'] = data['close'].rolling(window=period).rank(pct=True)
            factors[f'volume_percentile_{period}'] = data['volume'].rolling(window=period).rank(pct=True)
        
        # 相关性因子
        for period in [20, 50]:
            # 价格与成交量相关性
            factors[f'price_volume_corr_{period}'] = data['close'].rolling(window=period).corr(data['volume'])
            
            # 高低价相关性
            factors[f'high_low_corr_{period}'] = data['high'].rolling(window=period).corr(data['low'])
        
        # 偏度和峰度
        for period in [20, 50]:
            returns = data['close'].pct_change()
            factors[f'returns_skew_{period}'] = returns.rolling(window=period).skew()
            factors[f'returns_kurt_{period}'] = returns.rolling(window=period).kurt()
        
        # 分位数因子
        for period in [20, 50]:
            for q in [0.1, 0.25, 0.75, 0.9]:
                factors[f'price_q{int(q*100)}_{period}'] = data['close'].rolling(window=period).quantile(q)
        
        return factors
    
    def _build_advanced_factors(self, data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """构建高级因子"""
        factors = {}
        
        # 趋势强度因子
        for period in [20, 50]:
            # ADX
            high_diff = data['high'].diff()
            low_diff = data['low'].diff()
            
            plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
            minus_dm = -low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
            
            tr = self._calculate_true_range(data)
            
            plus_di = 100 * (plus_dm.rolling(window=period).mean() / tr.rolling(window=period).mean())
            minus_di = 100 * (minus_dm.rolling(window=period).mean() / tr.rolling(window=period).mean())
            
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            factors[f'adx_{period}'] = dx.rolling(window=period).mean()
        
        # 价格形态因子
        for period in [5, 10, 20]:
            # 价格突破
            high_max = data['high'].rolling(window=period).max()
            low_min = data['low'].rolling(window=period).min()
            
            factors[f'breakout_high_{period}'] = (data['close'] > high_max.shift(1)).astype(int)
            factors[f'breakout_low_{period}'] = (data['close'] < low_min.shift(1)).astype(int)
            
            # 价格通道位置
            factors[f'channel_position_{period}'] = (data['close'] - low_min) / (high_max - low_min)
        
        # 支撑阻力因子
        for period in [20, 50]:
            # 支撑位（局部最小值）
            support = data['low'].rolling(window=period, center=True).min()
            factors[f'support_level_{period}'] = support
            
            # 阻力位（局部最大值）
            resistance = data['high'].rolling(window=period, center=True).max()
            factors[f'resistance_level_{period}'] = resistance
            
            # 距离支撑阻力的比例
            factors[f'distance_to_support_{period}'] = (data['close'] - support) / data['close']
            factors[f'distance_to_resistance_{period}'] = (resistance - data['close']) / data['close']
        
        # 市场结构因子
        for period in [20, 50]:
            # 高低点比率
            high_ratio = data['high'] / data['high'].rolling(window=period).max()
            low_ratio = data['low'] / data['low'].rolling(window=period).min()
            factors[f'high_low_ratio_{period}'] = high_ratio / low_ratio
            
            # 价格效率
            price_change = abs(data['close'] - data['close'].shift(period))
            path_length = data['close'].diff().abs().rolling(window=period).sum()
            factors[f'price_efficiency_{period}'] = price_change / path_length
        
        return factors
    
    def _build_ml_factors(self, data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """构建机器学习因子"""
        factors = {}
        
        # 获取进度回调函数
        progress_callback = kwargs.get('progress_callback')
        
        try:
            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
            from sklearn.linear_model import LinearRegression, Ridge
            from sklearn.preprocessing import StandardScaler
            from sklearn.decomposition import PCA
            from sklearn.feature_selection import SelectKBest, f_regression
            
            # 步骤1: 准备特征 (20%)
            if progress_callback:
                progress_callback('factor_building', 20, '正在准备ML特征数据...')
            
            features = self._prepare_ml_features(data)
            target = self._prepare_ml_target(data)
            
            # 步骤2: 数据清理 (30%)
            if progress_callback:
                progress_callback('factor_building', 30, '正在清理和验证数据...')
            
            # 移除NaN值
            valid_idx = ~(features.isna().any(axis=1) | target.isna())
            features_clean = features.loc[valid_idx]
            target_clean = target.loc[valid_idx]
            
            if len(features_clean) < 100:
                print("⚠️ 数据量不足，无法构建ML因子")
                if progress_callback:
                    progress_callback('factor_building', 0, '数据量不足，无法构建ML因子')
                return factors
            
            # 步骤3: 特征标准化 (40%)
            if progress_callback:
                progress_callback('factor_building', 40, '正在标准化特征数据...')
            
            # 标准化特征
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features_clean)
            
            # 步骤4: 构建随机森林因子 (50%)
            if progress_callback:
                progress_callback('factor_building', 50, '正在训练随机森林模型...')
            
            try:
                rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                rf_model.fit(features_scaled, target_clean)
                rf_predictions = rf_model.predict(features_scaled)
                
                factor_series = pd.Series(index=data.index, dtype=float)
                factor_series.loc[valid_idx] = rf_predictions
                factors['ml_random_forest'] = factor_series
                print("✅ 随机森林因子构建成功")
            except Exception as e:
                print(f"❌ 随机森林因子构建失败: {e}")
            
            # 步骤5: 构建梯度提升因子 (60%)
            if progress_callback:
                progress_callback('factor_building', 60, '正在训练梯度提升模型...')
            
            try:
                gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
                gb_model.fit(features_scaled, target_clean)
                gb_predictions = gb_model.predict(features_scaled)
                
                factor_series = pd.Series(index=data.index, dtype=float)
                factor_series.loc[valid_idx] = gb_predictions
                factors['ml_gradient_boosting'] = factor_series
                print("✅ 梯度提升因子构建成功")
            except Exception as e:
                print(f"❌ 梯度提升因子构建失败: {e}")
            
            # 步骤6: 构建PCA因子 (70%)
            if progress_callback:
                progress_callback('factor_building', 70, '正在计算PCA主成分...')
            
            try:
                pca = PCA(n_components=3, random_state=42)
                pca_features = pca.fit_transform(features_scaled)
                
                for i in range(3):
                    factor_series = pd.Series(index=data.index, dtype=float)
                    factor_series.loc[valid_idx] = pca_features[:, i]
                    factors[f'ml_pca_component_{i+1}'] = factor_series
                print("✅ PCA因子构建成功")
            except Exception as e:
                print(f"❌ PCA因子构建失败: {e}")
            
            # 步骤7: 构建特征选择因子 (80%)
            if progress_callback:
                progress_callback('factor_building', 80, '正在选择最优特征...')
            
            try:
                selector = SelectKBest(score_func=f_regression, k=5)
                selected_features = selector.fit_transform(features_scaled, target_clean)
                
                for i in range(5):
                    factor_series = pd.Series(index=data.index, dtype=float)
                    factor_series.loc[valid_idx] = selected_features[:, i]
                    factors[f'ml_selected_feature_{i+1}'] = factor_series
                print("✅ 特征选择因子构建成功")
            except Exception as e:
                print(f"❌ 特征选择因子构建失败: {e}")
                
            # 步骤8: ML因子构建完成 (90%)
            if progress_callback:
                progress_callback('factor_building', 90, 'ML因子构建完成，正在整理结果...')
            
            print(f"🎉 ML因子构建完成，共生成 {len(factors)} 个因子")
            
        except ImportError:
            print("⚠️ sklearn未安装，跳过ML因子构建")
            if progress_callback:
                progress_callback('factor_building', 0, 'sklearn未安装，跳过ML因子构建')
        
        return factors
    
    def _clean_ml_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """清理ML特征数据，处理无穷大值和异常值"""
        try:
            # 复制特征数据
            cleaned_features = features.copy()
            
            # 处理无穷大值
            cleaned_features = cleaned_features.replace([np.inf, -np.inf], np.nan)
            
            # 处理异常大的值（超过3个标准差）
            for col in cleaned_features.columns:
                if cleaned_features[col].dtype in ['float64', 'float32']:
                    # 计算有效值的统计信息
                    valid_data = cleaned_features[col].dropna()
                    if len(valid_data) > 0:
                        mean_val = valid_data.mean()
                        std_val = valid_data.std()
                        
                        if std_val > 0:
                            # 将超过3个标准差的值设为NaN
                            upper_bound = mean_val + 3 * std_val
                            lower_bound = mean_val - 3 * std_val
                            cleaned_features[col] = cleaned_features[col].clip(lower_bound, upper_bound)
            
            # 用前向填充和后向填充处理NaN值
            cleaned_features = cleaned_features.fillna(method='ffill').fillna(method='bfill')
            
            # 如果还有NaN值，用0填充
            cleaned_features = cleaned_features.fillna(0)
            
            print(f"✅ ML特征清理完成，处理了 {features.isna().sum().sum()} 个NaN值")
            return cleaned_features
            
        except Exception as e:
            print(f"❌ ML特征清理失败: {e}")
            # 如果清理失败，返回原始特征
            return features
    
    def _build_crypto_factors(self, data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """构建加密货币特有因子"""
        factors = {}
        
        # 资金费率相关因子（如果有数据）
        if 'funding_rate' in data.columns:
            for period in [8, 24, 72]:
                factors[f'funding_rate_ma_{period}'] = data['funding_rate'].rolling(window=period).mean()
                factors[f'funding_rate_std_{period}'] = data['funding_rate'].rolling(window=period).std()
        
        # 永续溢价因子
        if 'mark_price' in data.columns and 'index_price' in data.columns:
            factors['perpetual_premium'] = (data['mark_price'] - data['index_price']) / data['index_price']
        
        # 网络价值因子（模拟）
        for period in [24, 72, 168]:
            # 模拟网络活跃度
            volume_ma = data['volume'].rolling(window=period).mean()
            price_ma = data['close'].rolling(window=period).mean()
            factors[f'network_value_{period}'] = volume_ma * price_ma
        
        # 波动率调整收益
        for period in [24, 72]:
            returns = data['close'].pct_change()
            volatility = returns.rolling(window=period).std()
            factors[f'volatility_adjusted_return_{period}'] = returns / volatility
        
        return factors
    
    def _build_pattern_factors(self, data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """构建价格形态因子"""
        factors = {}
        
        # 蜡烛图形态
        for period in [1, 3, 5]:
            # 锤子线
            body_size = abs(data['close'] - data['open'])
            lower_shadow = data['open'].where(data['close'] > data['open'], data['close']) - data['low']
            upper_shadow = data['high'] - data['open'].where(data['close'] > data['open'], data['close'])
            
            hammer_condition = (lower_shadow > 2 * body_size) & (upper_shadow < body_size)
            factors[f'hammer_pattern_{period}'] = hammer_condition.rolling(window=period).sum()
            
            # 十字星
            doji_condition = (body_size < 0.1 * (data['high'] - data['low']))
            factors[f'doji_pattern_{period}'] = doji_condition.rolling(window=period).sum()
        
        # 缺口因子
        for period in [1, 3, 5]:
            # 向上缺口
            gap_up = (data['low'] > data['high'].shift(1))
            factors[f'gap_up_{period}'] = gap_up.rolling(window=period).sum()
            
            # 向下缺口
            gap_down = (data['high'] < data['low'].shift(1))
            factors[f'gap_down_{period}'] = gap_down.rolling(window=period).sum()
        
        # 趋势线因子
        for period in [20, 50]:
            # 上升趋势线
            highs = data['high'].rolling(window=period, center=True).max()
            factors[f'uptrend_line_{period}'] = (data['close'] > highs * 0.98).astype(int)
            
            # 下降趋势线
            lows = data['low'].rolling(window=period, center=True).min()
            factors[f'downtrend_line_{period}'] = (data['close'] < lows * 1.02).astype(int)
        
        return factors
    
    def _build_composite_factors(self, data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """构建复合因子"""
        factors = {}
        
        # 多因子组合
        for period in [20, 50]:
            # 趋势动量组合
            sma_20 = data['close'].rolling(window=20).mean()
            sma_50 = data['close'].rolling(window=50).mean()
            trend = (sma_20 > sma_50).astype(int)
            
            rsi = self._calculate_rsi(data['close'], period)
            momentum = (rsi > 50).astype(int)
            
            factors[f'trend_momentum_{period}'] = trend * momentum
            
            # 波动率调整动量
            returns = data['close'].pct_change()
            volatility = returns.rolling(window=period).std()
            momentum_vol_adj = returns / volatility
            factors[f'momentum_vol_adj_{period}'] = momentum_vol_adj.rolling(window=period).mean()
        
        # 市场情绪因子
        for period in [20, 50]:
            # 价格成交量一致性
            price_up = (data['close'] > data['close'].shift(1)).astype(int)
            volume_up = (data['volume'] > data['volume'].shift(1)).astype(int)
            consistency = (price_up == volume_up).astype(int)
            factors[f'price_volume_consistency_{period}'] = consistency.rolling(window=period).mean()
        
        return factors
    
    def _build_sentiment_factors(self, data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """构建情绪因子"""
        factors = {}
        
        # 恐慌贪婪指数（模拟）
        for period in [24, 72, 168]:
            # 基于价格和成交量的情绪指标
            returns = data['close'].pct_change()
            volatility = returns.rolling(window=period).std()
            volume_ratio = data['volume'] / data['volume'].rolling(window=period).mean()
            
            # 恐慌指数：高波动率 + 高成交量
            fear_index = (volatility * volume_ratio).rolling(window=period).mean()
            factors[f'fear_index_{period}'] = fear_index
            
            # 贪婪指数：低波动率 + 稳定上涨
            greed_condition = (volatility < volatility.rolling(window=period).quantile(0.3)) & (returns > 0)
            factors[f'greed_index_{period}'] = greed_condition.rolling(window=period).mean()
        
        # 市场结构情绪
        for period in [20, 50]:
            # 新高新低比率
            new_highs = (data['close'] > data['close'].rolling(window=period).max().shift(1)).astype(int)
            new_lows = (data['close'] < data['close'].rolling(window=period).min().shift(1)).astype(int)
            
            factors[f'new_highs_ratio_{period}'] = new_highs.rolling(window=period).mean()
            factors[f'new_lows_ratio_{period}'] = new_lows.rolling(window=period).mean()
            factors[f'high_low_ratio_{period}'] = factors[f'new_highs_ratio_{period}'] / (factors[f'new_lows_ratio_{period}'] + 1e-8)
        
        return factors
    
    def _prepare_ml_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """准备机器学习特征"""
        features = pd.DataFrame(index=data.index)
        
        try:
            # 价格特征
            features['price'] = data['close']
            features['price_change'] = data['close'].pct_change()
            features['price_change_2'] = data['close'].pct_change(2)
            features['price_change_5'] = data['close'].pct_change(5)
            
            # 技术指标特征
            features['sma_20'] = data['close'].rolling(window=20).mean()
            features['sma_50'] = data['close'].rolling(window=50).mean()
            features['rsi_14'] = self._calculate_rsi(data['close'], 14)
            
            # 成交量特征
            features['volume'] = data['volume']
            features['volume_change'] = data['volume'].pct_change()
            features['volume_ma'] = data['volume'].rolling(window=20).mean()
            
            # 波动率特征
            returns = data['close'].pct_change()
            features['volatility'] = returns.rolling(window=20).std()
            
            # 数据清理：处理无穷大值和异常值
            features = self._clean_ml_features(features)
            
            return features
            
        except Exception as e:
            print(f"❌ 准备ML特征失败: {e}")
            # 返回空的特征DataFrame
            return pd.DataFrame(index=data.index)
    
    def _prepare_ml_target(self, data: pd.DataFrame) -> pd.Series:
        """准备机器学习目标变量"""
        # 使用未来1期的收益率作为目标
        return data['close'].pct_change().shift(-1)
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """计算RSI指标"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_true_range(self, data: pd.DataFrame) -> pd.Series:
        """计算真实波幅"""
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        
        return true_range
    
    def _save_factors_to_storage(self, built_factors: Dict, data: pd.DataFrame, **kwargs):
        """将构建的因子保存到V3存储系统"""
        print("💾 开始保存因子到V3存储系统...")
        
        for factor_type, factors in built_factors.items():
            for factor_name, factor_series in factors.items():
                try:
                    # 创建因子定义 - 移除V3后缀
                    factor_id = factor_name
                    
                    # 根据因子类型选择保存方法
                    if factor_type == 'ml':
                        # ML因子：智能检测artifact路径，优先保存为ml_model类型
                        models_dir = Path(__file__).parent.parent.parent / "factorlib" / "models"
                        artifact_file = models_dir / f"{factor_id}.pkl"
                        
                        # 检查多个可能的路径，确保能找到artifact
                        artifact_found = False
                        artifact_paths = [
                            artifact_file,
                            models_dir / f"{factor_id}.pkl",
                            Path.cwd() / "factorlib" / "models" / f"{factor_id}.pkl",
                            Path(__file__).parent.parent.parent / "factorlib" / "models" / f"{factor_id}.pkl"
                        ]
                        
                        for path in artifact_paths:
                            if path.exists():
                                artifact_file = path
                                artifact_found = True
                                print(f"✅ 找到ML因子artifact: {path}")
                                break
                        
                        if artifact_found:
                            # 保存为ml_model类型
                            success = self.storage.save_ml_model_factor(
                                factor_id=factor_id,
                                name=factor_name,
                                artifact_filename=artifact_file.name,
                                description=f"机器学习生成的{factor_name}因子",
                                category=factor_type,
                                parameters={}
                            )
                            if success:
                                print(f"✅ ML因子 {factor_name} 保存为ml_model类型成功")
                        else:
                            print(f"⚠️ 未找到ML因子artifact: {factor_id}.pkl，尝试保存为function类型")
                            # 即使没有artifact，ML因子也应该保存为function类型
                            # 因为它们的计算逻辑在functions文件中
                        success = self.storage.save_function_factor(
                            factor_id=factor_id,
                            name=factor_name,
                            function_code=self._generate_ml_factor_code(factor_name, factor_type),
                                description=f"机器学习生成的{factor_name}因子（函数实现）",
                            category=factor_type,
                            parameters={}
                        )
                        if success:
                            print(f"✅ ML因子 {factor_name} 保存为function类型成功")
                    else:
                        # 其他因子保存为公式类型
                        success = self.storage.save_formula_factor(
                            factor_id=factor_id,
                            name=factor_name,
                            formula=self._generate_factor_formula(factor_name, factor_type),
                            description=f"自动生成的{factor_name}因子",
                            category=factor_type,
                            parameters=self._extract_factor_parameters(factor_name)
                        )
                    
                    if success:
                        print(f"✅ 因子 {factor_name} 保存成功")
                    else:
                        print(f"❌ 因子 {factor_name} 保存失败")
                        
                except Exception as e:
                    print(f"❌ 保存因子 {factor_name} 时出错: {e}")
                    continue
        
        print("💾 因子保存完成")
    
    def _generate_ml_factor_code(self, factor_name: str, factor_type: str) -> str:
        """生成ML因子的Python代码"""
        
        # 判断是否为PCA因子
        is_pca_factor = factor_name.startswith('pca_component_')
        
        if is_pca_factor:
            return self._generate_pca_factor_code(factor_name, factor_type)
        else:
            return self._generate_standard_ml_factor_code(factor_name, factor_type)
    
    def _generate_pca_factor_code(self, factor_name: str, factor_type: str) -> str:
        """生成PCA因子的Python代码"""
        # 从因子名中提取组件索引
        component_index = int(factor_name.split('_')[-1]) - 1  # pca_component_3 -> 2
        
        return f"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def calculate(data: pd.DataFrame, **kwargs) -> pd.Series:
    \"\"\"
    计算{factor_name}因子
    
    智能加载预训练的PCA模型，使用pkl文件进行降维
    
    Args:
        data: 市场数据DataFrame，必须包含 OHLCV 列
        **kwargs: 其他参数
        
        Returns:
        因子值Series，第{component_index + 1}个主成分
    \"\"\"
    try:
        # 检查数据完整性
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if data is None or len(data) == 0:
            return pd.Series(dtype=float)
        
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            print(f"缺少必要的列: {{missing_cols}}")
            return pd.Series(index=data.index, dtype=float)
        
        # 智能查找模型文件
        model_paths = [
            Path(__file__).parent.parent / "models" / "{factor_name}.pkl",
            Path.cwd() / "factorlib" / "models" / "{factor_name}.pkl",
            Path(__file__).parent.parent.parent / "models" / "{factor_name}.pkl"
        ]
        
        artifact_file = None
        for path in model_paths:
            if path.exists():
                artifact_file = path
                break
        
        if artifact_file is None:
            print(f"未找到模型文件: {{factor_name}}.pkl")
            print(f"尝试过的路径: {{[str(p) for p in model_paths]}}")
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
            print(f"缺少特征列: {{missing}}")
            # 对缺失列补NaN，保持列齐全
            for c in missing:
                features[c] = np.nan
        
        X = features[feature_columns]
        
        # 清洗与标准化 - PCA不接受NaN值
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
            print(f"删除包含全NaN的行，从{{len(X_cleaned)}}行减少到{{valid_rows.sum()}}行")
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
            
            # 取第{component_index + 1}个主成分（索引为{component_index}）
            # 如果n_components < {component_index + 1}，则取最后一个
            component_index_actual = min({component_index}, model.n_components_ - 1)
            y_pred = X_transformed[:, component_index_actual]
            
        except Exception as e:
            print(f"PCA转换失败: {{e}}")
            # 如果PCA转换失败，返回NaN序列
            return pd.Series(index=data.index, dtype=float)
        
        return pd.Series(y_pred, index=data.index)
        
    except Exception as e:
        print(f"计算{factor_name}因子时出错: {{e}}")
        import traceback
        traceback.print_exc()
    return pd.Series(index=data.index, dtype=float)

def _build_features(data: pd.DataFrame) -> pd.DataFrame:
    \"\"\"构建与训练一致的特征\"\"\"
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
"""

    def _generate_standard_ml_factor_code(self, factor_name: str, factor_type: str) -> str:
        """生成标准ML因子的Python代码"""
        return f"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def calculate(data: pd.DataFrame, **kwargs) -> pd.Series:
    \"\"\"
    计算{factor_name}因子
    
    智能加载预训练的ML模型，使用pkl文件进行推理
    
    Args:
        data: 市场数据DataFrame，必须包含 OHLCV 列
        **kwargs: 其他参数
        
        Returns:
        因子值Series，预测结果
    \"\"\"
    try:
        # 检查数据完整性
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if data is None or len(data) == 0:
            return pd.Series(dtype=float)
        
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            print(f"缺少必要的列: {{missing_cols}}")
            return pd.Series(index=data.index, dtype=float)
        
        # 智能查找模型文件
        model_paths = [
            Path(__file__).parent.parent / "models" / "{factor_name}.pkl",
            Path.cwd() / "factorlib" / "models" / "{factor_name}.pkl",
            Path(__file__).parent.parent.parent / "models" / "{factor_name}.pkl"
        ]
        
        artifact_file = None
        for path in model_paths:
            if path.exists():
                artifact_file = path
                break
        
        if artifact_file is None:
            print(f"未找到模型文件: {{factor_name}}.pkl")
            print(f"尝试过的路径: {{[str(p) for p in model_paths]}}")
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
            print(f"缺少特征列: {{missing}}")
            # 对缺失列补NaN，保持列齐全
            for c in missing:
                features[c] = np.nan
        
        X = features[feature_columns]
        
        # 清洗与标准化
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(method='ffill').fillna(method='bfill')
        
        if scaler is not None:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X.values
        
        # 预测
        y_pred = model.predict(X_scaled)
        return pd.Series(y_pred, index=data.index)
        
    except Exception as e:
        print(f"计算{factor_name}因子时出错: {{e}}")
        import traceback
        traceback.print_exc()
    return pd.Series(index=data.index, dtype=float)

def _build_features(data: pd.DataFrame) -> pd.DataFrame:
    \"\"\"构建与训练一致的特征\"\"\"
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
"""
    
    def _generate_factor_formula(self, factor_name: str, factor_type: str) -> str:
        """生成因子的公式描述"""
        return f"# {factor_name} 因子的计算公式\n# 类型: {factor_type}\n# 自动生成于 {pd.Timestamp.now()}"
    
    def _extract_factor_parameters(self, factor_name: str) -> Dict:
        """提取因子参数"""
        # 从因子名称中提取参数
        params = {}
        
        # 安全解析尾部数字作为period，仅当确为数字时转换
        def _parse_trailing_period(name: str):
            segment = name.split('_')[-1]
            return int(segment) if segment.isdigit() else None

        if 'sma_' in factor_name:
            period = _parse_trailing_period(factor_name)
            if period is not None:
                params['period'] = period
        elif 'rsi_' in factor_name:
            period = _parse_trailing_period(factor_name)
            if period is not None:
                params['period'] = period
        elif 'bb_' in factor_name:
            period = _parse_trailing_period(factor_name)
            # 对无数字后缀的布林带名称使用安全默认值
            params['period'] = period if period is not None else 20
            params['std_dev'] = 2
        
        return params
    
    def get_available_factors(self) -> Dict[str, List[str]]:
        """获取所有可用因子的信息"""
        return {
            'technical': ['sma', 'ema', 'rsi', 'macd', 'bollinger_bands', 'atr', 'obv', 'vwap'],
            'statistical': ['zscore', 'percentile', 'correlation', 'skewness', 'kurtosis'],
            'advanced': ['adx', 'trend_strength', 'support_resistance', 'market_structure'],
            'ml': ['random_forest', 'gradient_boosting', 'pca', 'feature_selection'],
            'crypto': ['funding_rate', 'perpetual_premium', 'network_value'],
            'pattern': ['candlestick_patterns', 'gaps', 'trendlines'],
            'composite': ['trend_momentum', 'volatility_adjusted', 'market_sentiment'],
            'sentiment': ['fear_greed', 'market_structure_sentiment']
        }
    
    def build_custom_factor(self, data: pd.DataFrame, factor_name: str, 
                          factor_func: Callable, **kwargs) -> pd.Series:
        """
        构建自定义因子
        
        Args:
            data: 市场数据
            factor_name: 因子名称
            factor_func: 因子计算函数
            **kwargs: 其他参数
            
        Returns:
            自定义因子Series
        """
        try:
            factor = factor_func(data, **kwargs)
            if isinstance(factor, pd.Series):
                factor.name = factor_name
                return factor
            else:
                raise ValueError("因子函数必须返回pandas.Series")
        except Exception as e:
            print(f"构建自定义因子 {factor_name} 失败: {e}")
            return pd.Series(dtype=float) 