"""
因子交易API
为实时交易提供因子计算和调用接口
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
import logging
from pathlib import Path

from .factor_engine import factor_engine
from .data_loader import DataLoader


class FactorTradingAPI:
    """
    因子交易API
    专为交易系统设计的高效因子计算接口
    """
    
    def __init__(self, enable_cache: bool = True, cache_ttl_minutes: int = 5):
        """
        初始化交易API
        
        Args:
            enable_cache: 是否启用缓存
            cache_ttl_minutes: 缓存TTL（分钟）
        """
        self.engine = factor_engine
        self.data_loader = DataLoader()
        self.enable_cache = enable_cache
        self.cache_ttl_minutes = cache_ttl_minutes
        self.logger = logging.getLogger(__name__)
        
        # 实时数据缓存
        self._data_cache: Dict[str, Dict] = {}
        self._factor_cache: Dict[str, Dict] = {}
    
    def get_factor_value(self,
                        factor_id: str,
                        symbol: str,
                        timeframe: str,
                        lookback_periods: int = 500,
                        as_of_time: datetime = None,
                        use_cache: bool = True,
                        **factor_params) -> Optional[float]:
        """
        获取单个因子的最新值（用于实时交易）
        
        Args:
            factor_id: 因子ID
            symbol: 交易对
            timeframe: 时间框架
            lookback_periods: 回望期数
            as_of_time: 计算时点，None表示最新
            use_cache: 是否使用缓存
            **factor_params: 因子参数
            
        Returns:
            因子值
        """
        try:
            # 获取市场数据
            data = self._get_market_data(symbol, timeframe, lookback_periods, as_of_time)
            if data is None or len(data) == 0:
                return None
            
            # 构建缓存键
            cache_key = f"{factor_id}_{symbol}_{timeframe}_{str(factor_params)}"
            
            # 检查缓存
            if use_cache and self.enable_cache:
                cached_result = self._get_factor_cache(cache_key)
                if cached_result is not None:
                    return cached_result
            
            # 计算因子
            factor_result = self.engine.compute_single_factor(
                factor_id=factor_id,
                data=data,
                symbol=symbol,
                timeframe=timeframe,
                use_cache=False,  # 不使用引擎的缓存，我们有自己的缓存
                save_result=False,  # 实时计算不保存
                **factor_params
            )
            
            if factor_result is None:
                return None
            
            # 获取最新值
            if isinstance(factor_result, pd.Series):
                latest_value = factor_result.iloc[-1] if len(factor_result) > 0 else None
            elif isinstance(factor_result, pd.DataFrame):
                # 如果是DataFrame，取第一列的最新值
                latest_value = factor_result.iloc[-1, 0] if len(factor_result) > 0 else None
            else:
                latest_value = float(factor_result) if factor_result is not None else None
            
            # 缓存结果
            if use_cache and self.enable_cache and latest_value is not None:
                self._set_factor_cache(cache_key, latest_value)
            
            return latest_value
            
        except Exception as e:
            self.logger.error(f"获取因子值失败 {factor_id}: {e}")
            return None
    
    def get_multiple_factor_values(self,
                                  factor_ids: List[str],
                                  symbol: str,
                                  timeframe: str,
                                  lookback_periods: int = 500,
                                  as_of_time: datetime = None,
                                  use_cache: bool = True,
                                  **factor_params) -> Dict[str, float]:
        """
        获取多个因子的最新值
        
        Args:
            factor_ids: 因子ID列表
            symbol: 交易对
            timeframe: 时间框架
            lookback_periods: 回望期数
            as_of_time: 计算时点
            use_cache: 是否使用缓存
            **factor_params: 因子参数
            
        Returns:
            因子值字典
        """
        results = {}
        
        # 获取市场数据（共享）
        data = self._get_market_data(symbol, timeframe, lookback_periods, as_of_time)
        if data is None or len(data) == 0:
            return results
        
        # 批量计算因子
        try:
            factors_df = self.engine.compute_multiple_factors(
                factor_ids=factor_ids,
                data=data,
                symbol=symbol,
                timeframe=timeframe,
                parallel=True,
                use_cache=False,
                save_results=False,
                **factor_params
            )
            
            # 提取最新值
            if len(factors_df) > 0:
                latest_row = factors_df.iloc[-1]
                for factor_id in factor_ids:
                    if factor_id in latest_row:
                        results[factor_id] = latest_row[factor_id]
                    else:
                        # 查找带前缀的列
                        matching_cols = [col for col in latest_row.index if col.startswith(factor_id)]
                        if matching_cols:
                            results[factor_id] = latest_row[matching_cols[0]]
            
        except Exception as e:
            self.logger.error(f"批量获取因子值失败: {e}")
        
        return results
    
    def get_factor_signals(self,
                          factor_configs: List[Dict[str, Any]],
                          symbol: str,
                          timeframe: str,
                          lookback_periods: int = 500) -> Dict[str, Any]:
        """
        获取因子信号（用于交易决策）
        
        Args:
            factor_configs: 因子配置列表
                [
                    {
                        'factor_id': 'rsi',
                        'params': {'period': 14},
                        'signal_rules': {
                            'buy_threshold': 30,
                            'sell_threshold': 70,
                            'signal_type': 'threshold'
                        }
                    }
                ]
            symbol: 交易对
            timeframe: 时间框架
            lookback_periods: 回望期数
            
        Returns:
            信号字典
        """
        signals = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'timeframe': timeframe,
            'factor_signals': {},
            'combined_signal': 0,
            'signal_strength': 0
        }
        
        # 获取市场数据
        data = self._get_market_data(symbol, timeframe, lookback_periods)
        if data is None or len(data) == 0:
            return signals
        
        buy_signals = 0
        sell_signals = 0
        total_weight = 0
        
        for config in factor_configs:
            try:
                factor_id = config['factor_id']
                params = config.get('params', {})
                signal_rules = config.get('signal_rules', {})
                weight = config.get('weight', 1.0)
                
                # 计算因子值
                factor_value = self.get_factor_value(
                    factor_id, symbol, timeframe, lookback_periods, **params
                )
                
                if factor_value is None:
                    continue
                
                # 生成信号
                signal = self._generate_signal(factor_value, signal_rules)
                
                signals['factor_signals'][factor_id] = {
                    'value': factor_value,
                    'signal': signal,
                    'weight': weight,
                    'rules': signal_rules
                }
                
                # 累计信号
                if signal > 0:
                    buy_signals += signal * weight
                elif signal < 0:
                    sell_signals += abs(signal) * weight
                
                total_weight += weight
                
            except Exception as e:
                self.logger.error(f"处理因子信号失败 {config.get('factor_id')}: {e}")
        
        # 计算综合信号
        if total_weight > 0:
            net_signal = (buy_signals - sell_signals) / total_weight
            signals['combined_signal'] = np.clip(net_signal, -1, 1)
            signals['signal_strength'] = abs(net_signal)
        
        return signals
    
    def _generate_signal(self, factor_value: float, signal_rules: Dict[str, Any]) -> float:
        """
        根据规则生成信号
        
        Args:
            factor_value: 因子值
            signal_rules: 信号规则
            
        Returns:
            信号强度 (-1到1)
        """
        signal_type = signal_rules.get('signal_type', 'threshold')
        
        if signal_type == 'threshold':
            buy_threshold = signal_rules.get('buy_threshold')
            sell_threshold = signal_rules.get('sell_threshold')
            
            if buy_threshold is not None and factor_value <= buy_threshold:
                return 1.0  # 买入信号
            elif sell_threshold is not None and factor_value >= sell_threshold:
                return -1.0  # 卖出信号
            else:
                return 0.0  # 无信号
        
        elif signal_type == 'crossover':
            # 这里需要历史数据来判断交叉
            # 简化处理，返回0
            return 0.0
        
        elif signal_type == 'percentile':
            # 基于百分位的信号
            low_pct = signal_rules.get('low_percentile', 20)
            high_pct = signal_rules.get('high_percentile', 80)
            
            # 简化处理，假设因子值在0-100范围内
            if factor_value <= low_pct:
                return 1.0
            elif factor_value >= high_pct:
                return -1.0
            else:
                return 0.0
        
        return 0.0
    
    def _get_market_data(self,
                        symbol: str,
                        timeframe: str,
                        lookback_periods: int,
                        as_of_time: datetime = None) -> Optional[pd.DataFrame]:
        """
        获取市场数据（带缓存）
        
        Args:
            symbol: 交易对
            timeframe: 时间框架
            lookback_periods: 回望期数
            as_of_time: 截止时间
            
        Returns:
            市场数据
        """
        cache_key = f"{symbol}_{timeframe}_{lookback_periods}"
        
        # 检查缓存
        if self.enable_cache and cache_key in self._data_cache:
            cache_data = self._data_cache[cache_key]
            cache_time = cache_data['timestamp']
            
            # 检查缓存是否过期
            if (datetime.now() - cache_time).total_seconds() < self.cache_ttl_minutes * 60:
                return cache_data['data']
        
        try:
            # 加载数据
            end_time = as_of_time or datetime.now()
            start_time = end_time - timedelta(days=lookback_periods // 24)  # 粗略估算
            
            # 这里需要根据实际的数据加载方式调整
            # 可以从本地文件、数据库或API获取
            data = self._load_market_data_from_source(
                symbol, timeframe, start_time, end_time, lookback_periods
            )
            
            # 缓存数据
            if self.enable_cache and data is not None:
                self._data_cache[cache_key] = {
                    'data': data,
                    'timestamp': datetime.now()
                }
            
            return data
            
        except Exception as e:
            self.logger.error(f"加载市场数据失败 {symbol} {timeframe}: {e}")
            return None
    
    def _load_market_data_from_source(self,
                                     symbol: str,
                                     timeframe: str,
                                     start_time: datetime,
                                     end_time: datetime,
                                     lookback_periods: int) -> Optional[pd.DataFrame]:
        """
        从数据源加载市场数据
        
        这个方法需要根据实际的数据源进行实现
        """
        try:
            # 尝试从本地文件加载
            from pathlib import Path
            
            # 构建文件路径
            project_root = Path(__file__).parent.parent.parent
            data_dir = project_root / "data" / "binance" / "futures"
            
            # 解析symbol格式
            if '_' not in symbol:
                base_symbol = symbol
            else:
                base_symbol = symbol.split('_')[0]
            
            filename = f"{base_symbol}_USDT_USDT-{timeframe}-futures.feather"
            file_path = data_dir / filename
            
            if file_path.exists():
                data = pd.read_feather(file_path)
                
                # 设置时间索引
                if 'timestamp' in data.columns:
                    data['timestamp'] = pd.to_datetime(data['timestamp'])
                    data.set_index('timestamp', inplace=True)
                elif len(data.columns) > 5 and not isinstance(data.index, pd.DatetimeIndex):
                    # 假设第一列是时间
                    first_col = data.columns[0]
                    data[first_col] = pd.to_datetime(data[first_col])
                    data.set_index(first_col, inplace=True)
                
                # 确保必要的列存在
                required_columns = ['open', 'high', 'low', 'close', 'volume']
                if all(col in data.columns for col in required_columns):
                    # 过滤时间范围
                    if isinstance(data.index, pd.DatetimeIndex):
                        data = data[(data.index >= start_time) & (data.index <= end_time)]
                    
                    # 限制记录数
                    if len(data) > lookback_periods:
                        data = data.tail(lookback_periods)
                    
                    return data
            
            return None
            
        except Exception as e:
            self.logger.error(f"从源加载数据失败: {e}")
            return None
    
    def _get_factor_cache(self, cache_key: str) -> Optional[float]:
        """获取因子缓存"""
        if cache_key in self._factor_cache:
            cache_data = self._factor_cache[cache_key]
            cache_time = cache_data['timestamp']
            
            if (datetime.now() - cache_time).total_seconds() < self.cache_ttl_minutes * 60:
                return cache_data['value']
            else:
                # 删除过期缓存
                del self._factor_cache[cache_key]
        
        return None
    
    def _set_factor_cache(self, cache_key: str, value: float):
        """设置因子缓存"""
        self._factor_cache[cache_key] = {
            'value': value,
            'timestamp': datetime.now()
        }
    
    def clear_cache(self):
        """清理所有缓存"""
        self._data_cache.clear()
        self._factor_cache.clear()
        self.logger.info("已清理所有缓存")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        return {
            'data_cache_size': len(self._data_cache),
            'factor_cache_size': len(self._factor_cache),
            'cache_ttl_minutes': self.cache_ttl_minutes,
            'cache_enabled': self.enable_cache
        }
    
    def warmup_factors(self,
                      factor_ids: List[str],
                      symbols: List[str],
                      timeframes: List[str]) -> Dict[str, Any]:
        """
        预热因子（预先计算并缓存）
        
        Args:
            factor_ids: 因子ID列表
            symbols: 交易对列表
            timeframes: 时间框架列表
            
        Returns:
            预热结果统计
        """
        results = {
            'start_time': datetime.now(),
            'total_combinations': len(factor_ids) * len(symbols) * len(timeframes),
            'successful': 0,
            'failed': 0,
            'details': []
        }
        
        for symbol in symbols:
            for timeframe in timeframes:
                for factor_id in factor_ids:
                    try:
                        value = self.get_factor_value(
                            factor_id, symbol, timeframe, use_cache=False
                        )
                        
                        if value is not None:
                            results['successful'] += 1
                        else:
                            results['failed'] += 1
                        
                        results['details'].append({
                            'factor_id': factor_id,
                            'symbol': symbol,
                            'timeframe': timeframe,
                            'value': value,
                            'status': 'success' if value is not None else 'failed'
                        })
                        
                    except Exception as e:
                        results['failed'] += 1
                        results['details'].append({
                            'factor_id': factor_id,
                            'symbol': symbol,
                            'timeframe': timeframe,
                            'error': str(e),
                            'status': 'error'
                        })
        
        results['end_time'] = datetime.now()
        results['duration_seconds'] = (results['end_time'] - results['start_time']).total_seconds()
        
        self.logger.info(f"因子预热完成: {results['successful']} 成功, {results['failed']} 失败")
        return results


# 全局交易API实例
trading_api = FactorTradingAPI()
