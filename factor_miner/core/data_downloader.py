"""
真实数据下载模块
使用CCXT接口下载交易所数据
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from pathlib import Path
import ccxt
from datetime import datetime, timedelta
import time
import logging

from config.user_config import config_manager
from .data_health_checker import health_checker
from .data_processor import data_processor


class DataDownloader:
    """数据下载器"""
    
    def __init__(self):
        """初始化数据下载器"""
        self.logger = logging.getLogger(__name__)
    
    def get_exchange_instance(self, config_id: int = None) -> Optional[ccxt.Exchange]:
        """
        获取交易所实例
        
        Args:
            config_id: 配置ID，如果为None则使用默认配置
            
        Returns:
            交易所实例
        """
        try:
            if config_id is None:
                # 使用默认的 Binance 配置
                exchange_class = getattr(ccxt, 'binance')
                exchange = exchange_class({
                    'enableRateLimit': True,
                    'timeout': 30000,
                    'headers': {
                        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
                    }
                })
                return exchange
            
            # 使用配置文件
            config = config_manager.get_exchange_config(config_id)
            if not config:
                self.logger.error(f"配置不存在: {config_id}")
                return None
            
            exchange_class = getattr(ccxt, config['exchange_id'])
            exchange = exchange_class({
                'apiKey': config['api_key'],
                'secret': config['secret'],
                'password': config['password'],
                'sandbox': config['sandbox'],
                'enableRateLimit': True
            })
            
            return exchange
        except Exception as e:
            self.logger.error(f"创建交易所实例失败: {e}")
            return None
    
    def download_ohlcv(self, config_id: int = None, symbol: str = None, timeframe: str = None, 
                      start_date: str = None, end_date: str = None, trade_type: str = None, progress_callback=None) -> Dict:
        """
        下载OHLCV数据
        
        Args:
            config_id: 交易所配置ID
            symbol: 交易对
            timeframe: 时间框架
            start_date: 开始日期
            end_date: 结束日期
            trade_type: 交易类型 (spot, futures, perpetual, delivery)
            progress_callback: 进度回调函数
            
        Returns:
            下载结果
        """
        try:
            # 设置交易类型属性
            self.trade_type = trade_type
            
            exchange = self.get_exchange_instance(config_id)
            if not exchange:
                return {'success': False, 'error': '无法创建交易所实例'}
            
            # 转换日期格式
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            # 获取时间框架的毫秒数
            timeframe_ms = exchange.parse_timeframe(timeframe) * 1000
            
            # 计算需要下载的数据点数量
            total_ms = (end_dt - start_dt).total_seconds() * 1000
            total_candles = int(total_ms / timeframe_ms)
            
            if progress_callback:
                progress_callback(0, f"开始下载 {symbol} {timeframe} 数据...")
            
            # 分批下载数据
            all_data = []
            current_dt = start_dt
            
            while current_dt < end_dt:
                try:
                    # 计算本次下载的结束时间
                    batch_end = min(current_dt + timedelta(days=30), end_dt)
                    
                    # 下载数据
                    ohlcv = exchange.fetch_ohlcv(
                        symbol, 
                        timeframe, 
                        int(current_dt.timestamp() * 1000),
                        limit=1000
                    )
                    
                    if ohlcv:
                        all_data.extend(ohlcv)
                    
                    # 更新进度
                    progress = min(100, int((current_dt - start_dt).total_seconds() / (end_dt - start_dt).total_seconds() * 100))
                    if progress_callback:
                        progress_callback(progress, f"已下载 {len(all_data)} 条数据...")
                    
                    # 移动到下一批
                    current_dt = batch_end
                    
                    # 限速
                    time.sleep(exchange.rateLimit / 1000)
                    
                except Exception as e:
                    self.logger.error(f"下载批次失败: {e}")
                    break
            
            if not all_data:
                return {'success': False, 'error': '没有下载到数据'}
            
            # 转换为DataFrame - 直接命名为 date，避免后续复杂操作
            df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('date', inplace=True)  # 设置 date 为索引
            df.drop('timestamp', axis=1, inplace=True)
            
            # 去重和排序（按 date 索引）
            df = df.drop_duplicates().sort_index()
            
            # 过滤日期范围（按 date 索引）
            df = df[(df.index >= start_date) & (df.index <= end_date)]
            
            # 数据健康度检查
            health_report = health_checker.check_data_health(df, timeframe, symbol)
            if not health_report['is_healthy']:
                self.logger.warning(f"数据健康度检查未通过: {health_report['summary']}")
                # 尝试修复数据问题
                df = self._fix_data_issues(df, health_report)
            
            # 最终去重处理 - 先重置索引，然后去重
            df_temp = df.reset_index()
            # 重命名 index 列为 date 列
            df_temp = df_temp.rename(columns={'index': 'date'})
            df_temp = data_processor.remove_duplicates(df_temp, 'date')
            
            # 数据已经重置索引，直接使用
            df_save = df_temp
            print("去重和重置索引完成")
            
            # 保存数据 - 使用与现有文件一致的命名格式
            # 例如：BTC_USDT_USDT-2h-futures.feather
            if hasattr(self, 'trade_type') and self.trade_type:
                if self.trade_type == 'futures':
                    filename = f"{symbol.replace('/', '_')}_USDT-{timeframe}-futures.feather"
                elif self.trade_type == 'spot':
                    filename = f"{symbol.replace('/', '_')}_USDT-{timeframe}-spot.feather"
                elif self.trade_type in ['perpetual', 'delivery']:
                    filename = f"{symbol.replace('/', '_')}_USDT-{timeframe}-{self.trade_type}.feather"
                else:
                    filename = f"{symbol}_{timeframe}_{start_date}_{end_date}.feather"
            else:
                filename = f"{symbol}_{timeframe}_{start_date}_{end_date}.feather"
            
            # 根据交易类型确定存储目录
            if hasattr(self, 'trade_type') and self.trade_type:
                if self.trade_type == 'futures':
                    # 确保不创建子目录，直接存储到 futures 目录
                    save_path = Path("data/binance/futures") / filename
                elif self.trade_type == 'spot':
                    save_path = Path("data/binance/spot") / filename
                elif self.trade_type in ['perpetual', 'delivery']:
                    save_path = Path(f"data/binance/{self.trade_type}") / filename
                else:
                    save_path = Path("data/binance") / filename
            else:
                # 默认存储到 binance 目录
                save_path = Path("data/binance") / filename
            
            # 检查现有文件以确定目标时区
            target_tz = None
            if save_path.exists():
                try:
                    existing_df = pd.read_feather(save_path)
                    if 'date' in existing_df.columns and existing_df['date'].dt.tz is not None:
                        target_tz = existing_df['date'].dt.tz
                        print(f"检测到现有文件时区: {target_tz}")
                except Exception as e:
                    print(f"读取现有文件失败，使用默认时区: {e}")
            
            # 统一时区
            if target_tz is not None:
                if df_save['date'].dt.tz is None:
                    # 新数据无时区，添加时区
                    df_save['date'] = df_save['date'].dt.tz_localize(target_tz)
                    print(f"为新数据添加时区: {target_tz}")
                elif df_save['date'].dt.tz != target_tz:
                    # 新数据时区不同，转换为相同时区
                    df_save['date'] = df_save['date'].dt.tz_convert(target_tz)
                    print(f"转换新数据时区到: {target_tz}")
            else:
                # 没有现有文件或现有文件无时区，确保新数据也无时区
                if df_save['date'].dt.tz is not None:
                    df_save['date'] = df_save['date'].dt.tz_localize(None)
                    print("移除新数据时区")
            
            # 检查是否存在同名文件，如果存在则合并数据（失败时保留源文件不变）
            merge_attempted = False
            merge_successful = False
            alt_saved = False
            alt_path = None
            if save_path.exists():
                merge_attempted = True
                try:
                    print(f"发现同名文件，尝试合并数据: {save_path}")
                    existing_df = pd.read_feather(save_path)
                    
                    # 确保两个数据框的 date 列都是 datetime 类型，并统一时区
                    target_tz = None
                    if 'date' in existing_df.columns:
                        if existing_df['date'].dtype == 'int64':
                            existing_df['date'] = pd.to_datetime(existing_df['date'], unit='ms')
                        elif existing_df['date'].dtype == 'int32':
                            existing_df['date'] = pd.to_datetime(existing_df['date'], unit='s')
                        
                        # 统一时区：如果现有数据有时区，新数据也使用相同时区
                        if hasattr(existing_df['date'], 'dt') and existing_df['date'].dt.tz is not None:
                            target_tz = existing_df['date'].dt.tz
                            print(f"现有数据时区: {target_tz}")
                        else:
                            print("现有数据无时区")
                    
                    if 'date' in df_save.columns:
                        if df_save['date'].dtype == 'int64':
                            df_save['date'] = pd.to_datetime(df_save['date'], unit='ms')
                        elif df_save['date'].dtype == 'int32':
                            df_save['date'] = pd.to_datetime(df_save['date'], unit='s')
                        
                        # 统一时区
                        if target_tz is not None:
                            if df_save['date'].dt.tz is None:
                                # 新数据无时区，添加时区
                                df_save['date'] = df_save['date'].dt.tz_localize(target_tz)
                                print(f"为新数据添加时区: {target_tz}")
                            elif df_save['date'].dt.tz != target_tz:
                                # 新数据时区不同，转换为相同时区
                                df_save['date'] = df_save['date'].dt.tz_convert(target_tz)
                                print(f"转换新数据时区到: {target_tz}")
                        else:
                            # 现有数据无时区，确保新数据也无时区
                            if df_save['date'].dt.tz is not None:
                                df_save['date'] = df_save['date'].dt.tz_localize(None)
                                print("移除新数据时区以匹配现有数据")
                    
                    # 合并数据，按 date 去重，保留最新的数据
                    combined_df = pd.concat([existing_df, df_save], ignore_index=True)
                    combined_df = combined_df.drop_duplicates(subset=['date'], keep='last').sort_values('date')
                    
                    print(f"合并完成：原数据 {len(existing_df)} 行，新数据 {len(df_save)} 行，合并后 {len(combined_df)} 行")
                    df_save = combined_df
                    merge_successful = True
                except Exception as e:
                    merge_successful = False
                    print(f"合并数据失败，源文件将保持不变，新数据将另存为新文件: {e}")
            
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 安全写入：
            # 1) 若合并成功，写入临时文件并原子替换源文件
            # 2) 若合并失败且存在源文件，则源文件不变，另存新文件
            # 3) 若不存在源文件，直接写入目标文件
            complete_message = None
            actual_path = save_path
            if save_path.exists() and merge_attempted and merge_successful:
                tmp_path = save_path.with_suffix('.tmp')
                df_save.to_feather(tmp_path)
                tmp_path.replace(save_path)
                complete_message = f"下载完成并合并！共 {len(df_save)} 条数据"
            elif save_path.exists() and merge_attempted and not merge_successful:
                # 生成另存文件名
                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                alt_path = save_path.with_name(f"{save_path.stem}-new-{ts}{save_path.suffix}")
                df_save.to_feather(alt_path)
                actual_path = alt_path
                alt_saved = True
                complete_message = f"下载完成，但合并失败，已另存为新文件（不影响原文件）。共 {len(df_save)} 条数据"
            else:
                # 不存在源文件，直接写入
                df_save.to_feather(save_path)
                complete_message = f"下载完成！共 {len(df_save)} 条数据"
            
            if progress_callback and complete_message:
                progress_callback(100, complete_message)
            
            # 确保返回的日期范围使用正确的日期列
            # 使用合并后的数据计算实际的日期范围
            start_date_str = end_date_str = "未知"
            if 'date' in df_save.columns:
                try:
                    if pd.api.types.is_datetime64_any_dtype(df_save['date']):
                        start_date_str = df_save['date'].min().strftime('%Y-%m-%d')
                        end_date_str = df_save['date'].max().strftime('%Y-%m-%d')
                    else:
                        df_save['date'] = pd.to_datetime(df_save['date'])
                        start_date_str = df_save['date'].min().strftime('%Y-%m-%d')
                        end_date_str = df_save['date'].max().strftime('%Y-%m-%d')
                except Exception as e:
                    print(f"处理日期范围时出错: {e}")
                    start_date_str = end_date_str = "处理失败"
            
            return {
                'success': True,
                'data': df_save,
                'filename': filename,
                'file_path': str(actual_path),
                'data_points': len(df_save),
                'date_range': {
                    'start': start_date_str,
                    'end': end_date_str
                }
            }
            
        except Exception as e:
            self.logger.error(f"下载数据失败: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_available_symbols(self, config_id: int) -> Dict:
        """
        获取可用的交易对
        
        Args:
            config_id: 交易所配置ID
            
        Returns:
            可用交易对信息
        """
        try:
            exchange = self.get_exchange_instance(config_id)
            if not exchange:
                return {'success': False, 'error': '无法创建交易所实例'}
            
            markets = exchange.load_markets()
            
            # 过滤USDT交易对
            usdt_symbols = [symbol for symbol in markets.keys() if symbol.endswith('/USDT')]
            
            return {
                'success': True,
                'exchange_name': exchange.name,
                'total_symbols': len(markets),
                'usdt_symbols': usdt_symbols[:50],  # 只返回前50个
                'all_symbols': list(markets.keys())[:100]  # 只返回前100个
            }
            
        except Exception as e:
            self.logger.error(f"获取交易对失败: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_timeframes(self, config_id: int) -> Dict:
        """
        获取支持的时间框架
        
        Args:
            config_id: 交易所配置ID
            
        Returns:
            支持的时间框架
        """
        try:
            exchange = self.get_exchange_instance(config_id)
            if not exchange:
                return {'success': False, 'error': '无法创建交易所实例'}
            
            timeframes = exchange.timeframes if hasattr(exchange, 'timeframes') else {}
            
            return {
                'success': True,
                'exchange_name': exchange.name,
                'timeframes': timeframes
            }
            
        except Exception as e:
            self.logger.error(f"获取时间框架失败: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_data_info(self, file_path: str) -> Dict:
        """
        获取数据文件信息
        
        Args:
            file_path: 数据文件路径
            
        Returns:
            数据信息
        """
        try:
            df = pd.read_feather(file_path)
            
            return {
                'success': True,
                'shape': df.shape,
                'columns': list(df.columns),
                'date_range': {
                    'start': df.index.min().strftime('%Y-%m-%d %H:%M:%S'),
                    'end': df.index.max().strftime('%Y-%m-%d %H:%M:%S')
                },
                'data_points': len(df),
                'file_size': Path(file_path).stat().st_size / 1024 / 1024,  # MB
                'memory_usage': df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
            }
            
        except Exception as e:
            self.logger.error(f"获取数据信息失败: {e}")
            return {'success': False, 'error': str(e)}
    
    def _fix_data_issues(self, df: pd.DataFrame, health_report: Dict) -> pd.DataFrame:
        """
        修复数据问题
        
        Args:
            df: 原始数据
            health_report: 健康度检查报告
            
        Returns:
            修复后的数据
        """
        try:
            df_fixed = df.copy()
            issues = health_report.get('issues', [])
            
            # 首先检查数据类型问题
            df_fixed = self._fix_data_types(df_fixed)
            
            for issue in issues:
                if 'OHLC数据逻辑错误' in issue:
                    # 修复OHLC逻辑错误
                    df_fixed = self._fix_ohlc_logic(df_fixed)
                elif '价格为0或负数' in issue:
                    # 修复价格问题
                    df_fixed = self._fix_price_issues(df_fixed)
                elif '成交量为负数' in issue:
                    # 修复成交量问题
                    df_fixed = self._fix_volume_issues(df_fixed)
            
            # 最终去重 - 修复：使用正确的列名
            if 'datetime' in df_fixed.columns:
                df_fixed = data_processor.remove_duplicates(df_fixed, 'datetime')
            elif 'date' in df_fixed.columns:
                df_fixed = data_processor.remove_duplicates(df_fixed, 'date')
            else:
                self.logger.warning("找不到时间列，跳过去重")
            
            self.logger.info(f"数据修复完成，原始数据 {len(df)} 条，修复后 {len(df_fixed)} 条")
            return df_fixed
            
        except Exception as e:
            self.logger.error(f"数据修复失败: {e}")
            return df
    
    def _fix_ohlc_logic(self, df: pd.DataFrame) -> pd.DataFrame:
        """修复OHLC逻辑错误"""
        try:
            df_fixed = df.copy()
            
            # 确保 high >= low
            df_fixed['high'] = df_fixed[['high', 'low']].max(axis=1)
            df_fixed['low'] = df_fixed[['high', 'low']].min(axis=1)
            
            # 确保 open 和 close 在 high 和 low 之间
            df_fixed['open'] = df_fixed['open'].clip(df_fixed['low'], df_fixed['high'])
            df_fixed['close'] = df_fixed['close'].clip(df_fixed['low'], df_fixed['high'])
            
            return df_fixed
        except Exception as e:
            self.logger.error(f"修复OHLC逻辑失败: {e}")
            return df
    
    def _fix_price_issues(self, df: pd.DataFrame) -> pd.DataFrame:
        """修复价格问题"""
        try:
            df_fixed = df.copy()
            
            # 将0或负数价格替换为前一个有效价格
            for col in ['open', 'high', 'low', 'close']:
                if col in df_fixed.columns:
                    df_fixed[col] = df_fixed[col].replace([0, -np.inf, np.inf], np.nan)
                    df_fixed[col] = df_fixed[col].fillna(method='ffill')
            
            return df_fixed
        except Exception as e:
            self.logger.error(f"修复价格问题失败: {e}")
            return df
    
    def _fix_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """修复数据类型问题"""
        try:
            df_fixed = df.copy()
            
            # 修复OHLC列的数据类型
            ohlc_columns = ['open', 'high', 'low', 'close']
            for col in ohlc_columns:
                if col in df_fixed.columns:
                    # 如果列是datetime类型但应该是数值，进行修复
                    if pd.api.types.is_datetime64_any_dtype(df_fixed[col]):
                        self.logger.warning(f"发现 {col} 列类型错误（应该是数值但实际是datetime），尝试修复...")
                        try:
                            # 尝试转换为数值类型
                            df_fixed[col] = pd.to_numeric(df_fixed[col], errors='coerce')
                            # 如果转换失败，用前一个有效值填充
                            if df_fixed[col].isna().all():
                                self.logger.error(f"无法修复 {col} 列，将使用前一个有效值")
                                df_fixed[col] = df_fixed[col].fillna(method='ffill')
                        except Exception as e:
                            self.logger.error(f"修复 {col} 列失败: {e}")
                            # 使用前一个有效值填充
                            df_fixed[col] = df_fixed[col].fillna(method='ffill')
            
            return df_fixed
            
        except Exception as e:
            self.logger.error(f"修复数据类型失败: {e}")
            return df
    
    def _fix_volume_issues(self, df: pd.DataFrame) -> pd.DataFrame:
        """修复成交量问题"""
        try:
            df_fixed = df.copy()
            
            if 'volume' in df_fixed.columns:
                # 将负数成交量替换为0
                df_fixed['volume'] = df_fixed['volume'].clip(lower=0)
            
            return df_fixed
        except Exception as e:
            self.logger.error(f"修复成交量问题失败: {e}")
            return df
    
    def list_downloaded_data(self) -> List[Dict]:
        """
        列出已下载的数据文件
        
        Returns:
            数据文件列表
        """
        try:
            data_dir = Path("data/binance")
            if not data_dir.exists():
                return []
            
            data_files = []
            for file_path in data_dir.glob("*.feather"):
                try:
                    info = self.get_data_info(str(file_path))
                    if info['success']:
                        data_files.append({
                            'filename': file_path.name,
                            'file_path': str(file_path),
                            'file_size': info['file_size'],
                            'data_points': info['data_points'],
                            'date_range': info['date_range']
                        })
                except Exception as e:
                    self.logger.error(f"读取文件信息失败 {file_path}: {e}")
            
            return sorted(data_files, key=lambda x: x['filename'])
            
        except Exception as e:
            self.logger.error(f"列出数据文件失败: {e}")
            return [] 