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
import os

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
                # 检测环境变量中的代理配置
                http_proxy = os.getenv('HTTP_PROXY')
                https_proxy = os.getenv('HTTPS_PROXY')
                proxies = {}
                if http_proxy:
                    proxies['http'] = http_proxy
                if https_proxy:
                    proxies['https'] = https_proxy

                exchange = exchange_class({
                    'enableRateLimit': True,
                    'timeout': 30000,
                    'proxies': proxies if proxies else None,
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
        下载OHLCV数据 - 重构版本：确保数据完整性
        
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

            # 基础去重和排序
            df = df[~df.index.duplicated(keep='last')].sort_index()
            print(f"原始下载数据: {len(df)} 条")

            # 过滤日期范围
            df = df[(df.index >= start_date) & (df.index <= end_date)]
            print(f"过滤后数据: {len(df)} 条")

            # 阶段1: 与现有数据合并和验证
            df_merged = self._merge_with_existing_data(df, symbol, timeframe, start_date, end_date)

            # 阶段2: 检测和补全数据间断
            df_complete = self._fill_data_gaps(df_merged, symbol, timeframe, start_date, end_date, progress_callback)

            # 阶段3: 最终验证 - 只有100分才能保存
            if not self._final_validation(df_complete, timeframe, symbol):
                print("⚠️ 数据验证失败，开始自动修复...")
                df_complete = self._auto_fix_data_issues(df_complete, timeframe, symbol, max_retries=20)

                # 再次验证
                if not self._final_validation(df_complete, timeframe, symbol):
                    return {'success': False, 'error': '数据验证失败，自动修复后仍无法达到100分标准'}

                print("🎉 自动修复成功，数据达到100分标准！")

            # 准备保存数据
            print("=== 保存前数据检查 ===")
            print(f"df_complete 索引名: {df_complete.index.name}")
            print(f"df_complete 列名: {df_complete.columns.tolist()}")
            print(f"df_complete 形状: {df_complete.shape}")
            print(f"df_complete 数据类型:")
            print(df_complete.dtypes)
            print("========================")

            # 检查是否需要重置索引
            # if df_complete.index.name == 'date' and 'date' in df_complete.columns:
            #     print("⚠️ 检测到 date 索引和 date 列冲突，需要重置索引")
            #     df_save = df_complete.reset_index()
            # else:
            #     print("✅ 没有索引和列冲突，直接使用原数据")
            #     df_save = df_complete.copy()
            df_save = df_complete.copy()



            print(f"最终验证通过，准备保存: {len(df_save)} 条数据")

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

    def _merge_with_existing_data(self, new_df: pd.DataFrame, symbol: str, timeframe: str,
                                 start_date: str, end_date: str) -> pd.DataFrame:
        """
        与现有数据合并和验证
        
        Args:
            new_df: 新下载的数据
            symbol: 交易对
            timeframe: 时间框架
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            合并后的数据
        """
        try:
            # 构建文件路径
            if hasattr(self, 'trade_type') and self.trade_type:
                if self.trade_type == 'futures':
                    filename = f"{symbol.replace('/', '_')}_USDT-{timeframe}-futures.feather"
                    save_path = Path("data/binance/futures") / filename
                elif self.trade_type == 'spot':
                    filename = f"{symbol.replace('/', '_')}_USDT-{timeframe}-spot.feather"
                    save_path = Path("data/binance/spot") / filename
                else:
                    filename = f"{symbol}_{timeframe}_{start_date}_{end_date}.feather"
                    save_path = Path("data/binance") / filename
            else:
                filename = f"{symbol}_{timeframe}_{start_date}_{end_date}.feather"
                save_path = Path("data/binance") / filename

            # 检查是否存在现有文件
            if not save_path.exists():
                print(f"没有现有文件，直接使用新数据")
                return new_df

            print(f"发现现有文件: {save_path}")

            try:
                # 读取现有数据
                existing_df = pd.read_feather(save_path)
                print(f"现有数据: {len(existing_df)} 条")

                # 确保现有数据有date列
                if 'date' not in existing_df.columns:
                    print("现有数据没有date列，跳过合并")
                    return new_df

                # 转换时间列
                existing_df['date'] = pd.to_datetime(existing_df['date'])
                existing_df = existing_df.set_index('date')

                # 合并数据
                combined_df = pd.concat([existing_df, new_df], ignore_index=False)

                # 去重（保留最新的数据）- 按 index（时间点）去重
                combined_df = combined_df[~combined_df.index.duplicated(keep='last')].sort_index()

                print(f"合并完成: 现有 {len(existing_df)} 条 + 新 {len(new_df)} 条 = 合并后 {len(combined_df)} 条")

                return combined_df

            except Exception as e:
                print(f"读取现有文件失败: {e}，跳过合并")
                return new_df

        except Exception as e:
            print(f"合并现有数据失败: {e}")
            return new_df

    def _fill_data_gaps(self, df: pd.DataFrame, symbol: str, timeframe: str,
                        start_date: str, end_date: str, progress_callback=None) -> pd.DataFrame:
        """
        检测和补全数据间断
        
        Args:
            df: 输入数据
            symbol: 交易对
            timeframe: 时间框架
            start_date: 开始日期
            end_date: 结束日期
            progress_callback: 进度回调
            
        Returns:
            补全后的数据
        """
        try:
            if progress_callback:
                progress_callback(85, "检测数据间断...")

            # 检测间断
            gaps = self._detect_data_gaps(df, timeframe)

            if not gaps:
                print("没有发现数据间断")
                if progress_callback:
                    progress_callback(90, "数据完整，无需补全")
                return df

            print(f"发现 {len(gaps)} 个数据间断，开始补全...")

            if progress_callback:
                progress_callback(87, f"发现 {len(gaps)} 个间断，开始补全...")

            # 补全间断
            df_complete = self._download_missing_data(df, gaps, symbol, timeframe, progress_callback)

            if progress_callback:
                progress_callback(95, "数据间断补全完成")

            return df_complete

        except Exception as e:
            print(f"补全数据间断失败: {e}")
            return df

    def _detect_data_gaps(self, df: pd.DataFrame, timeframe: str) -> List[Dict]:
        """
        检测数据中的时间间断
        
        Args:
            df: 数据DataFrame
            timeframe: 时间框架
            
        Returns:
            间断信息列表
        """
        try:
            gaps = []

            # 确保数据格式标准化
            df_work = self._ensure_data_format(df)

            # 确保时区一致 - 移除时区信息
            if df_work.index.tz is not None:
                df_work.index = df_work.index.tz_localize(None)

            # 计算预期时间间隔
            timeframe_intervals = {
                '1m': pd.Timedelta('1 minute'),
                '3m': pd.Timedelta('3 minutes'),
                '5m': pd.Timedelta('5 minutes'),
                '15m': pd.Timedelta('15 minutes'),
                '30m': pd.Timedelta('30 minutes'),
                '1h': pd.Timedelta('1 hour'),
                '2h': pd.Timedelta('2 hours'),
                '4h': pd.Timedelta('4 hours'),
                '6h': pd.Timedelta('6 hours'),
                '8h': pd.Timedelta('8 hours'),
                '12h': pd.Timedelta('12 hours'),
                '1d': pd.Timedelta('1 day'),
            }

            expected_interval = timeframe_intervals.get(timeframe, pd.Timedelta('1 minute'))

            # 计算时间差
            time_diff = df_work.index.to_series().diff()
            # print('time_diff 完整内容:')
            # print(time_diff.to_string())

            # 检测大断层（超过预期间隔的2.5倍）
            gap_threshold = expected_interval * 1.5
            large_gaps = time_diff[time_diff > gap_threshold]

            for idx, gap in large_gaps.items():
                gap_start = idx - gap
                gap_end = idx

                gaps.append({
                    'start_time': gap_start,
                    'end_time': gap_end,
                    'duration': gap,
                    'expected_interval': expected_interval,
                    'missing_intervals': int(gap.total_seconds() / expected_interval.total_seconds())
                })

            print(f"检测到 {len(gaps)} 个数据间断")
            return gaps

        except Exception as e:
            print(f"检测数据间断失败: {e}")
            return []

    def _download_missing_data(self, df: pd.DataFrame, gaps: List[Dict], symbol: str,
                              timeframe: str, progress_callback=None) -> pd.DataFrame:
        """
        下载缺失的数据
        
        Args:
            df: 原始数据
            gaps: 间断信息
            symbol: 交易对
            timeframe: 时间框架
            progress_callback: 进度回调
            
        Returns:
            补全后的数据
        """
        try:
            if not gaps:
                return df

            exchange = self.get_exchange_instance()
            if not exchange:
                print("无法创建交易所实例，跳过数据补全")
                return df

            df_complete = df.copy()
            total_gaps = len(gaps)

            for i, gap in enumerate(gaps):
                if progress_callback:
                    progress = 87 + int((i + 1) / total_gaps * 8)
                    progress_callback(progress, f"补全间断 {i+1}/{total_gaps}...")

                try:
                    # 下载缺失数据 - 确保时区一致性
                    start_time = gap['start_time']
                    end_time = gap['end_time']

                    # 移除时区信息以避免比较错误
                    if hasattr(start_time, 'tz') and start_time.tz is not None:
                        start_time = start_time.tz_localize(None)
                    if hasattr(end_time, 'tz') and end_time.tz is not None:
                        end_time = end_time.tz_localize(None)

                    start_timestamp = int(start_time.timestamp() * 1000)
                    end_timestamp = int(end_time.timestamp() * 1000)

                    # 分批下载缺失数据
                    missing_data = []
                    current_ts = start_timestamp

                    while current_ts < end_timestamp:
                        # 计算剩余时间区间
                        remaining_time = end_timestamp - current_ts
                        timeframe_ms = exchange.parse_timeframe(timeframe) * 1000

                        # 计算理论上剩余的数据条数
                        remaining_candles = remaining_time // timeframe_ms

                        # 动态设置 limit，但不超过1000
                        dynamic_limit = min(remaining_candles, 1000)
                        #print(f"dynamic_limit: {dynamic_limit}")

                        ohlcv = exchange.fetch_ohlcv(
                            symbol,
                            timeframe,
                            current_ts,
                            limit=dynamic_limit
                        )

                        if not ohlcv:
                            print(f"没有数据，继续用当前的current_ts尝试下载")
                            # 如果没有数据，继续用当前的current_ts尝试下载
                            # 不要推进时间，因为可能这个时间点确实没有数据
                            continue
                        else:
                            # 如果有数据，使用最后一条数据的时间戳推进
                            missing_data.extend(ohlcv)
                            last_timestamp = ohlcv[-1][0]  # 最后一条数据的时间戳
                            #print(f"last_timestamp: {last_timestamp}")
                            current_ts = last_timestamp + timeframe_ms

                        # 限速
                        time.sleep(exchange.rateLimit / 1000)

                    if missing_data:
                        # 转换为DataFrame
                        missing_df = pd.DataFrame(missing_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                        missing_df['date'] = pd.to_datetime(missing_df['timestamp'], unit='ms')
                        missing_df = missing_df.set_index('date').drop('timestamp', axis=1)

                        # 在合并之前确保时区一致
                        # 如果原始数据有时区，先移除时区
                        if hasattr(df_complete.index, 'tz') and df_complete.index.tz is not None:
                            print('移除原始数据时区信息')
                            df_complete.index = df_complete.index.tz_localize(None)

                        # 确保新数据也无时区
                        if hasattr(missing_df.index, 'tz') and missing_df.index.tz is not None:
                            print('移除新数据时区信息')
                            missing_df.index = missing_df.index.tz_localize(None)

                        # 合并数据
                        df_complete = pd.concat([df_complete, missing_df], ignore_index=False)
                        print('df_complete 合并成功')

                        # 确保没有重复的 date 列
                        if 'date' in df_complete.columns and df_complete.index.name == 'date':
                            print('检测到重复的 date 列，移除列中的 date')
                            df_complete = df_complete.drop('date', axis=1)

                        # 去重前的详细统计
                        #print(f"去重前数据: {len(df_complete)} 条")
                        print(f"去重前重复时间点数量: {df_complete.index.duplicated().sum()} 条")
                        #print(f"去重前时间范围: {df_complete.index.min()} 到 {df_complete.index.max()}")

                        # 记录去重前的数据量
                        before_count = len(df_complete)

                        # 执行去重 - 按 index（时间点）去重，不是按列去重
                        df_complete = df_complete[~df_complete.index.duplicated(keep='last')].sort_index()

                        # 去重后的详细统计
                        #print(f"去重后数据: {len(df_complete)} 条")
                        print(f"去重后重复时间点数量: {df_complete.index.duplicated().sum()} 条")
                        #print(f"去重后时间范围: {df_complete.index.min()} 到 {df_complete.index.max()}")

                        # 计算实际移除的数量
                        removed_count = before_count - len(df_complete)
                        if removed_count > 0:
                            print(f"✅ 成功移除 {removed_count} 条重复数据")
                        else:
                            print("✅ 没有发现重复数据，无需移除")

                        # 验证去重是否成功
                        if df_complete.index.duplicated().sum() == 0:
                            print("✅ 去重验证成功，没有重复时间点")
                        else:
                            print(f"❌ 去重验证失败，仍有 {df_complete.index.duplicated().sum()} 个重复时间点")

                        print(f"补全间断 {i+1}: 添加 {len(missing_data)} 条数据")

                except Exception as e:
                    print(f"补全间断 {i+1} 失败: {e}")
                    continue


            print(f"数据补全完成，最终数据: {len(df_complete)} 条")
            return df_complete

        except Exception as e:
            print(f"下载缺失数据失败: {e}")
            return df

    def _final_validation(self, df: pd.DataFrame, timeframe: str, symbol: str) -> bool:
        """
        最终验证：确保数据完美，只有100分才能保存
        
        Args:
            df: 数据DataFrame
            timeframe: 时间框架
            symbol: 交易对
            
        Returns:
            是否通过验证
        """
        try:
            print("开始最终数据验证...")

            # 确保数据格式标准化
            df_check = self._ensure_data_format(df)

            # 1. 检查无重复时间点
            duplicate_count = df_check.index.duplicated().sum()
            if duplicate_count > 0:
                print(f"❌ 验证失败：发现 {duplicate_count} 个重复时间点")
                return False

            print("✅ 无重复时间点")

            # 2. 检查无时间间断
            gaps = self._detect_data_gaps(df_check, timeframe)
            if gaps:
                print(f"❌ 验证失败：仍有 {len(gaps)} 个数据间断")
                return False

            print("✅ 无时间间断")

            # 3. 检查覆盖率≥95%
            coverage = self._calculate_coverage(df_check, timeframe)
            if coverage < 95.0:
                print(f"❌ 验证失败：数据覆盖率 {coverage:.2f}% < 95%")
                return False

            print(f"✅ 数据覆盖率: {coverage:.2f}%")

            # 4. 健康度检查100分
            # 健康度检查器期望数据有date列，所以我们需要重置索引
            df_for_health_check = self._ensure_data_format_for_health_check(df_check)
            health_report = health_checker.check_data_health(df_for_health_check, timeframe, symbol)
            if not health_report['is_healthy']:
                print(f"❌ 验证失败：健康度检查未通过 - {health_report['summary']}")
                return False

            if health_report['health_score'] < 100.0:
                print(f"❌ 验证失败：健康度分数 {health_report['health_score']} < 100")
                return False

            print(f"✅ 健康度检查通过: {health_report['health_score']}分")
            print("🎉 最终验证通过！数据完美，可以保存")

            return True

        except Exception as e:
            print(f"最终验证失败: {e}")
            return False

    def _auto_fix_data_issues(self, df: pd.DataFrame, timeframe: str, symbol: str, max_retries: int = 20) -> pd.DataFrame:
        """
        自动修复数据问题，最多重试指定次数
        
        Args:
            df: 输入数据
            timeframe: 时间框架
            symbol: 交易对
            max_retries: 最大重试次数
            
        Returns:
            修复后的数据
        """
        print(f"🔧 开始自动修复数据问题，最大重试次数: {max_retries}")

        df_fixed = df.copy()

        for attempt in range(max_retries):
            print(f"\\n🔄 修复尝试 {attempt + 1}/{max_retries}")

            # 1. 自动去重
            df_fixed = self._remove_duplicates_auto(df_fixed)

            # 2. 检测和补全间断
            gaps = self._detect_data_gaps(df_fixed, timeframe)
            if gaps:
                print(f"发现 {len(gaps)} 个数据间断，开始补全...")
                df_fixed = self._download_missing_data(df_fixed, gaps, symbol, timeframe)
            else:
                print("没有发现数据间断")

            # 3. 验证修复结果
            if self._final_validation(df_fixed, timeframe, symbol):
                print(f"🎉 数据修复成功！在第 {attempt + 1} 次尝试后通过验证")
                return df_fixed
            else:
                print(f"⚠️ 第 {attempt + 1} 次修复后仍未通过验证，继续尝试...")

                # 如果还有重复，进行更彻底的清理
                if df_fixed.index.duplicated().sum() > 0:
                    print("检测到重复时间点，进行深度清理...")
                    df_fixed = self._deep_clean_duplicates(df_fixed)

                # 如果还有间断，尝试重新下载补全
                gaps = self._detect_data_gaps(df_fixed, timeframe)
                if gaps:
                    print("检测到数据间断，尝试重新下载补全...")
                    df_fixed = self._download_missing_data(df_fixed, gaps, symbol, timeframe)

        print(f"❌ 达到最大重试次数 {max_retries}，数据修复失败")
        return df_fixed

    def _ensure_data_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        确保数据既有date索引又有date列，不改变原始数据
        
        Args:
            df: 输入数据
            
        Returns:
            格式标准化的数据副本
        """
        try:
            df_work = df.copy()  # 创建副本，不修改原始数据

            # 如果只有索引，添加列
            if df_work.index.name == 'date' and 'date' not in df_work.columns:
                df_work['date'] = df_work.index
                print("✅ 数据格式标准化：添加date列")
            # 如果只有列，设置索引
            elif 'date' in df_work.columns and df_work.index.name != 'date':
                df_work = df_work.set_index('date')
                print("✅ 数据格式标准化：设置date索引")
            # 如果既有索引又有列，确保一致性
            elif df_work.index.name == 'date' and 'date' in df_work.columns:
                # 确保索引和列的值一致
                if not df_work.index.equals(df_work['date']):
                    df_work['date'] = df_work.index
                    print("✅ 数据格式标准化：同步date列和索引")
                else:
                    print("✅ 数据格式已标准化")
            else:
                print("⚠️ 数据格式异常，尝试修复...")
                # 尝试从索引创建date列
                if df_work.index.name == 'date':
                    df_work['date'] = df_work.index
                # 或者从列创建索引
                elif 'date' in df_work.columns:
                    df_work = df_work.set_index('date')
                else:
                    print("❌ 无法识别数据格式")
                    return df

            return df_work

        except Exception as e:
            print(f"数据格式标准化失败: {e}")
            return df

    def _ensure_data_format_for_health_check(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        为健康度检查准备数据格式（只有date列，没有date索引）
        
        Args:
            df: 输入数据
            
        Returns:
            适合健康度检查的数据格式
        """
        try:
            df_work = df.copy()

            # 如果有date索引，重置索引
            if df_work.index.name == 'date':
                # 如果同时有date列，先重命名索引
                if 'date' in df_work.columns:
                    df_work.index.name = 'date_index'
                    print("✅ 为健康度检查准备数据：重命名索引避免冲突")

                df_work = df_work.reset_index()
                print("✅ 为健康度检查准备数据：重置索引")

                # 如果现在有date_index列，将其重命名为date，并移除原来的date列
                if 'date_index' in df_work.columns and 'date' in df_work.columns:
                    df_work['date'] = df_work['date_index']
                    df_work = df_work.drop('date_index', axis=1)
                    print("✅ 为健康度检查准备数据：统一date列")

            # 确保有date列
            if 'date' not in df_work.columns:
                print("❌ 数据缺少date列，无法进行健康度检查")
                return df

            # 移除重复的date列（如果存在）
            if df_work.columns.duplicated().any():
                df_work = df_work.loc[:, ~df_work.columns.duplicated()]
                print("✅ 移除重复列")

            return df_work

        except Exception as e:
            print(f"为健康度检查准备数据失败: {e}")
            return df

    def _remove_duplicates_auto(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        自动去除重复数据
        
        Args:
            df: 输入数据
            
        Returns:
            去重后的数据
        """
        try:
            # 确保数据格式标准化
            df_work = self._ensure_data_format(df)

            # 统计重复情况
            duplicate_count = df_work.index.duplicated().sum()
            if duplicate_count == 0:
                print("✅ 没有发现重复时间点")
                return df_work

            print(f"🔍 发现 {duplicate_count} 个重复时间点，开始去重...")

            # 方法1: 保留最新的数据
            df_clean = df_work[~df_work.index.duplicated(keep='last')]
            print(f"方法1去重后: {len(df_clean)} 条 (保留最新)")

            # 方法2: 如果还有重复，尝试基于OHLCV的去重
            if df_clean.index.duplicated().sum() > 0:
                print("方法1仍有重复，尝试基于OHLCV的去重...")
                df_clean = self._remove_duplicates_by_ohlcv(df_clean)

            # 方法3: 如果还有重复，尝试基于时间窗口的去重
            if df_clean.index.duplicated().sum() > 0:
                print("方法2仍有重复，尝试基于时间窗口的去重...")
                df_clean = self._remove_duplicates_by_time_window(df_clean)

            final_duplicates = df_clean.index.duplicated().sum()
            print(f"去重完成，剩余重复: {final_duplicates} 个")

            return df_clean

        except Exception as e:
            print(f"自动去重失败: {e}")
            return df

    def _remove_duplicates_by_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        基于OHLCV数据去重
        
        Args:
            df: 输入数据
            
        Returns:
            去重后的数据
        """
        try:
            # 确保数据格式标准化
            df_work = self._ensure_data_format(df)

            # 重置索引，基于所有列去重
            df_temp = df_work.reset_index()

            # 基于时间和其他列去重
            df_clean = df_temp.drop_duplicates(subset=['date', 'open', 'high', 'low', 'close', 'volume'], keep='last')

            # 重新设置索引
            df_clean = df_clean.set_index('date')

            print(f"基于OHLCV去重后: {len(df_clean)} 条")
            return df_clean

        except Exception as e:
            print(f"基于OHLCV去重失败: {e}")
            return df

    def _remove_duplicates_by_time_window(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        基于时间窗口去重
        
        Args:
            df: 输入数据
            
        Returns:
            去重后的数据
        """
        try:
            # 确保数据格式标准化
            df_work = self._ensure_data_format(df)

            # 重置索引
            df_temp = df_work.reset_index()

            # 基于时间窗口去重（1秒内的数据视为重复）
            df_temp['date_rounded'] = df_temp['date'].dt.round('1S')
            df_clean = df_temp.drop_duplicates(subset=['date_rounded'], keep='last')

            # 移除辅助列，重新设置索引
            df_clean = df_clean.drop('date_rounded', axis=1).set_index('date')

            print(f"基于时间窗口去重后: {len(df_clean)} 条")
            return df_clean

        except Exception as e:
            print(f"基于时间窗口去重失败: {e}")
            return df

    def _deep_clean_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        深度清理重复数据
        
        Args:
            df: 输入数据
            
        Returns:
            清理后的数据
        """
        try:
            print("🔧 开始深度清理重复数据...")

            # 确保数据格式标准化
            df_work = self._ensure_data_format(df)

            # 重置索引
            df_temp = df_work.reset_index()

            # 1. 基于精确时间戳去重
            df_clean = df_temp.drop_duplicates(subset=['date'], keep='last')

            # 2. 基于时间窗口去重（1秒内）
            df_clean['date_rounded'] = df_clean['date'].dt.round('1S')
            df_clean = df_clean.drop_duplicates(subset=['date_rounded'], keep='last')

            # 3. 基于OHLCV去重
            df_clean = df_clean.drop_duplicates(subset=['date_rounded', 'open', 'high', 'low', 'close', 'volume'], keep='last')

            # 4. 移除辅助列，重新设置索引
            df_clean = df_clean.drop('date_rounded', axis=1).set_index('date')

            # 5. 排序
            df_clean = df_clean.sort_index()

            print(f"深度清理完成: {len(df_clean)} 条")
            return df_clean

        except Exception as e:
            print(f"深度清理失败: {e}")
            return df





    def _get_timeframe_interval(self, timeframe: str) -> pd.Timedelta:
        """
        获取时间框架对应的时间间隔
        
        Args:
            timeframe: 时间框架
            
        Returns:
            时间间隔
        """
        timeframe_intervals = {
            '1m': pd.Timedelta('1 minute'),
            '3m': pd.Timedelta('3 minutes'),
            '5m': pd.Timedelta('5 minutes'),
            '15m': pd.Timedelta('15 minutes'),
            '30m': pd.Timedelta('30 minutes'),
            '1h': pd.Timedelta('1 hour'),
            '2h': pd.Timedelta('2 hours'),
            '4h': pd.Timedelta('4 hours'),
            '6h': pd.Timedelta('6 hours'),
            '8h': pd.Timedelta('8 hours'),
            '12h': pd.Timedelta('12 hours'),
            '1d': pd.Timedelta('1 day'),
        }

        return timeframe_intervals.get(timeframe, pd.Timedelta('1 minute'))

    def _calculate_coverage(self, df: pd.DataFrame, timeframe: str) -> float:
        """
        计算数据覆盖率
        
        Args:
            df: 数据DataFrame
            timeframe: 时间框架
            
        Returns:
            覆盖率百分比
        """
        try:
            if df.empty:
                return 0.0

            # 确保数据格式标准化
            df_check = self._ensure_data_format(df)

            # 确保时区一致 - 移除时区信息
            if df_check.index.tz is not None:
                df_check.index = df_check.index.tz_localize(None)

            # 计算时间跨度
            time_span = df_check.index.max() - df_check.index.min()

            # 计算预期数据条数
            timeframe_intervals = {
                '1m': pd.Timedelta('1 minute'),
                '3m': pd.Timedelta('3 minutes'),
                '5m': pd.Timedelta('5 minutes'),
                '15m': pd.Timedelta('15 minutes'),
                '30m': pd.Timedelta('30 minutes'),
                '1h': pd.Timedelta('1 hour'),
                '2h': pd.Timedelta('2 hours'),
                '4h': pd.Timedelta('4 hours'),
                '6h': pd.Timedelta('6 hours'),
                '8h': pd.Timedelta('8 hours'),
                '12h': pd.Timedelta('12 hours'),
                '1d': pd.Timedelta('1 day'),
            }

            expected_interval = timeframe_intervals.get(timeframe, pd.Timedelta('1 minute'))
            expected_count = time_span.total_seconds() / expected_interval.total_seconds() + 1
            actual_count = len(df_check)

            coverage = (actual_count / expected_count * 100) if expected_count > 0 else 0
            return round(coverage, 2)

        except Exception as e:
            print(f"计算覆盖率失败: {e}")
            return 0.0

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
