"""
真实数据下载模块
使用CCXT接口下载交易所数据
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from pathlib import Path
import ccxt
from datetime import datetime, timedelta, timezone
import time
import logging
import os
import tempfile
import threading

from config.user_config import config_manager
from .health_checker import health_checker
from .processor import data_processor, from_epoch_milliseconds
from .naming import data_filename, data_path, parse_data_filename


# Shared by ordinary and batch downloads in this process. Atomic replacement
# alone cannot protect a read/merge/validate/write transaction from lost updates.
_file_update_locks = {}
_file_update_locks_guard = threading.Lock()


class DataGapFillError(RuntimeError):
    """Gap repair stopped with missing data; callers must not save a complete dataset."""


class DataDownloader:
    """数据下载器"""

    def __init__(self):
        """初始化数据下载器"""
        self.logger = logging.getLogger(__name__)
        self.exchange_id = "binance"

    def get_exchange_instance(self, config_id: int = None, exchange_id: str = None, trade_type: str = None) -> Optional[ccxt.Exchange]:
        """
        获取交易所实例
        
        Args:
            config_id: 配置ID，如果为None则使用默认配置
            exchange_id: 直接指定交易所名称，如 'binance'
            trade_type: 交易类型 (spot 或 futures)，用于初始化 CCXT 的 defaultType
            
        Returns:
            交易所实例
        """
        try:
            if config_id is None:
                # 使用指定的 exchange_id，或者默认的 Binance 配置
                target_exchange = exchange_id if exchange_id else 'binance'
                exchange_class = getattr(ccxt, target_exchange)
                # 检测环境变量中的代理配置
                http_proxy = os.getenv('HTTP_PROXY')
                https_proxy = os.getenv('HTTPS_PROXY')
                proxies = {}
                if http_proxy:
                    proxies['http'] = http_proxy
                if https_proxy:
                    proxies['https'] = https_proxy
                    
                # Setup CCXT options
                options = {}
                if trade_type:
                    options['defaultType'] = 'future' if trade_type == 'futures' else 'spot'

                exchange = exchange_class({
                    'enableRateLimit': True,
                    'timeout': 30000,
                    'proxies': proxies if proxies else None,
                    'options': options,
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
                'enableRateLimit': True,
                'options': {'defaultType': 'future' if trade_type == 'futures' else 'spot'} if trade_type else {},
            })

            return exchange
        except Exception as e:
            self.logger.error(f"创建交易所实例失败: {e}")
            return None

    def download_ohlcv(self, config_id: int = None, symbol: str = None, timeframe: str = None,
                      start_date: str = None, end_date: str = None, trade_type: str = None,
                      progress_callback=None, exchange_id: str = None) -> Dict:
        """
        下载OHLCV数据 - 重构版本：确保数据完整性
        
        Args:
            config_id: 交易所配置ID
            symbol: 交易对
            timeframe: 时间框架
            start_date: UTC 开始日期或时间戳（包含）
            end_date: UTC 结束日期或时间戳（不包含）
            trade_type: 交易类型 (spot, futures, perpetual, delivery)
            progress_callback: 进度回调函数
            exchange_id: 直接指定交易所；未提供时沿用默认配置
            
        Returns:
            下载结果
        """
        try:
            # 设置交易类型属性
            self.trade_type = trade_type
            filename = data_filename(symbol, timeframe, trade_type)

            exchange = self.get_exchange_instance(config_id, exchange_id=exchange_id, trade_type=trade_type)
            if not exchange:
                return {'success': False, 'error': '无法创建交易所实例'}
            self.exchange_id = exchange.id
            # Request-local identity survives concurrent calls on one instance.
            save_path = data_path("data", exchange.id, symbol, timeframe, trade_type).resolve()

            # 转换日期格式
            start_dt = pd.to_datetime(start_date, utc=True)
            end_dt = pd.to_datetime(end_date, utc=True)

            if end_dt <= start_dt:
                raise ValueError("end_date must be later than start_date (UTC, end exclusive).")
            interval = pd.Timedelta(seconds=exchange.parse_timeframe(timeframe))
            first_bar = start_dt.ceil(interval)
            if progress_callback:
                progress_callback(0, f"开始下载 {symbol} {timeframe} 数据...")
            all_data = self._fetch_ohlcv_range(
                exchange, symbol, timeframe, first_bar.value // 1_000_000,
                (end_dt.value + 999_999) // 1_000_000,
            )

            if not all_data:
                return {'success': False, 'error': '没有下载到数据'}

            # 转换为DataFrame - 直接命名为 date，避免后续复杂操作
            df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['date'] = from_epoch_milliseconds(df['timestamp'])
            df = df.set_index('date')  # 设置 date 为索引
            df = df.drop(columns='timestamp')

            # 基础去重和排序
            df = df[~df.index.duplicated(keep='last')].sort_index()
            print(f"原始下载数据: {len(df)} 条")

            # 过滤日期范围
            df = df[(df.index >= start_dt.tz_localize(None)) & (df.index < end_dt.tz_localize(None))]
            print(f"过滤后数据: {len(df)} 条")

            with self._target_update_lock(save_path):
                # 阶段1: 与现有数据合并和验证
                df_merged = self._merge_with_existing_data(
                    df, symbol, timeframe, start_date, end_date, save_path=save_path)

                # 阶段2: 检测和补全数据间断
                df_complete = self._fill_data_gaps(df_merged, symbol, timeframe, start_date, end_date, progress_callback, exchange=exchange)
                self._validate_requested_range(df_complete, timeframe, start_date, end_date, exchange)

                # 阶段3: 最终验证 - 只有100分才能保存
                if not self._final_validation(df_complete, timeframe, symbol):
                    print("⚠️ 数据验证失败，开始自动修复...")
                    df_complete = self._auto_fix_data_issues(df_complete, timeframe, symbol, max_retries=20, exchange=exchange)

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
                if df_complete.index.name == 'date':
                    print("⚠️ 检测到 date 是索引，重置索引为普通列以供后续时区处理和保存")
                    df_save = df_complete.reset_index()
                else:
                    df_save = df_complete.copy()



                print(f"最终验证通过，准备保存: {len(df_save)} 条数据")

                # Persist UTC instants; never attach a historical file's timezone
                # directly to naive UTC candles, which would shift their meaning.
                df_save['date'] = pd.to_datetime(df_save['date'], utc=True).dt.tz_localize(None)

                # Existing rows were merged before validation. Publish exactly that
                # validated frame, without reintroducing unchecked data afterward.
                self._write_feather_atomic(df_save, save_path)
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
                'file_path': str(save_path),
                'data_points': len(df_save),
                'date_range': {
                    'start': start_date_str,
                    'end': end_date_str
                }
            }

        except Exception as e:
            self.logger.error(f"下载数据失败: {e}")
            return {'success': False, 'error': str(e)}

    @staticmethod
    def _target_update_lock(save_path: Path):
        """Serialize this process's updates to one canonical target only."""
        canonical_path = save_path.resolve()
        with _file_update_locks_guard:
            if canonical_path not in _file_update_locks:
                _file_update_locks[canonical_path] = threading.RLock()
            return _file_update_locks[canonical_path]

    @staticmethod
    def _write_feather_atomic(df: pd.DataFrame, save_path: Path) -> None:
        """Publish only a complete Feather file, preserving the target on failure."""
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=save_path.parent, suffix='.feather', delete=False) as handle:
            temporary_path = Path(handle.name)
        try:
            df.to_feather(temporary_path)
            temporary_path.replace(save_path)
        finally:
            temporary_path.unlink(missing_ok=True)

    def _merge_with_existing_data(self, new_df: pd.DataFrame, symbol: str, timeframe: str,
                                 start_date: str, end_date: str, save_path: Optional[Path] = None) -> pd.DataFrame:
        """Normalize existing timestamps and merge before the health gate.

        Unreadable or unrecognized existing data must stop the download; it must
        not be silently skipped and then overwritten with only the new range.
        """
        if save_path is None:
            save_path = data_path("data", self.exchange_id, symbol, timeframe, self.trade_type)
        if not save_path.exists():
            return new_df

        existing_df = pd.read_feather(save_path)
        # Match RealDataClient's column precedence and timestamp-ms semantics.
        if 'date' in existing_df.columns:
            dates = pd.to_datetime(existing_df['date'], utc=True)
        elif 'time' in existing_df.columns:
            dates = pd.to_datetime(existing_df['time'], utc=True)
        elif 'timestamp' in existing_df.columns:
            dates = from_epoch_milliseconds(existing_df['timestamp'])
        else:
            raise ValueError(f"Existing data has no date, time or timestamp column: {save_path}")
        if dates.isna().any():
            raise ValueError(f"Existing data contains missing timestamps: {save_path}")

        existing_df = existing_df.drop(columns=['time', 'timestamp'], errors='ignore')
        existing_df['date'] = dates.dt.tz_localize(None)
        existing_df = existing_df.set_index('date')
        new_df = new_df.copy()
        new_df.index = pd.to_datetime(new_df.index, utc=True).tz_localize(None)
        combined_df = pd.concat([existing_df, new_df], ignore_index=False)
        return combined_df[~combined_df.index.duplicated(keep='last')].sort_index()

    def _fill_data_gaps(self, df: pd.DataFrame, symbol: str, timeframe: str,
                        start_date: str, end_date: str, progress_callback=None, exchange=None) -> pd.DataFrame:
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
            df_complete = self._download_missing_data(df, gaps, symbol, timeframe, progress_callback, exchange=exchange)

            if progress_callback:
                progress_callback(95, "数据间断补全完成")

            return df_complete

        except DataGapFillError:
            raise
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

            # Naive timestamps in stored OHLCV represent UTC; preserve instants
            # when the input has a timezone, including across DST transitions.
            df_work.index = pd.to_datetime(df_work.index, utc=True).tz_localize(None)

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
                gap_start = idx - gap + expected_interval
                gap_end = idx

                gaps.append({
                    'start_time': gap_start,
                    'end_time': gap_end,
                    'duration': gap,
                    'expected_interval': expected_interval,
                    'missing_intervals': int(gap.total_seconds() / expected_interval.total_seconds()) - 1
                })

            print(f"检测到 {len(gaps)} 个数据间断")
            return gaps

        except Exception as e:
            print(f"检测数据间断失败: {e}")
            return []

    def _fetch_ohlcv_range(self, exchange, symbol: str, timeframe: str,
                          start_ms: int, end_ms: int, limit: int = 1000,
                          retry_attempts: int = 3, delay_seconds: float = 0,
                          progress_callback=None) -> list:
        """Page over [start, end), retrying only a bounded number of stalled calls."""
        interval = int(exchange.parse_timeframe(timeframe) * 1000)
        if interval <= 0 or end_ms <= start_ms:
            raise DataGapFillError(f"Invalid requested interval for {symbol} {timeframe}")
        cursor, stalled, records = start_ms, 0, []
        while cursor < end_ms:
            remaining = (end_ms - cursor + interval - 1) // interval
            cause = None
            try:
                response = exchange.fetch_ohlcv(symbol, timeframe, cursor, limit=min(remaining, limit))
                rows = [row for row in (response or []) if cursor <= row[0] < end_ms]
            except Exception as exc:
                rows, cause = [], exc
            if rows:
                records.extend(rows)
                cursor = max(row[0] for row in rows) + interval
                stalled = 0
            else:
                stalled += 1
                message = (
                    f"Unresolved gap for {symbol} {timeframe}: UTC ms [{cursor}, {end_ms}); "
                    f"no advancing data ({stalled}/{retry_attempts} attempts)."
                )
                if cause is not None:
                    message += f" {type(cause).__name__}: {cause}"
                self.logger.warning(message)
                if progress_callback:
                    progress_callback(87, message)
                if stalled >= retry_attempts:
                    raise DataGapFillError(message) from cause
            if cursor < end_ms:
                time.sleep(max(float(exchange.rateLimit) / 1000, delay_seconds))
        return records

    def _validate_requested_range(self, df: pd.DataFrame, timeframe: str,
                                  start_date: str, end_date: str, exchange) -> None:
        """A contiguous subset must not be reported as a complete requested interval."""
        start = pd.to_datetime(start_date, utc=True).tz_localize(None)
        end = pd.to_datetime(end_date, utc=True).tz_localize(None)
        interval = pd.Timedelta(seconds=exchange.parse_timeframe(timeframe))
        if interval <= pd.Timedelta(0) or end <= start:
            raise DataGapFillError("Requested UTC interval must be non-empty.")
        first_bar = start.ceil(interval)
        index = pd.to_datetime(df.index, utc=True).tz_localize(None).unique()
        index = index[(index >= start) & (index < end)]
        aligned = index[(index - first_bar) % interval == pd.Timedelta(0)]
        expected = max(0, (end - first_bar + interval - pd.Timedelta(1, unit='ns')) // interval)
        if len(aligned) != expected or len(aligned) != len(index):
            coverage = len(aligned) / expected if expected else 0
            raise DataGapFillError(
                f"Requested interval incomplete: {len(aligned)}/{expected} bars "
                f"({coverage:.2%}) in UTC [{start}, {end}) for {timeframe}."
            )

    def _download_missing_data(self, df: pd.DataFrame, gaps: List[Dict], symbol: str,
                              timeframe: str, progress_callback=None, exchange=None) -> pd.DataFrame:
        """Repair UTC gaps with the original exchange; stop after 3 stalled responses.

        Gap intervals are [first missing bar, next existing bar). Incomplete
        repair raises DataGapFillError so download callers return failure before
        saving. The existing file and the unresolved gap remain intact.
        """
        if not gaps:
            return df
        if exchange is None:
            raise DataGapFillError("Gap repair requires the original download exchange instance.")

        df_complete = self._ensure_data_format(df).drop(columns=['date'], errors='ignore')
        df_complete.index = pd.to_datetime(df_complete.index, utc=True).tz_localize(None)
        for i, gap in enumerate(gaps):
            start_timestamp = int(pd.to_datetime(gap['start_time'], utc=True).timestamp() * 1000)
            end_timestamp = int(pd.to_datetime(gap['end_time'], utc=True).timestamp() * 1000)
            if progress_callback:
                progress_callback(87 + int(i / len(gaps) * 8), f"补全间断 {i+1}/{len(gaps)}...")
            missing_data = self._fetch_ohlcv_range(
                exchange, symbol, timeframe, start_timestamp, end_timestamp,
                progress_callback=progress_callback,
            )

            if missing_data:
                missing_df = pd.DataFrame(missing_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                missing_df['date'] = from_epoch_milliseconds(missing_df['timestamp'])
                missing_df = missing_df.set_index('date').drop(columns='timestamp')
                df_complete = pd.concat([df_complete, missing_df])
                # Gap repair must not overwrite a bar already present in the source.
                df_complete = df_complete[~df_complete.index.duplicated(keep='first')].sort_index()

        remaining_gaps = self._detect_data_gaps(df_complete, timeframe)
        if remaining_gaps:
            raise DataGapFillError(f"Unresolved gaps for {symbol} {timeframe}: {remaining_gaps}")
        return df_complete

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

    def _auto_fix_data_issues(self, df: pd.DataFrame, timeframe: str, symbol: str, max_retries: int = 20, exchange=None) -> pd.DataFrame:
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
                df_fixed = self._download_missing_data(df_fixed, gaps, symbol, timeframe, exchange=exchange)
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
                    df_fixed = self._download_missing_data(df_fixed, gaps, symbol, timeframe, exchange=exchange)

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
                    df_fixed[col] = df_fixed[col].ffill()

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
                                df_fixed[col] = df_fixed[col].ffill()
                        except Exception as e:
                            self.logger.error(f"修复 {col} 列失败: {e}")
                            # 使用前一个有效值填充
                            df_fixed[col] = df_fixed[col].ffill()

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
            for file_path in data_dir.rglob("*.feather"):
                try:
                    parse_data_filename(file_path.name)
                    info = self.get_data_info(str(file_path))
                    if info['success']:
                        data_files.append({
                            'filename': file_path.name,
                            'file_path': str(file_path),
                            'file_size': info['file_size'],
                            'data_points': info['data_points'],
                            'date_range': info['date_range']
                        })
                except ValueError:
                    self.logger.debug(f"跳过非标准命名数据文件: {file_path}")
                except Exception as e:
                    self.logger.error(f"读取文件信息失败 {file_path}: {e}")

            return sorted(data_files, key=lambda x: x['filename'])

        except Exception as e:
            self.logger.error(f"列出数据文件失败: {e}")
            return []
