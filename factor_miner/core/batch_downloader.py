#!/usr/bin/env python3
"""
智能分批下载器
为不同时间框架优化分批下载策略
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import ccxt
from datetime import datetime, timedelta
import time
import logging
from dataclasses import dataclass

from .data_downloader import DataDownloader
from .data_health_checker import health_checker
from .data_processor import data_processor


@dataclass
class BatchConfig:
    """分批下载配置"""
    timeframe: str
    batch_days: int  # 每批下载的天数
    max_candles_per_batch: int  # 每批最大K线数量
    delay_seconds: float  # 批次间延迟秒数
    retry_attempts: int  # 重试次数


class SmartBatchDownloader(DataDownloader):
    """智能分批下载器"""
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        # 为不同时间框架配置分批策略
        self.batch_configs = {
            '1m': BatchConfig('1m', 1, 1000, 1.0, 3),      # 1分钟：每天一批，最多1000根K线
            '5m': BatchConfig('5m', 3, 1000, 0.8, 3),      # 5分钟：每3天一批
            '15m': BatchConfig('15m', 7, 1000, 0.6, 3),    # 15分钟：每周一批
            '30m': BatchConfig('30m', 14, 1000, 0.5, 3),   # 30分钟：每2周一批
            '1h': BatchConfig('1h', 30, 1000, 0.3, 3),     # 1小时：每月一批
            '4h': BatchConfig('4h', 90, 1000, 0.2, 3),     # 4小时：每3个月一批
            '1d': BatchConfig('1d', 365, 1000, 0.1, 3),    # 1天：每年一批
        }
    
    def get_batch_config(self, timeframe: str) -> BatchConfig:
        """获取时间框架的分批配置"""
        return self.batch_configs.get(timeframe, self.batch_configs['1h'])
    
    def calculate_optimal_batch_size(self, timeframe: str, start_date: str, end_date: str) -> Tuple[int, int]:
        """
        计算最优分批大小
        
        Args:
            timeframe: 时间框架
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            (每批天数, 总批次数)
        """
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        total_days = (end_dt - start_dt).days
        
        config = self.get_batch_config(timeframe)
        batch_days = config.batch_days
        
        # 根据总天数调整批次大小
        if total_days <= 7:
            batch_days = 1
        elif total_days <= 30:
            batch_days = 3
        elif total_days <= 90:
            batch_days = 7
        elif total_days <= 365:
            batch_days = 30
        else:
            batch_days = 90
        
        total_batches = (total_days + batch_days - 1) // batch_days
        return batch_days, total_batches
    
    def download_ohlcv_batch(self, config_id: int = None, symbol: str = None, 
                            timeframe: str = None, start_date: str = None, 
                            end_date: str = None, trade_type: str = None, 
                            progress_callback=None) -> Dict:
        """
        智能分批下载OHLCV数据
        
        Args:
            config_id: 交易所配置ID
            symbol: 交易对
            timeframe: 时间框架
            start_date: 开始日期
            end_date: 结束日期
            trade_type: 交易类型
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
            
            # 获取分批配置
            batch_days, total_batches = self.calculate_optimal_batch_size(timeframe, start_date, end_date)
            config = self.get_batch_config(timeframe)
            
            if progress_callback:
                progress_callback(0, f"开始分批下载 {symbol} {timeframe} 数据...")
                progress_callback(0, f"总批次数: {total_batches}, 每批天数: {batch_days}")
            
            # 分批下载数据
            all_data = []
            current_dt = start_dt
            batch_count = 0
            
            while current_dt < end_dt:
                try:
                    batch_count += 1
                    
                    # 计算本次下载的结束时间
                    batch_end = min(current_dt + timedelta(days=batch_days), end_dt)
                    
                    # 计算进度
                    progress = min(95, int((current_dt - start_dt).days / (end_dt - start_dt).days * 90))
                    
                    if progress_callback:
                        progress_callback(progress, f"下载第 {batch_count}/{total_batches} 批: "
                                        f"{current_dt.strftime('%Y-%m-%d')} 到 {batch_end.strftime('%Y-%m-%d')}")
                    
                    # 下载数据
                    ohlcv = exchange.fetch_ohlcv(
                        symbol, 
                        timeframe, 
                        int(current_dt.timestamp() * 1000),
                        limit=config.max_candles_per_batch
                    )
                    
                    if ohlcv:
                        all_data.extend(ohlcv)
                        if progress_callback:
                            progress_callback(progress + 5, f"第 {batch_count} 批完成，"
                                            f"当前总计: {len(all_data)} 条数据")
                    else:
                        self.logger.warning(f"第 {batch_count} 批没有数据")
                    
                    # 移动到下一批
                    current_dt = batch_end
                    
                    # 限速和延迟
                    time.sleep(config.delay_seconds)
                    
                    # 检查是否需要重试
                    if batch_count % 10 == 0:  # 每10批检查一次
                        time.sleep(config.delay_seconds * 2)  # 额外延迟
                    
                except Exception as e:
                    self.logger.error(f"第 {batch_count} 批下载失败: {e}")
                    
                    # 重试逻辑
                    retry_count = 0
                    while retry_count < config.retry_attempts:
                        try:
                            time.sleep(config.delay_seconds * 2)
                            retry_count += 1
                            
                            if progress_callback:
                                progress_callback(progress, f"第 {batch_count} 批重试 {retry_count}/{config.retry_attempts}")
                            
                            ohlcv = exchange.fetch_ohlcv(
                                symbol, 
                                timeframe, 
                                int(current_dt.timestamp() * 1000),
                                limit=config.max_candles_per_batch
                            )
                            
                            if ohlcv:
                                all_data.extend(ohlcv)
                                break
                            else:
                                self.logger.warning(f"第 {batch_count} 批重试 {retry_count} 次后仍无数据")
                                
                        except Exception as retry_e:
                            self.logger.error(f"第 {batch_count} 批重试 {retry_count} 失败: {retry_e}")
                    
                    # 如果重试失败，继续下一批
                    current_dt = batch_end
            
            if not all_data:
                return {'success': False, 'error': '没有下载到数据'}
            
            if progress_callback:
                progress_callback(95, f"数据下载完成，共 {len(all_data)} 条，正在处理...")
            
            # 转换为DataFrame - 直接命名为 date，避免后续复杂操作
            df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            # 转换时间戳并处理时区问题 - 直接命名为 date
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # 如果时间戳是UTC时间，转换为本地时间
            # 注意：CCXT返回的时间戳通常是UTC时间
            if df['date'].dt.tz is None:
                # 假设是UTC时间，转换为本地时间
                df['date'] = df['date'].dt.tz_localize('UTC').dt.tz_convert('Asia/Shanghai')
            
            df.set_index('date', inplace=True)  # 设置 date 为索引
            df.drop('timestamp', axis=1, inplace=True)
            
            print(f"原始下载数据: {len(df)} 条")
            print(f"数据时间范围: {df.index.min()} 到 {df.index.max()}")
            
            # 去重和排序（按 date 索引）
            df = df[~df.index.duplicated(keep='last')].sort_index()
            print(f"去重后数据: {len(df)} 条")
            
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
            
            if progress_callback:
                progress_callback(98, f"数据处理完成，准备保存...")
            
            # 保存数据
            save_result = self._save_data_with_merge(df_save, symbol, timeframe, start_date, end_date)
            
            if progress_callback:
                progress_callback(100, f"下载完成！{save_result.get('message', '数据已保存')}")
            
            return {
                'success': True,
                'data': df_save,
                'total_records': len(df_save),
                'timeframe': timeframe,
                'symbol': symbol,
                'start_date': start_date,
                'end_date': end_date,
                'batch_info': {
                    'total_batches': total_batches,
                    'batch_days': batch_days,
                    'actual_batches': batch_count
                },
                'message': save_result.get('message', '数据下载完成')
            }
            
        except Exception as e:
            error_msg = f"下载失败: {e}"
            self.logger.error(error_msg)
            if progress_callback:
                progress_callback(0, error_msg)
            return {'success': False, 'error': error_msg}
    
    def _save_data_with_merge(self, df_save: pd.DataFrame, symbol: str, timeframe: str, 
                             start_date: str, end_date: str) -> Dict:
        """保存数据并处理合并逻辑"""
        # 直接保存数据，避免重复调用
        try:
            # 构建文件名
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
            
            # 确定存储目录
            if hasattr(self, 'trade_type') and self.trade_type:
                if self.trade_type == 'futures':
                    save_path = Path("data/binance/futures") / filename
                elif self.trade_type == 'spot':
                    save_path = Path("data/binance/spot") / filename
                elif self.trade_type in ['perpetual', 'delivery']:
                    save_path = Path(f"data/binance/{self.trade_type}") / filename
                else:
                    save_path = Path("data/binance") / filename
            else:
                save_path = Path("data/binance") / filename
            
            # 检查现有文件并合并
            if save_path.exists():
                try:
                    existing_df = pd.read_feather(save_path)
                    
                    # 确保两个数据框的 date 列都是 datetime 类型
                    if 'date' in existing_df.columns:
                        if existing_df['date'].dtype == 'int64':
                            existing_df['date'] = pd.to_datetime(existing_df['date'], unit='ms')
                        elif existing_df['date'].dtype == 'int32':
                            existing_df['date'] = pd.to_datetime(existing_df['date'], unit='s')
                    
                    # 新数据已经有 date 列，无需处理
                    print("合并时：数据格式已正确")
                    
                    # 合并数据，按 date 去重，保留最新的数据
                    combined_df = pd.concat([existing_df, df_save], ignore_index=True)
                    combined_df = combined_df.drop_duplicates(subset=['date'], keep='last').sort_values('date')
                    
                    print(f"合并完成：原数据 {len(existing_df)} 行，新数据 {len(df_save)} 行，合并后 {len(combined_df)} 行")
                    df_save = combined_df
                    
                except Exception as e:
                    print(f"合并数据失败，将另存为新文件: {e}")
                    # 生成新文件名
                    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                    save_path = save_path.with_name(f"{save_path.stem}-new-{ts}{save_path.suffix}")
            
            # 保存数据
            save_path.parent.mkdir(parents=True, exist_ok=True)
            df_save.to_feather(save_path)
            
            return {
                'success': True,
                'message': f'数据保存成功，共 {len(df_save)} 条记录',
                'file_path': str(save_path)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'保存失败: {e}'
            }


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
                print("⚠️  警告：找不到时间列，跳过去重")
            
            print(f"🔧 数据修复完成，原始数据 {len(df)} 条，修复后 {len(df_fixed)} 条")
            return df_fixed
            
        except Exception as e:
            print(f"❌ 数据修复失败: {e}")
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
            print(f"❌ 修复OHLC逻辑失败: {e}")
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
            print(f"❌ 修复价格问题失败: {e}")
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
                        print(f"⚠️  发现 {col} 列类型错误（应该是数值但实际是datetime），尝试修复...")
                        try:
                            # 尝试转换为数值类型
                            df_fixed[col] = pd.to_numeric(df_fixed[col], errors='coerce')
                            # 如果转换失败，用前一个有效值填充
                            if df_fixed[col].isna().all():
                                print(f"❌ 无法修复 {col} 列，将使用前一个有效值")
                                df_fixed[col] = df_fixed[col].fillna(method='ffill')
                        except Exception as e:
                            print(f"❌ 修复 {col} 列失败: {e}")
                            # 使用前一个有效值填充
                            df_fixed[col] = df_fixed[col].fillna(method='ffill')
            
            return df_fixed
            
        except Exception as e:
            print(f"❌ 修复数据类型失败: {e}")
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
            print(f"❌ 修复成交量问题失败: {e}")
            return df


# 创建全局实例
batch_downloader = SmartBatchDownloader()
