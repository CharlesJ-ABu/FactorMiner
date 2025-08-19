#!/usr/bin/env python3
"""
数据处理器
处理数据去重、断层检测和补全任务生成
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class DataProcessor:
    """数据处理器"""
    
    def __init__(self):
        """初始化数据处理器"""
        self.logger = logging.getLogger(__name__)
        
        # 不同时间框架的预期间隔
        self.timeframe_intervals = {
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
    
    def remove_duplicates(self, df: pd.DataFrame, time_col: str = 'date') -> pd.DataFrame:
        """
        智能去重
        
        Args:
            df: 数据DataFrame
            time_col: 时间列名
            
        Returns:
            去重后的DataFrame
        """
        if df is None or df.empty:
            return df
        
        try:
            # 确保时间列存在
            if time_col not in df.columns:
                self.logger.warning(f"时间列 {time_col} 不存在，使用第一列作为时间列")
                time_col = df.columns[0]
            
            # 转换时间列
            df_clean = df.copy()
            df_clean[time_col] = pd.to_datetime(df_clean[time_col])
            
            # 1. 时间索引去重（保留最新的数据）
            df_clean = df_clean.drop_duplicates(subset=[time_col], keep='last')
            
            # 2. 数据内容去重（如果时间相同但内容不同，保留最新的）
            # 基于时间排序，然后去重
            df_clean = df_clean.sort_values(time_col).drop_duplicates(subset=[time_col], keep='last')
            
            # 3. 检查去重效果
            original_count = len(df)
            cleaned_count = len(df_clean)
            removed_count = original_count - cleaned_count
            
            if removed_count > 0:
                self.logger.info(f"去重完成: 原始 {original_count} 条 -> 清理后 {cleaned_count} 条，移除 {removed_count} 条重复数据")
            
            return df_clean
            
        except Exception as e:
            self.logger.error(f"去重失败: {e}")
            return df
    
    def detect_gaps(self, df: pd.DataFrame, timeframe: str, time_col: str = 'date') -> List[Dict]:
        """
        检测数据断层
        
        Args:
            df: 数据DataFrame
            timeframe: 时间框架
            time_col: 时间列名
            
        Returns:
            断层信息列表
        """
        if df is None or df.empty:
            return []
        
        try:
            # 确保时间列存在
            if time_col not in df.columns:
                self.logger.warning(f"时间列 {time_col} 不存在，无法检测断层")
                return []
            
            # 转换时间列并排序
            df_time = df.copy()
            df_time[time_col] = pd.to_datetime(df_time[time_col])
            df_time = df_time.sort_values(time_col)
            
            # 获取时间框架间隔
            expected_interval = self.timeframe_intervals.get(timeframe, pd.Timedelta('1 minute'))
            
            # 计算时间差
            time_diff = df_time[time_col].diff()
            
            # 定义断层阈值（超过预期间隔的3倍认为是断层）
            gap_threshold = expected_interval * 3
            
            # 检测大断层
            large_gaps = time_diff[time_diff > gap_threshold]
            
            gaps_info = []
            for idx, gap in large_gaps.items():
                # 获取断层前后的时间点
                gap_start = df_time[time_col].iloc[idx - 1] if idx > 0 else df_time[time_col].iloc[0]
                gap_end = df_time[time_col].iloc[idx]
                
                # 计算断层持续时间
                gap_duration = gap
                gap_hours = gap_duration.total_seconds() / 3600
                
                # 计算预期应该有的数据条数
                expected_records = int(gap_duration.total_seconds() / expected_interval.total_seconds())
                
                gaps_info.append({
                    'position': str(idx),
                    'gap_start': gap_start.isoformat(),
                    'gap_end': gap_end.isoformat(),
                    'gap_duration': str(gap_duration),
                    'gap_hours': round(gap_hours, 2),
                    'expected_interval': str(expected_interval),
                    'expected_records': expected_records,
                    'timeframe': timeframe
                })
            
            if gaps_info:
                self.logger.info(f"检测到 {len(gaps_info)} 个数据断层")
            
            return gaps_info
            
        except Exception as e:
            self.logger.error(f"断层检测失败: {e}")
            return []
    
    def generate_download_tasks(self, gaps: List[Dict], symbol: str, 
                               trade_type: str = 'futures') -> List[Dict]:
        """
        生成补全下载任务
        
        Args:
            gaps: 断层信息列表
            symbol: 交易对符号
            trade_type: 交易类型
            
        Returns:
            下载任务列表
        """
        if not gaps:
            return []
        
        download_tasks = []
        
        for gap in gaps:
            try:
                # 解析断层时间
                gap_start = datetime.fromisoformat(gap['gap_start'])
                gap_end = datetime.fromisoformat(gap['gap_end'])
                
                # 计算下载时间范围（稍微扩展一点，确保覆盖）
                buffer_hours = 2  # 前后各扩展2小时
                download_start = gap_start - timedelta(hours=buffer_hours)
                download_end = gap_end + timedelta(hours=buffer_hours)
                
                # 创建下载任务
                task = {
                    'task_id': f"fill_gap_{symbol}_{gap['timeframe']}_{gap_start.strftime('%Y%m%d_%H%M')}",
                    'symbol': symbol,
                    'timeframe': gap['timeframe'],
                    'trade_type': trade_type,
                    'start_date': download_start.strftime('%Y-%m-%d'),
                    'end_date': download_end.strftime('%Y-%m-%d'),
                    'gap_info': gap,
                    'priority': 'high',  # 断层补全任务优先级高
                    'created_at': datetime.now().isoformat()
                }
                
                download_tasks.append(task)
                
            except Exception as e:
                self.logger.error(f"生成断层补全任务失败: {e}")
                continue
        
        self.logger.info(f"为 {len(gaps)} 个断层生成了 {len(download_tasks)} 个下载任务")
        return download_tasks
    
    def merge_data(self, existing_df: pd.DataFrame, new_df: pd.DataFrame, 
                   time_col: str = 'date') -> Dict:
        """
        合并数据
        
        Args:
            existing_df: 现有数据
            new_df: 新数据
            time_col: 时间列名
            
        Returns:
            合并结果字典
        """
        try:
            if existing_df is None or existing_df.empty:
                return {
                    'success': True,
                    'merged_df': new_df,
                    'message': '没有现有数据，直接使用新数据',
                    'original_count': 0,
                    'new_count': len(new_df),
                    'merged_count': len(new_df)
                }
            
            if new_df is None or new_df.empty:
                return {
                    'success': True,
                    'merged_df': existing_df,
                    'message': '没有新数据，保持现有数据不变',
                    'original_count': len(existing_df),
                    'new_count': 0,
                    'merged_count': len(existing_df)
                }
            
            # 确保两个数据框都有时间列
            if time_col not in existing_df.columns or time_col not in new_df.columns:
                return {
                    'success': False,
                    'error': f'时间列 {time_col} 不存在',
                    'merged_df': None
                }
            
            # 转换时间列
            existing_clean = existing_df.copy()
            new_clean = new_df.copy()
            
            existing_clean[time_col] = pd.to_datetime(existing_clean[time_col])
            new_clean[time_col] = pd.to_datetime(new_clean[time_col])
            
            # 合并数据
            combined_df = pd.concat([existing_clean, new_clean], ignore_index=True)
            
            # 去重（保留最新的数据）
            merged_df = combined_df.drop_duplicates(subset=[time_col], keep='last')
            
            # 按时间排序
            merged_df = merged_df.sort_values(time_col)
            
            # 统计信息
            original_count = len(existing_df)
            new_count = len(new_df)
            merged_count = len(merged_df)
            removed_duplicates = original_count + new_count - merged_count
            
            message = f"合并完成: 原数据 {original_count} 条 + 新数据 {new_count} 条 = 合并后 {merged_count} 条"
            if removed_duplicates > 0:
                message += f"，移除重复 {removed_duplicates} 条"
            
            return {
                'success': True,
                'merged_df': merged_df,
                'message': message,
                'original_count': original_count,
                'new_count': new_count,
                'merged_count': merged_count,
                'removed_duplicates': removed_duplicates
            }
            
        except Exception as e:
            self.logger.error(f"数据合并失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'merged_df': None
            }


# 创建全局实例
data_processor = DataProcessor()
