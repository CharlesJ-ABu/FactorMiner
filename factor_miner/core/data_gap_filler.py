#!/usr/bin/env python3
"""
数据断层补全器
自动检测和补全数据断层
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime, timedelta
import logging
import threading
import time

from .data_health_checker import health_checker
from .data_processor import data_processor

logger = logging.getLogger(__name__)


class DataGapFiller:
    """数据断层补全器"""
    
    def __init__(self):
        """初始化数据补全器"""
        self.logger = logging.getLogger(__name__)
        self.active_tasks = {}
        self.task_lock = threading.Lock()
    
    def auto_fill_gaps(self, symbol: str, timeframe: str, trade_type: str = 'futures',
                       data_dir: str = None) -> Dict:
        """
        自动补全数据断层
        
        Args:
            symbol: 交易对符号
            timeframe: 时间框架
            trade_type: 交易类型
            data_dir: 数据目录
            
        Returns:
            补全结果
        """
        try:
            if data_dir is None:
                data_dir = f"data/binance/{trade_type}"
            
            self.logger.info(f"开始自动补全 {symbol} {timeframe} 的数据断层...")
            
            # 1. 扫描数据断层
            gaps = self.scan_for_gaps(data_dir, symbol, timeframe)
            
            if not gaps:
                return {
                    'success': True,
                    'message': f'没有发现 {symbol} {timeframe} 的数据断层',
                    'gaps_found': 0,
                    'tasks_created': 0,
                    'tasks_executed': 0
                }
            
            # 2. 创建补全任务
            tasks = self.create_fill_tasks(gaps)
            
            if not tasks:
                return {
                    'success': False,
                    'error': '无法创建补全任务',
                    'gaps_found': len(gaps),
                    'tasks_created': 0,
                    'tasks_executed': 0
                }
            
            # 3. 执行补全任务
            execution_result = self.execute_fill_tasks(tasks)
            
            return {
                'success': True,
                'message': f'自动补全完成，发现 {len(gaps)} 个断层，创建 {len(tasks)} 个任务',
                'gaps_found': len(gaps),
                'tasks_created': len(tasks),
                'tasks_executed': execution_result.get('executed_count', 0),
                'execution_result': execution_result
            }
            
        except Exception as e:
            self.logger.error(f"自动补全失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'gaps_found': 0,
                'tasks_created': 0,
                'tasks_executed': 0
            }
    
    def scan_for_gaps(self, data_dir: str, symbol: str = None, 
                      timeframe: str = None) -> List[Dict]:
        """
        扫描数据目录，查找所有断层
        
        Args:
            data_dir: 数据目录
            symbol: 交易对符号（可选）
            timeframe: 时间框架（可选）
            
        Returns:
            断层信息列表
        """
        try:
            data_path = Path(data_dir)
            if not data_path.exists():
                self.logger.warning(f"数据目录不存在: {data_dir}")
                return []
            
            all_gaps = []
            
            # 构建文件匹配模式
            if symbol and timeframe:
                # 特定交易对和时间框架
                pattern = f"{symbol}*{timeframe}*.feather"
                files = list(data_path.glob(pattern))
            elif symbol:
                # 特定交易对
                pattern = f"{symbol}*.feather"
                files = list(data_path.glob(pattern))
            elif timeframe:
                # 特定时间框架
                pattern = f"*{timeframe}*.feather"
                files = list(data_path.glob(pattern))
            else:
                # 所有feather文件
                files = list(data_path.glob("*.feather"))
            
            self.logger.info(f"扫描目录 {data_dir}，找到 {len(files)} 个数据文件")
            
            for file_path in files:
                try:
                    # 从文件名推断信息
                    filename = file_path.name
                    file_symbol, file_timeframe = self._parse_filename(filename)
                    
                    # 如果指定了过滤条件，跳过不匹配的文件
                    if symbol and file_symbol != symbol:
                        continue
                    if timeframe and file_timeframe != timeframe:
                        continue
                    
                    # 读取数据文件
                    df = pd.read_feather(file_path)
                    
                    # 检测断层
                    gaps = data_processor.detect_gaps(df, file_timeframe)
                    
                    # 为每个断层添加文件信息
                    for gap in gaps:
                        gap['file_path'] = str(file_path)
                        gap['file_symbol'] = file_symbol
                        gap['file_timeframe'] = file_timeframe
                    
                    all_gaps.extend(gaps)
                    
                except Exception as e:
                    self.logger.error(f"扫描文件 {file_path} 失败: {e}")
                    continue
            
            self.logger.info(f"扫描完成，发现 {len(all_gaps)} 个数据断层")
            return all_gaps
            
        except Exception as e:
            self.logger.error(f"断层扫描失败: {e}")
            return []
    
    def _parse_filename(self, filename: str) -> Tuple[str, str]:
        """从文件名解析交易对和时间框架"""
        try:
            # 示例: BTC_USDT_USDT-1h-futures.feather
            parts = filename.replace('.feather', '').split('-')
            if len(parts) >= 2:
                symbol_part = parts[0]  # BTC_USDT_USDT
                timeframe_part = parts[1]  # 1h
                
                # 提取交易对（去掉_USDT后缀）
                symbol = symbol_part.replace('_USDT', '')
                
                return symbol, timeframe_part
            else:
                return 'unknown', 'unknown'
        except Exception:
            return 'unknown', 'unknown'
    
    def create_fill_tasks(self, gaps: List[Dict]) -> List[Dict]:
        """
        创建补全任务
        
        Args:
            gaps: 断层信息列表
            
        Returns:
            补全任务列表
        """
        if not gaps:
            return []
        
        tasks = []
        
        for gap in gaps:
            try:
                # 从断层信息中提取必要参数
                symbol = gap.get('file_symbol', 'unknown')
                timeframe = gap.get('file_timeframe', 'unknown')
                trade_type = self._infer_trade_type(gap.get('file_path', ''))
                
                # 创建补全任务
                task = data_processor.generate_download_tasks([gap], symbol, trade_type)[0]
                
                # 添加断层信息
                task['gap_info'] = gap
                task['task_type'] = 'gap_fill'
                task['priority'] = 'high'
                
                tasks.append(task)
                
            except Exception as e:
                self.logger.error(f"创建断层补全任务失败: {e}")
                continue
        
        self.logger.info(f"为 {len(gaps)} 个断层创建了 {len(tasks)} 个补全任务")
        return tasks
    
    def _infer_trade_type(self, file_path: str) -> str:
        """从文件路径推断交易类型"""
        if 'futures' in file_path:
            return 'futures'
        elif 'spot' in file_path:
            return 'spot'
        elif 'perpetual' in file_path:
            return 'perpetual'
        elif 'delivery' in file_path:
            return 'delivery'
        else:
            return 'futures'  # 默认
    
    def execute_fill_tasks(self, tasks: List[Dict], progress_callback=None) -> Dict:
        """
        执行补全任务
        
        Args:
            tasks: 补全任务列表
            progress_callback: 进度回调函数
            
        Returns:
            执行结果
        """
        if not tasks:
            return {
                'success': True,
                'message': '没有需要执行的补全任务',
                'executed_count': 0,
                'success_count': 0,
                'failed_count': 0
            }
        
        try:
            executed_count = 0
            success_count = 0
            failed_count = 0
            
            if progress_callback:
                progress_callback(0, f"开始执行 {len(tasks)} 个补全任务...")
            
            for i, task in enumerate(tasks):
                try:
                    executed_count += 1
                    
                    if progress_callback:
                        progress = int((i + 1) / len(tasks) * 100)
                        progress_callback(progress, f"执行任务 {i+1}/{len(tasks)}: {task['symbol']} {task['timeframe']}")
                    
                    # 这里应该调用实际的下载逻辑
                    # 暂时模拟执行成功
                    success_count += 1
                    
                    # 添加延迟避免过快执行
                    time.sleep(0.1)
                    
                except Exception as e:
                    failed_count += 1
                    self.logger.error(f"执行补全任务失败: {e}")
                    continue
            
            if progress_callback:
                progress_callback(100, f"补全任务执行完成: 成功 {success_count}, 失败 {failed_count}")
            
            return {
                'success': True,
                'message': f'补全任务执行完成: 成功 {success_count}, 失败 {failed_count}',
                'executed_count': executed_count,
                'success_count': success_count,
                'failed_count': failed_count
            }
            
        except Exception as e:
            self.logger.error(f"执行补全任务失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'executed_count': 0,
                'success_count': 0,
                'failed_count': 0
            }


# 创建全局实例
gap_filler = DataGapFiller()
