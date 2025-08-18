"""
辅助函数模块
包含各种工具函数
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from pathlib import Path
import json
import logging
from datetime import datetime

def setup_logging(log_file: str = None, level: str = 'INFO'):
    """
    设置日志
    
    Args:
        log_file: 日志文件路径
        level: 日志级别
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )

def save_results(results: Dict, filepath: str, format: str = 'json'):
    """
    保存结果
    
    Args:
        results: 结果字典
        filepath: 文件路径
        format: 文件格式
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'json':
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    elif format == 'csv':
        if isinstance(results, pd.DataFrame):
            results.to_csv(filepath)
        else:
            pd.DataFrame(results).to_csv(filepath)
    else:
        raise ValueError(f"不支持的文件格式: {format}")

def load_results(filepath: str, format: str = 'json'):
    """
    加载结果
    
    Args:
        filepath: 文件路径
        format: 文件格式
        
    Returns:
        加载的结果
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"文件不存在: {filepath}")
    
    if format == 'json':
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    elif format == 'csv':
        return pd.read_csv(filepath, index_col=0, parse_dates=True)
    else:
        raise ValueError(f"不支持的文件格式: {format}")

def calculate_returns(data: pd.DataFrame, method: str = 'pct_change', 
                     periods: int = 1) -> pd.Series:
    """
    计算收益率
    
    Args:
        data: 价格数据
        method: 计算方法 ('pct_change', 'log_return')
        periods: 周期数
        
    Returns:
        收益率序列
    """
    if method == 'pct_change':
        return data.pct_change(periods=periods)
    elif method == 'log_return':
        return np.log(data / data.shift(periods))
    else:
        raise ValueError(f"不支持的计算方法: {method}")

def align_data(data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    对齐数据
    
    Args:
        data_dict: 数据字典
        
    Returns:
        对齐后的数据字典
    """
    # 找到所有数据的公共索引
    common_index = None
    for name, data in data_dict.items():
        if common_index is None:
            common_index = data.index
        else:
            common_index = common_index.intersection(data.index)
    
    # 对齐所有数据
    aligned_data = {}
    for name, data in data_dict.items():
        aligned_data[name] = data.loc[common_index]
    
    return aligned_data

def validate_data(data: pd.DataFrame, required_columns: List[str] = None) -> bool:
    """
    验证数据
    
    Args:
        data: 数据DataFrame
        required_columns: 必需的列名
        
    Returns:
        是否有效
    """
    if data is None or data.empty:
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            print(f"缺少必需的列: {missing_columns}")
            return False
    
    return True

def get_data_info(data: pd.DataFrame) -> Dict:
    """
    获取数据信息
    
    Args:
        data: 数据DataFrame
        
    Returns:
        数据信息字典
    """
    return {
        'shape': data.shape,
        'columns': list(data.columns),
        'dtypes': data.dtypes.to_dict(),
        'missing_values': data.isnull().sum().to_dict(),
        'date_range': {
            'start': data.index.min(),
            'end': data.index.max(),
            'days': (data.index.max() - data.index.min()).days
        }
    }

def format_number(value: float, decimals: int = 4) -> str:
    """
    格式化数字
    
    Args:
        value: 数值
        decimals: 小数位数
        
    Returns:
        格式化后的字符串
    """
    if pd.isna(value):
        return 'N/A'
    return f"{value:.{decimals}f}"

def create_summary_report(results: Dict, title: str = "分析报告") -> str:
    """
    创建摘要报告
    
    Args:
        results: 结果字典
        title: 报告标题
        
    Returns:
        报告内容
    """
    report = []
    report.append("=" * 80)
    report.append(title)
    report.append("=" * 80)
    report.append(f"生成时间: {datetime.now()}")
    report.append("")
    
    for section, content in results.items():
        report.append(f"=== {section} ===")
        if isinstance(content, dict):
            for key, value in content.items():
                if isinstance(value, float):
                    report.append(f"{key}: {format_number(value)}")
                else:
                    report.append(f"{key}: {value}")
        elif isinstance(content, list):
            for item in content:
                report.append(f"- {item}")
        else:
            report.append(str(content))
        report.append("")
    
    return "\n".join(report) 