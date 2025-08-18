"""
因子API模块
提供因子挖掘的API接口
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from pathlib import Path
import json
from datetime import datetime

from ..core import DataLoader, FactorBuilder, FactorEvaluator, FactorOptimizer
from ..utils import save_results, load_results, create_summary_report


class FactorAPI:
    """
    因子API类
    提供因子挖掘的完整API接口
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化API
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.data_loader = DataLoader(config)
        self.factor_builder = FactorBuilder(config)
        self.evaluator = FactorEvaluator()
        self.optimizer = FactorOptimizer()
        
    def load_data(self, symbol: str, timeframe: str = '1h', 
                 start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict:
        """
        加载数据
        
        Args:
            symbol: 交易对
            timeframe: 时间框架
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            数据信息字典
        """
        try:
            data = self.data_loader.get_data(
                symbol=symbol,
                interval=timeframe,
                start_date=start_date,
                end_date=end_date,
                data_source='binance'
            )
            
            if data.empty:
                return {
                    'success': False,
                    'error': '数据加载失败或数据为空'
                }
            
            return {
                'success': True,
                'data': data,
                'info': {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'shape': data.shape,
                    'date_range': {
                        'start': data.index.min().strftime('%Y-%m-%d'),
                        'end': data.index.max().strftime('%Y-%m-%d')
                    }
                }
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def build_factors(self, data: pd.DataFrame, factor_types: Optional[List[str]] = None,
                     **kwargs) -> Dict:
        """
        构建因子
        
        Args:
            data: 市场数据
            factor_types: 因子类型列表
            **kwargs: 其他参数
            
        Returns:
            因子构建结果
        """
        try:
            factors_df = self.factor_builder.build_all_factors(
                data, factor_types=factor_types, **kwargs
            )
            
            return {
                'success': True,
                'factors': factors_df,
                'info': {
                    'total_factors': len(factors_df.columns),
                    'factor_names': list(factors_df.columns),
                    'data_points': len(factors_df)
                }
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def evaluate_factors(self, factors_df: pd.DataFrame, returns: pd.Series,
                        metrics: Optional[List[str]] = None) -> Dict:
        """
        评估因子
        
        Args:
            factors_df: 因子数据
            returns: 收益率数据
            metrics: 评估指标列表
            
        Returns:
            评估结果
        """
        try:
            if metrics is None:
                metrics = ['ic', 'ir', 'win_rate', 'effectiveness_score']
            
            evaluation_results = self.evaluator.evaluate_multiple_factors(
                factors_df, returns, metrics=metrics
            )
            
            return {
                'success': True,
                'evaluation': evaluation_results,
                'info': {
                    'total_factors': len(evaluation_results),
                    'metrics': metrics
                }
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def optimize_factors(self, factors_df: pd.DataFrame, returns: pd.Series,
                        method: str = 'greedy', max_factors: int = 20) -> Dict:
        """
        优化因子组合
        
        Args:
            factors_df: 因子数据
            returns: 收益率数据
            method: 优化方法
            max_factors: 最大因子数量
            
        Returns:
            优化结果
        """
        try:
            self.optimizer.set_data(None, returns)
            
            selected_factors, score = self.optimizer.optimize_factor_combination(
                factors_df, max_factors=max_factors, method=method
            )
            
            return {
                'success': True,
                'selected_factors': selected_factors,
                'score': score,
                'info': {
                    'method': method,
                    'max_factors': max_factors,
                    'selected_count': len(selected_factors)
                }
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def create_ensemble(self, factors_df: pd.DataFrame, returns: pd.Series,
                       method: str = 'ic_weight') -> Dict:
        """
        创建集成因子
        
        Args:
            factors_df: 因子数据
            returns: 收益率数据
            method: 集成方法
            
        Returns:
            集成结果
        """
        try:
            self.optimizer.set_data(None, returns)
            
            ensemble_factor = self.optimizer.create_ensemble_factor(factors_df, method=method)
            score = self.optimizer._calculate_metric(ensemble_factor, returns, 'effectiveness_score')
            
            return {
                'success': True,
                'ensemble_factor': ensemble_factor,
                'score': score,
                'info': {
                    'method': method,
                    'total_factors': len(factors_df.columns)
                }
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def run_complete_analysis(self, symbol: str, timeframe: str = '1h',
                            factor_types: Optional[List[str]] = None,
                            **kwargs) -> Dict:
        """
        运行完整的因子分析
        
        Args:
            symbol: 交易对
            timeframe: 时间框架
            factor_types: 因子类型列表
            **kwargs: 其他参数
            
        Returns:
            完整分析结果
        """
        try:
            # 1. 加载数据
            data_result = self.load_data(symbol, timeframe)
            if not data_result['success']:
                return data_result
            
            data = data_result['data']
            returns = data['close'].pct_change().shift(-1).dropna()
            
            # 2. 构建因子
            factors_result = self.build_factors(data, factor_types, **kwargs)
            if not factors_result['success']:
                return factors_result
            
            factors_df = factors_result['factors']
            
            # 3. 评估因子
            evaluation_result = self.evaluate_factors(factors_df, returns)
            if not evaluation_result['success']:
                return evaluation_result
            
            # 4. 优化因子组合
            optimization_result = self.optimize_factors(factors_df, returns)
            
            # 5. 创建集成因子
            ensemble_result = self.create_ensemble(factors_df, returns)
            
            # 6. 生成报告
            report = create_summary_report({
                '数据信息': data_result['info'],
                '因子信息': factors_result['info'],
                '评估信息': evaluation_result['info'],
                '优化结果': optimization_result.get('info', {}),
                '集成结果': ensemble_result.get('info', {})
            }, "因子挖掘完整分析报告")
            
            return {
                'success': True,
                'data_info': data_result['info'],
                'factors_info': factors_result['info'],
                'evaluation': evaluation_result['evaluation'],
                'optimization': optimization_result,
                'ensemble': ensemble_result,
                'report': report
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def save_analysis_results(self, results: Dict, filepath: str) -> Dict:
        """
        保存分析结果
        
        Args:
            results: 分析结果
            filepath: 文件路径
            
        Returns:
            保存结果
        """
        try:
            save_results(results, filepath, format='json')
            return {
                'success': True,
                'filepath': filepath
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_factor_info(self) -> Dict:
        """
        获取因子信息
        
        Returns:
            因子信息
        """
        try:
            factor_info = self.factor_builder.get_factor_info()
            return {
                'success': True,
                'factor_info': factor_info
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            } 