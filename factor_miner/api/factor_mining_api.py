"""
因子挖掘API模块
提供完整的因子挖掘、评估和优化功能
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from ..core import DataLoader, FactorBuilder, FactorEvaluator, FactorOptimizer
from ..core.factor_storage import TransparentFactorStorage
from ..core.factor_engine import FactorEngine
from ..utils import save_results, load_results, create_summary_report


class FactorMiningAPI:
    """
    因子挖掘API类
    提供完整的因子挖掘、评估和优化功能
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
        self.storage = TransparentFactorStorage()
        self.engine = FactorEngine()
        
    def run_complete_mining(self, 
                           symbol: str, 
                           timeframe: str = '1h',
                           factor_types: Optional[List[str]] = None,
                           start_date: Optional[str] = None, 
                           end_date: Optional[str] = None,
                           mining_config: Optional[Dict] = None) -> Dict:
        """
        运行完整的因子挖掘流程
        
        Args:
            symbol: 交易对
            timeframe: 时间框架
            factor_types: 因子类型列表
            start_date: 开始日期
            end_date: 结束日期
            mining_config: 挖掘配置
            
        Returns:
            挖掘结果字典
        """
        print("🚀 开始因子挖掘流程...")
        print(f"交易对: {symbol}, 时间框架: {timeframe}")
        print(f"因子类型: {factor_types or '全部'}")
        print(f"时间范围: {start_date} 到 {end_date}")
        print()
        
        try:
            # 1. 数据加载
            print("📊 步骤1: 加载市场数据...")
            data_result = self.load_data(symbol, timeframe, start_date, end_date)
            if not data_result['success']:
                return {
                    'success': False,
                    'error': f"数据加载失败: {data_result['error']}"
                }
            
            data = data_result['data']
            print(f"✅ 数据加载成功: {data.shape[0]} 条记录")
            
            # 2. 因子构建
            print("\n🔧 步骤2: 构建因子...")
            factors_result = self.build_factors(data, factor_types, mining_config)
            if not factors_result['success']:
                return {
                    'success': False,
                    'error': f"因子构建失败: {factors_result['error']}"
                }
            
            factors_df = factors_result['factors']
            print(f"✅ 因子构建成功: {factors_df.shape[1]} 个因子")
            
            # 3. 因子评估
            print("\n📈 步骤3: 评估因子...")
            evaluation_result = self.evaluate_factors(factors_df, data, mining_config)
            if not evaluation_result['success']:
                return {
                    'success': False,
                    'error': f"因子评估失败: {evaluation_result['error']}"
                }
            
            evaluation = evaluation_result['evaluation']
            print(f"✅ 因子评估完成: {len(evaluation)} 个因子已评估")
            
            # 4. 因子优化
            print("\n⚡ 步骤4: 优化因子组合...")
            optimization_result = self.optimize_factor_combination(factors_df, data, mining_config)
            if not optimization_result['success']:
                print(f"⚠️ 因子优化失败: {optimization_result['error']}")
                optimization_result = {'success': False, 'selected_factors': []}
            
            # 5. 生成报告
            print("\n📋 步骤5: 生成挖掘报告...")
            report = self.generate_mining_report(
                symbol, timeframe, data, factors_df, 
                evaluation, optimization_result, mining_config
            )
            
            # 6. 保存结果
            print("\n💾 步骤6: 保存挖掘结果...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results = {
                'success': True,
                'mining_info': {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'start_date': start_date,
                    'end_date': end_date,
                    'factor_types': factor_types,
                    'mining_config': mining_config,
                    'timestamp': timestamp
                },
                'data_info': {
                    'shape': data.shape,
                    'date_range': {
                        'start': data.index.min().strftime('%Y-%m-%d'),
                        'end': data.index.max().strftime('%Y-%m-%d')
                    }
                },
                'factors_info': {
                    'total_factors': factors_df.shape[1],
                    'factor_names': list(factors_df.columns),
                    'factor_types': factor_types
                },
                'evaluation': evaluation,
                'optimization': optimization_result,
                'report': report,
                'raw_factors': factors_df.to_dict('series')
            }
            
            # 保存到文件
            output_path = self._save_mining_results(results, symbol, timestamp)
            results['output_path'] = output_path
            
            print(f"🎉 因子挖掘流程完成！结果已保存到: {output_path}")
            
            return results
            
        except Exception as e:
            error_msg = f"因子挖掘过程中发生错误: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                'success': False,
                'error': error_msg
            }
    
    def load_data(self, symbol: str, timeframe: str = '1h', 
                 start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict:
        """
        加载市场数据
        
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
    
    def build_factors(self, data: pd.DataFrame, 
                     factor_types: Optional[List[str]] = None,
                     mining_config: Optional[Dict] = None) -> Dict:
        """
        构建因子
        
        Args:
            data: 市场数据
            factor_types: 因子类型列表
            mining_config: 挖掘配置
            
        Returns:
            因子构建结果
        """
        try:
            # 应用挖掘配置
            if mining_config and 'factor_params' in mining_config:
                kwargs = mining_config['factor_params']
            else:
                kwargs = {}
            
            # 构建因子
            factors_df = self.factor_builder.build_all_factors(
                data, 
                factor_types=factor_types, 
                save_to_storage=True,
                **kwargs
            )
            
            if factors_df.empty:
                return {
                    'success': False,
                    'error': '因子构建失败，结果为空'
                }
            
            return {
                'success': True,
                'factors': factors_df,
                'info': {
                    'total_factors': factors_df.shape[1],
                    'factor_names': list(factors_df.columns),
                    'factor_types': factor_types
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def save_evaluation_results_to_v3(self, factors_df: pd.DataFrame,
                                    evaluation_results: Dict,
                                    mining_config: Optional[Dict] = None) -> bool:
        """
        将评估结果保存到V3存储系统（复用因子评估网页的存储方法）
        
        Args:
            factors_df: 因子DataFrame
            evaluation_results: 评估结果字典
            mining_config: 挖掘配置
            
        Returns:
            是否保存成功
        """
        try:
            print("💾 开始保存评估结果到V3系统...")
            # 直接使用核心层IO
            from factor_miner.core.evaluation_io import save_evaluation_results

            for factor_name in factors_df.columns:
                if factor_name in evaluation_results:
                    factor_eval = evaluation_results[factor_name]

                    metadata = {
                        'mining_session': True,
                        'mining_config': mining_config or {},
                        'factor_type': mining_config.get('factor_types', []) if mining_config else [],
                        'symbols': mining_config.get('symbols', []) if mining_config else [],
                        'timeframes': mining_config.get('timeframes', []) if mining_config else []
                    }

                    save_evaluation_results(factor_name, factor_eval, metadata)
                    print(f"✅ 因子 {factor_name} 评估结果保存成功")

            print("💾 评估结果保存完成")
            return True

        except Exception as e:
            print(f"❌ 保存评估结果失败: {e}")
            return False
    
    def evaluate_factors(self, factors_df: pd.DataFrame, 
                        data: pd.DataFrame,
                        mining_config: Optional[Dict] = None) -> Dict:
        """
        评估因子
        
        Args:
            factors_df: 因子DataFrame
            data: 市场数据
            mining_config: 挖掘配置
            
        Returns:
            评估结果
        """
        try:
            evaluation_results = {}
            
            # 计算收益率
            returns = data['close'].pct_change()
            
            print(f"📊 开始评估 {factors_df.shape[1]} 个因子...")
            
            # 批量评估因子
            for i, factor_name in enumerate(factors_df.columns):
                try:
                    factor_series = factors_df[factor_name].dropna()
                    
                    if len(factor_series) < 30:
                        print(f"⚠️ 因子 {factor_name} 数据不足，跳过评估")
                        continue
                    
                    # 对齐数据
                    common_idx = factor_series.index.intersection(returns.index)
                    if len(common_idx) < 30:
                        print(f"⚠️ 因子 {factor_name} 数据对齐后不足，跳过评估")
                        continue
                    
                    factor_aligned = factor_series.loc[common_idx]
                    returns_aligned = returns.loc[common_idx]
                    
                    # 评估因子（使用与因子评估网页相同的方法）
                    try:
                        evaluation = self.evaluator.evaluate_single_factor(
                            factor_aligned, returns_aligned, factor_name
                        )
                    except Exception as e:
                        print(f"⚠️ 因子 {factor_name} 评估失败，使用基础评估: {e}")
                        # 如果评估失败，使用基础评估
                        evaluation = {
                            'factor_name': factor_name,
                            'ic_pearson': self.evaluator.stats.calculate_ic(factor_aligned, returns_aligned, 'pearson'),
                            'ic_spearman': self.evaluator.stats.calculate_ic(factor_aligned, returns_aligned, 'spearman'),
                            'win_rate': self.evaluator.stats.calculate_factor_win_rate(factor_aligned, returns_aligned),
                            'data_length': len(factor_aligned),
                            'missing_ratio': factor_aligned.isna().mean()
                        }
                    
                    evaluation_results[factor_name] = evaluation
                    
                    if (i + 1) % 10 == 0:
                        print(f"📈 评估进度: {i+1}/{len(factors_df.columns)}")
                        
                except Exception as e:
                    print(f"❌ 评估因子 {factor_name} 失败: {e}")
                    continue
            
            print(f"✅ 因子评估完成，共评估 {len(evaluation_results)} 个因子")
            
            return {
                'success': True,
                'evaluation': evaluation_results
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def optimize_factor_combination(self, factors_df: pd.DataFrame, 
                                  data: pd.DataFrame,
                                  mining_config: Optional[Dict] = None) -> Dict:
        """
        优化因子组合
        
        Args:
            factors_df: 因子DataFrame
            data: 市场数据
            mining_config: 挖掘配置
            
        Returns:
            优化结果
        """
        try:
            # 检查数据列名
            print(f"数据列名: {list(data.columns)}")
            print(f"数据形状: {data.shape}")
            
            # 尝试找到收盘价列
            close_col = None
            for col in ['close', 'Close', 'CLOSE', 'price', 'Price', 'PRICE']:
                if col in data.columns:
                    close_col = col
                    break
            
            if close_col is None:
                return {
                    'success': False,
                    'error': f'数据中未找到收盘价列。可用列: {list(data.columns)}'
                }
            
            print(f"使用收盘价列: {close_col}")
            
            # 设置数据
            returns = data[close_col].pct_change()
            self.optimizer.set_data(factors_df, returns)
            
            # 获取优化配置
            if mining_config and 'optimization' in mining_config:
                opt_config = mining_config['optimization']
                max_factors = opt_config.get('max_factors', 10)
                method = opt_config.get('method', 'greedy')
            else:
                max_factors = 10
                method = 'greedy'
            
            print(f"⚡ 开始优化因子组合，方法: {method}, 最大因子数: {max_factors}")
            
            # 运行优化
            best_combination, best_score = self.optimizer.optimize_factor_combination(
                factors_df, max_factors=max_factors, method=method
            )
            
            if best_combination is None:
                return {
                    'success': False,
                    'error': '因子组合优化失败'
                }
            
            print(f"✅ 因子组合优化完成，最佳得分: {best_score:.4f}")
            print(f"📊 选择的因子: {best_combination}")
            
            return {
                'success': True,
                'selected_factors': best_combination,
                'score': best_score,
                'method': method,
                'max_factors': max_factors
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def generate_mining_report(self, symbol: str, timeframe: str, 
                              data: pd.DataFrame, factors_df: pd.DataFrame,
                              evaluation: Dict, optimization: Dict,
                              mining_config: Optional[Dict] = None) -> str:
        """
        生成挖掘报告
        
        Args:
            symbol: 交易对
            timeframe: 时间框架
            data: 市场数据
            factors_df: 因子DataFrame
            evaluation: 评估结果
            optimization: 优化结果
            mining_config: 挖掘配置
            
        Returns:
            报告文本
        """
        report_lines = []
        
        # 标题
        report_lines.append("=" * 80)
        report_lines.append("🚀 FACTOR MINING REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # 基本信息
        report_lines.append("📊 MINING INFORMATION")
        report_lines.append("-" * 40)
        report_lines.append(f"Symbol: {symbol}")
        report_lines.append(f"Timeframe: {timeframe}")
        report_lines.append(f"Date Range: {data.index.min().strftime('%Y-%m-%d')} to {data.index.max().strftime('%Y-%m-%d')}")
        report_lines.append(f"Data Points: {data.shape[0]:,}")
        report_lines.append(f"Factor Types: {mining_config.get('factor_types', 'All') if mining_config else 'All'}")
        report_lines.append("")
        
        # 因子信息
        report_lines.append("🔧 FACTOR INFORMATION")
        report_lines.append("-" * 40)
        report_lines.append(f"Total Factors: {factors_df.shape[1]}")
        report_lines.append(f"Factor Categories: {', '.join(set([name.split('_')[0] for name in factors_df.columns]))}")
        report_lines.append("")
        
        # 评估结果摘要
        if evaluation:
            report_lines.append("📈 EVALUATION SUMMARY")
            report_lines.append("-" * 40)
            
            # 计算平均指标
            avg_ic = np.mean([eval_data.get('ic_pearson', np.nan) for eval_data in evaluation.values() if not np.isnan(eval_data.get('ic_pearson', np.nan))])
            avg_ir = np.mean([eval_data.get('sharpe_ratio', np.nan) for eval_data in evaluation.values() if not np.isnan(eval_data.get('sharpe_ratio', np.nan))])
            avg_win_rate = np.mean([eval_data.get('win_rate', np.nan) for eval_data in evaluation.values() if not np.isnan(eval_data.get('win_rate', np.nan))])
            
            report_lines.append(f"Average IC: {avg_ic:.4f}")
            report_lines.append(f"Average IR: {avg_ir:.4f}")
            report_lines.append(f"Average Win Rate: {avg_win_rate:.4f}")
            report_lines.append("")
            
            # 最佳因子
            best_factors = sorted(
                [(name, data.get('ic_pearson', 0)) for name, data in evaluation.items()],
                key=lambda x: x[1] if not np.isnan(x[1]) else 0,
                reverse=True
            )[:5]
            
            report_lines.append("🏆 TOP 5 FACTORS (by IC)")
            report_lines.append("-" * 40)
            for i, (name, ic) in enumerate(best_factors, 1):
                report_lines.append(f"{i}. {name}: IC = {ic:.4f}")
            report_lines.append("")
        
        # 优化结果
        if optimization.get('success'):
            report_lines.append("⚡ OPTIMIZATION RESULTS")
            report_lines.append("-" * 40)
            report_lines.append(f"Method: {optimization.get('method', 'Unknown')}")
            report_lines.append(f"Best Score: {optimization.get('score', 0):.4f}")
            report_lines.append(f"Selected Factors: {len(optimization.get('selected_factors', []))}")
            report_lines.append("")
            
            if optimization.get('selected_factors'):
                report_lines.append("📊 SELECTED FACTORS:")
                for factor in optimization['selected_factors']:
                    report_lines.append(f"  - {factor}")
                report_lines.append("")
        
        # 配置信息
        if mining_config:
            report_lines.append("⚙️ MINING CONFIGURATION")
            report_lines.append("-" * 40)
            for key, value in mining_config.items():
                report_lines.append(f"{key}: {value}")
            report_lines.append("")
        
        # 时间戳
        report_lines.append(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return "\n".join(report_lines)
    
    def _save_mining_results(self, results: Dict, symbol: str, timestamp: str) -> str:
        """
        保存挖掘结果
        
        Args:
            results: 挖掘结果
            symbol: 交易对
            timestamp: 时间戳
            
        Returns:
            保存路径
        """
        try:
            # 创建结果目录
            results_dir = Path("factorlib") / "mining_history" / "factor_mining" / symbol
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存完整结果
            output_path = results_dir / f"mining_results_{timestamp}.json"
            
            # 转换DataFrame为可序列化格式
            serializable_results = results.copy()
            if 'raw_factors' in serializable_results:
                serializable_results['raw_factors'] = {
                    name: series.to_dict() for name, series in serializable_results['raw_factors'].items()
                }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False, default=str)
            
            # 保存因子数据为CSV
            if 'raw_factors' in results:
                csv_path = results_dir / f"factors_{timestamp}.csv"
                factors_df = pd.DataFrame(results['raw_factors'])
                factors_df.to_csv(csv_path)
                print(f"📊 因子数据已保存到: {csv_path}")
            
            return str(output_path)
            
        except Exception as e:
            print(f"❌ 保存结果失败: {e}")
            return ""
    
    def get_mining_history(self, symbol: Optional[str] = None) -> List[Dict]:
        """
        获取挖掘历史
        
        Args:
            symbol: 交易对，如果为None则返回所有
            
        Returns:
            挖掘历史列表
        """
        try:
            results_dir = Path("factorlib") / "mining_history" / "factor_mining"
            if not results_dir.exists():
                return []
            
            history = []
            
            for symbol_dir in results_dir.iterdir():
                if symbol_dir.is_dir():
                    if symbol and symbol_dir.name != symbol:
                        continue
                        
                    for result_file in symbol_dir.glob("mining_results_*.json"):
                        try:
                            with open(result_file, 'r', encoding='utf-8') as f:
                                result_data = json.load(f)
                            
                            history.append({
                                'symbol': symbol_dir.name,
                                'timestamp': result_file.stem.split('_')[-1],
                                'file_path': str(result_file),
                                'mining_info': result_data.get('mining_info', {}),
                                'factors_count': result_data.get('factors_info', {}).get('total_factors', 0)
                            })
                        except Exception as e:
                            print(f"读取结果文件 {result_file} 失败: {e}")
                            continue
            
            # 按时间排序
            history.sort(key=lambda x: x['timestamp'], reverse=True)
            
            return history
            
        except Exception as e:
            print(f"获取挖掘历史失败: {e}")
            return []
    
    def load_mining_result(self, file_path: str) -> Dict:
        """
        加载挖掘结果
        
        Args:
            file_path: 结果文件路径
            
        Returns:
            挖掘结果
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                result_data = json.load(f)
            
            # 转换因子数据回DataFrame
            if 'raw_factors' in result_data:
                factors_dict = result_data['raw_factors']
                factors_df = pd.DataFrame(factors_dict)
                result_data['factors_df'] = factors_df
            
            return result_data
            
        except Exception as e:
            return {
                'success': False,
                'error': f"加载挖掘结果失败: {e}"
            }
