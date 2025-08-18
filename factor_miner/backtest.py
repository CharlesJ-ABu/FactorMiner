"""
回测模块
包含策略回测和性能评估功能
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import warnings
warnings.filterwarnings('ignore')


class Backtester:
    """
    回测器
    执行因子策略回测和性能评估
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化回测器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        
    def run_backtest(self,
                    factors: pd.DataFrame,
                    data: pd.DataFrame,
                    strategy_config: Optional[Dict] = None,
                    **kwargs) -> Dict:
        """
        运行回测
        
        Args:
            factors: 因子数据
            data: 市场数据
            strategy_config: 策略配置
            **kwargs: 其他参数
            
        Returns:
            回测结果字典
        """
        if strategy_config is None:
            strategy_config = self._get_default_strategy_config()
        
        # 合并数据和因子
        backtest_data = self._prepare_backtest_data(factors, data)
        
        # 生成交易信号
        signals = self._generate_signals(backtest_data, strategy_config)
        
        # 执行回测
        portfolio = self._execute_backtest(backtest_data, signals, strategy_config)
        
        # 计算性能指标
        performance = self._calculate_performance(portfolio, strategy_config)
        
        return {
            'portfolio': portfolio,
            'signals': signals,
            'performance': performance,
            'strategy_config': strategy_config
        }
    
    def _get_default_strategy_config(self) -> Dict:
        """
        获取默认策略配置
        """
        return {
            'strategy_type': 'long_short',  # long_short, long_only, short_only
            'rebalance_frequency': 'daily',  # daily, weekly, monthly
            'n_groups': 5,  # 分组数量
            'transaction_cost': 0.001,  # 交易成本
            'slippage': 0.0005,  # 滑点
            'max_position': 1.0,  # 最大仓位
            'stop_loss': None,  # 止损
            'take_profit': None,  # 止盈
            'risk_free_rate': 0.02,  # 无风险利率
            'benchmark': None  # 基准
        }
    
    def _prepare_backtest_data(self, factors: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """
        准备回测数据
        """
        # 合并因子和市场数据
        backtest_data = pd.concat([data, factors], axis=1)
        
        # 获取正确的列名
        close_col = 'close' if 'close' in backtest_data.columns else 'S_DQ_CLOSE'
        
        # 计算收益率
        backtest_data['returns'] = backtest_data[close_col].pct_change()
        
        # 处理缺失值
        backtest_data = backtest_data.dropna()
        
        return backtest_data
    
    def _generate_signals(self, data: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """
        生成交易信号
        """
        signals = pd.DataFrame(index=data.index)
        
        strategy_type = config['strategy_type']
        n_groups = config['n_groups']
        
        # 获取因子列
        factor_cols = [col for col in data.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'returns']]
        
        for factor in factor_cols:
            # 按因子值分组
            factor_data = data[factor].dropna()
            
            if len(factor_data) > n_groups:
                # 计算分组边界
                quantiles = np.linspace(0, 1, n_groups + 1)
                boundaries = factor_data.quantile(quantiles)
                
                # 生成信号
                signal = pd.Series(0, index=data.index)
                
                for i in range(n_groups):
                    if i == 0:  # 第一组（最低分组）
                        mask = (data[factor] <= boundaries.iloc[i+1])
                        if strategy_type in ['long_short', 'short_only']:
                            signal[mask] = -1
                        elif strategy_type == 'long_only':
                            signal[mask] = 0
                    elif i == n_groups - 1:  # 最后一组（最高分组）
                        mask = (data[factor] > boundaries.iloc[i])
                        if strategy_type in ['long_short', 'long_only']:
                            signal[mask] = 1
                        elif strategy_type == 'short_only':
                            signal[mask] = 0
                    else:  # 中间分组
                        mask = (data[factor] > boundaries.iloc[i]) & (data[factor] <= boundaries.iloc[i+1])
                        signal[mask] = 0
                
                signals[f'{factor}_signal'] = signal
        
        return signals
    
    def _execute_backtest(self, data: pd.DataFrame, signals: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """
        执行回测
        """
        portfolio = pd.DataFrame(index=data.index)
        # 获取正确的列名
        close_col = 'close' if 'close' in data.columns else 'S_DQ_CLOSE'
        
        portfolio['price'] = data[close_col]
        portfolio['returns'] = data['returns']
        
        # 获取信号列
        signal_cols = [col for col in signals.columns if col.endswith('_signal')]
        
        for signal_col in signal_cols:
            factor_name = signal_col.replace('_signal', '')
            signal = signals[signal_col]
            
            # 计算策略收益率
            strategy_returns = signal.shift(1) * data['returns']
            
            # 应用交易成本
            transaction_cost = config.get('transaction_cost', 0.001)
            position_change = signal.diff().abs()
            cost = position_change * transaction_cost
            strategy_returns = strategy_returns - cost
            
            # 应用滑点
            slippage = config.get('slippage', 0.0005)
            slippage_cost = signal.shift(1).abs() * slippage
            strategy_returns = strategy_returns - slippage_cost
            
            portfolio[f'{factor_name}_strategy_returns'] = strategy_returns
            
            # 计算累积收益率
            portfolio[f'{factor_name}_cumulative_returns'] = (1 + strategy_returns).cumprod()
            
            # 计算回撤
            portfolio[f'{factor_name}_drawdown'] = self._calculate_drawdown(
                portfolio[f'{factor_name}_cumulative_returns']
            )
        
        return portfolio
    
    def _calculate_drawdown(self, cumulative_returns: pd.Series) -> pd.Series:
        """
        计算回撤
        """
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        return drawdown
    
    def _calculate_performance(self, portfolio: pd.DataFrame, config: Dict) -> Dict:
        """
        计算性能指标
        """
        performance = {}
        
        # 获取策略收益率列
        strategy_cols = [col for col in portfolio.columns if col.endswith('_strategy_returns')]
        
        for strategy_col in strategy_cols:
            factor_name = strategy_col.replace('_strategy_returns', '')
            returns = portfolio[strategy_col].dropna()
            
            if len(returns) > 0:
                # 基础统计指标
                total_return = (1 + returns).prod() - 1
                annual_return = (1 + total_return) ** (252 / len(returns)) - 1
                volatility = returns.std() * np.sqrt(252)
                sharpe_ratio = (annual_return - config.get('risk_free_rate', 0.02)) / volatility if volatility > 0 else 0
                
                # 最大回撤
                cumulative_returns = (1 + returns).cumprod()
                max_drawdown = self._calculate_drawdown(cumulative_returns).min()
                
                # 胜率
                win_rate = (returns > 0).mean()
                
                # 盈亏比
                winning_returns = returns[returns > 0]
                losing_returns = returns[returns < 0]
                profit_loss_ratio = abs(winning_returns.mean() / losing_returns.mean()) if len(losing_returns) > 0 else np.inf
                
                # 夏普比率（年化）
                sharpe_ratio_annual = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
                
                # 索提诺比率
                downside_returns = returns[returns < 0]
                downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
                sortino_ratio = (annual_return - config.get('risk_free_rate', 0.02)) / downside_deviation if downside_deviation > 0 else 0
                
                # 卡尔马比率
                calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
                
                # 信息比率（相对于基准）
                if config.get('benchmark') is not None:
                    benchmark_returns = config['benchmark']
                    excess_returns = returns - benchmark_returns
                    information_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0
                else:
                    information_ratio = np.nan
                
                performance[factor_name] = {
                    'total_return': total_return,
                    'annual_return': annual_return,
                    'volatility': volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'sharpe_ratio_annual': sharpe_ratio_annual,
                    'sortino_ratio': sortino_ratio,
                    'calmar_ratio': calmar_ratio,
                    'information_ratio': information_ratio,
                    'max_drawdown': max_drawdown,
                    'win_rate': win_rate,
                    'profit_loss_ratio': profit_loss_ratio,
                    'num_trades': len(returns),
                    'avg_return': returns.mean(),
                    'skewness': returns.skew(),
                    'kurtosis': returns.kurtosis()
                }
        
        return performance
    
    def run_multiple_strategies(self,
                              factors: pd.DataFrame,
                              data: pd.DataFrame,
                              strategy_configs: List[Dict],
                              **kwargs) -> Dict:
        """
        运行多个策略回测
        
        Args:
            factors: 因子数据
            data: 市场数据
            strategy_configs: 策略配置列表
            **kwargs: 其他参数
            
        Returns:
            多策略回测结果
        """
        results = {}
        
        for i, config in enumerate(strategy_configs):
            strategy_name = config.get('name', f'strategy_{i}')
            try:
                result = self.run_backtest(factors, data, config, **kwargs)
                results[strategy_name] = result
                print(f"完成策略 {strategy_name} 的回测")
            except Exception as e:
                print(f"策略 {strategy_name} 回测失败: {e}")
                continue
        
        return results
    
    def compare_strategies(self, results: Dict) -> pd.DataFrame:
        """
        比较多个策略的性能
        
        Args:
            results: 多策略回测结果
            
        Returns:
            策略比较DataFrame
        """
        comparison_data = []
        
        for strategy_name, result in results.items():
            performance = result['performance']
            
            for factor_name, metrics in performance.items():
                row = {
                    'strategy': strategy_name,
                    'factor': factor_name,
                    **metrics
                }
                comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        return comparison_df
    
    def generate_backtest_report(self, results: Dict, output_path: str = None) -> str:
        """
        生成回测报告
        
        Args:
            results: 回测结果
            output_path: 输出文件路径
            
        Returns:
            报告内容
        """
        report = []
        report.append("# 因子策略回测报告\n")
        
        # 策略配置
        if 'strategy_config' in results:
            config = results['strategy_config']
            report.append("## 策略配置\n")
            for key, value in config.items():
                report.append(f"- {key}: {value}")
            report.append("")
        
        # 性能指标
        if 'performance' in results:
            report.append("## 性能指标\n")
            performance = results['performance']
            
            for factor_name, metrics in performance.items():
                report.append(f"### {factor_name}\n")
                report.append(f"- 总收益率: {metrics['total_return']:.2%}")
                report.append(f"- 年化收益率: {metrics['annual_return']:.2%}")
                report.append(f"- 年化波动率: {metrics['volatility']:.2%}")
                report.append(f"- 夏普比率: {metrics['sharpe_ratio']:.4f}")
                report.append(f"- 最大回撤: {metrics['max_drawdown']:.2%}")
                report.append(f"- 胜率: {metrics['win_rate']:.2%}")
                report.append(f"- 盈亏比: {metrics['profit_loss_ratio']:.4f}")
                report.append(f"- 交易次数: {metrics['num_trades']}")
                report.append("")
        
        report_text = "\n".join(report)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
        
        return report_text
    
    def plot_backtest_results(self, results: Dict, **kwargs):
        """
        绘制回测结果图表
        
        Args:
            results: 回测结果
            **kwargs: 其他参数
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        portfolio = results['portfolio']
        
        # 绘制累积收益率
        plt.figure(figsize=(15, 10))
        
        # 子图1：累积收益率
        plt.subplot(2, 2, 1)
        cumulative_cols = [col for col in portfolio.columns if col.endswith('_cumulative_returns')]
        for col in cumulative_cols:
            factor_name = col.replace('_cumulative_returns', '')
            plt.plot(portfolio.index, portfolio[col], label=factor_name)
        plt.title('累积收益率')
        plt.legend()
        plt.grid(True)
        
        # 子图2：回撤
        plt.subplot(2, 2, 2)
        drawdown_cols = [col for col in portfolio.columns if col.endswith('_drawdown')]
        for col in drawdown_cols:
            factor_name = col.replace('_drawdown', '')
            plt.plot(portfolio.index, portfolio[col], label=factor_name)
        plt.title('回撤')
        plt.legend()
        plt.grid(True)
        
        # 子图3：收益率分布
        plt.subplot(2, 2, 3)
        strategy_cols = [col for col in portfolio.columns if col.endswith('_strategy_returns')]
        for col in strategy_cols:
            factor_name = col.replace('_strategy_returns', '')
            returns = portfolio[col].dropna()
            plt.hist(returns, bins=50, alpha=0.7, label=factor_name)
        plt.title('收益率分布')
        plt.legend()
        plt.grid(True)
        
        # 子图4：性能指标对比
        plt.subplot(2, 2, 4)
        if 'performance' in results:
            performance = results['performance']
            metrics = ['sharpe_ratio', 'max_drawdown', 'win_rate']
            factor_names = list(performance.keys())
            
            x = np.arange(len(factor_names))
            width = 0.25
            
            for i, metric in enumerate(metrics):
                values = [performance[factor][metric] for factor in factor_names]
                plt.bar(x + i * width, values, width, label=metric)
            
            plt.xlabel('因子')
            plt.ylabel('指标值')
            plt.title('性能指标对比')
            plt.xticks(x + width, factor_names, rotation=45)
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # 绘制相关性热力图
        if len(strategy_cols) > 1:
            plt.figure(figsize=(10, 8))
            strategy_returns = portfolio[strategy_cols]
            corr_matrix = strategy_returns.corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title('策略收益率相关性')
            plt.show() 