"""
可视化模块
包含各种图表和报告生成功能
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class Visualizer:
    """
    可视化器
    生成各种图表和报告
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化可视化器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self._setup_plotting_style()
        
    def _setup_plotting_style(self):
        """
        设置绘图样式
        """
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 设置绘图样式
        plt.style.use('seaborn-v0_8')
        
        # 设置颜色主题
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff7f0e',
            'info': '#17a2b8',
            'light': '#f8f9fa',
            'dark': '#343a40'
        }
    
    def plot_results(self, results: Dict, plot_type: str = 'all', **kwargs):
        """
        绘制结果图表
        
        Args:
            results: 结果数据
            plot_type: 图表类型
            **kwargs: 其他参数
        """
        if plot_type == 'all':
            self._plot_all_results(results, **kwargs)
        elif plot_type == 'evaluation':
            self._plot_evaluation_results(results.get('evaluation', {}), **kwargs)
        elif plot_type == 'backtest':
            self._plot_backtest_results(results.get('backtest', {}), **kwargs)
        elif plot_type == 'factors':
            self._plot_factor_analysis(results.get('factors', {}), **kwargs)
        else:
            print(f"不支持的图表类型: {plot_type}")
    
    def _plot_all_results(self, results: Dict, **kwargs):
        """
        绘制所有结果图表
        """
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        
        # 评估结果
        if 'evaluation' in results:
            self._plot_ic_analysis(results['evaluation'], axes[0, 0])
            self._plot_correlation_heatmap(results['evaluation'], axes[0, 1])
        
        # 回测结果
        if 'backtest' in results:
            self._plot_cumulative_returns(results['backtest'], axes[1, 0])
            self._plot_performance_metrics(results['backtest'], axes[1, 1])
        
        plt.tight_layout()
        plt.show()
    
    def _plot_evaluation_results(self, evaluation_results: Dict, **kwargs):
        """
        绘制评估结果图表
        """
        if not evaluation_results:
            print("没有评估结果数据")
            return
        
        # IC分析
        if 'ic' in evaluation_results:
            self._plot_ic_analysis(evaluation_results['ic'])
        
        # 相关性分析
        if 'correlation' in evaluation_results:
            self._plot_correlation_heatmap(evaluation_results['correlation'])
        
        # 稳定性分析
        if 'stability' in evaluation_results:
            self._plot_stability_analysis(evaluation_results['stability'])
    
    def _plot_backtest_results(self, backtest_results: Dict, **kwargs):
        """
        绘制回测结果图表
        """
        if not backtest_results:
            print("没有回测结果数据")
            return
        
        # 累积收益率
        if 'portfolio' in backtest_results:
            self._plot_cumulative_returns(backtest_results)
        
        # 性能指标
        if 'performance' in backtest_results:
            self._plot_performance_metrics(backtest_results)
    
    def _plot_ic_analysis(self, ic_results: Dict, ax=None):
        """
        绘制IC分析图表
        """
        if not ic_results:
            return
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))
        
        # IC统计
        periods = []
        ic_means = []
        ic_stds = []
        ic_irs = []
        
        for period, data in ic_results.items():
            if isinstance(data, dict) and 'mean' in data:
                periods.append(period)
                ic_means.append(data['mean'])
                ic_stds.append(data['std'])
                ic_irs.append(data.get('ir', 0))
        
        if periods:
            x = np.arange(len(periods))
            width = 0.35
            
            # 绘制平均IC
            bars1 = ax.bar(x - width/2, ic_means, width, label='平均IC', color=self.colors['primary'])
            ax.bar_label(bars1, fmt='%.3f')
            
            # 绘制信息比率
            ax2 = ax.twinx()
            bars2 = ax2.bar(x + width/2, ic_irs, width, label='信息比率', color=self.colors['secondary'])
            ax2.bar_label(bars2, fmt='%.3f')
            
            ax.set_xlabel('预测期')
            ax.set_ylabel('IC值', color=self.colors['primary'])
            ax2.set_ylabel('信息比率', color=self.colors['secondary'])
            ax.set_title('IC分析')
            ax.set_xticks(x)
            ax.set_xticklabels(periods)
            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
    
    def _plot_correlation_heatmap(self, correlation_results: Dict, ax=None):
        """
        绘制相关性热力图
        """
        if not correlation_results or 'correlation_matrix' not in correlation_results:
            return
        
        corr_matrix = correlation_results['correlation_matrix']
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 10))
        
        # 绘制热力图
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title('因子相关性热力图')
    
    def _plot_stability_analysis(self, stability_results: Dict):
        """
        绘制稳定性分析图表
        """
        if not stability_results:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 自相关性
        factors = list(stability_results.keys())
        autocorrs = [stability_results[f]['autocorrelation'] for f in factors]
        
        axes[0, 0].bar(factors, autocorrs, color=self.colors['primary'])
        axes[0, 0].set_title('因子自相关性')
        axes[0, 0].set_ylabel('自相关系数')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 变化波动率
        change_vols = [stability_results[f]['change_volatility'] for f in factors]
        axes[0, 1].bar(factors, change_vols, color=self.colors['secondary'])
        axes[0, 1].set_title('因子变化波动率')
        axes[0, 1].set_ylabel('波动率')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 分布稳定性
        dist_stabilities = [stability_results[f]['distribution_stability'] for f in factors]
        axes[1, 0].bar(factors, dist_stabilities, color=self.colors['success'])
        axes[1, 0].set_title('分布稳定性')
        axes[1, 0].set_ylabel('稳定性指标')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 数据质量
        data_qualities = [stability_results[f]['data_quality'] for f in factors]
        axes[1, 1].bar(factors, data_qualities, color=self.colors['info'])
        axes[1, 1].set_title('数据质量')
        axes[1, 1].set_ylabel('完整度')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_cumulative_returns(self, backtest_results: Dict, ax=None):
        """
        绘制累积收益率图表
        """
        if not backtest_results or 'portfolio' not in backtest_results:
            return
        
        portfolio = backtest_results['portfolio']
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))
        
        # 获取累积收益率列
        cumulative_cols = [col for col in portfolio.columns if col.endswith('_cumulative_returns')]
        
        for col in cumulative_cols:
            factor_name = col.replace('_cumulative_returns', '')
            ax.plot(portfolio.index, portfolio[col], label=factor_name, linewidth=2)
        
        ax.set_title('累积收益率')
        ax.set_xlabel('日期')
        ax.set_ylabel('累积收益率')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
    
    def _plot_performance_metrics(self, backtest_results: Dict, ax=None):
        """
        绘制性能指标图表
        """
        if not backtest_results or 'performance' not in backtest_results:
            return
        
        performance = backtest_results['performance']
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))
        
        # 选择关键指标
        metrics = ['sharpe_ratio', 'max_drawdown', 'win_rate']
        factors = list(performance.keys())
        
        x = np.arange(len(factors))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            values = [performance[factor][metric] for factor in factors]
            ax.bar(x + i * width, values, width, label=metric)
        
        ax.set_xlabel('因子')
        ax.set_ylabel('指标值')
        ax.set_title('性能指标对比')
        ax.set_xticks(x + width)
        ax.set_xticklabels(factors, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def generate_report(self, data: pd.DataFrame = None, factors: pd.DataFrame = None,
                       evaluation_results: Dict = None, backtest_results: Dict = None,
                       output_path: str = 'factor_mining_report.html', **kwargs):
        """
        生成HTML报告
        
        Args:
            data: 市场数据
            factors: 因子数据
            evaluation_results: 评估结果
            backtest_results: 回测结果
            output_path: 输出文件路径
            **kwargs: 其他参数
        """
        html_content = self._generate_html_content(data, factors, evaluation_results, backtest_results)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"报告已生成: {output_path}")
    
    def _generate_html_content(self, data: pd.DataFrame, factors: pd.DataFrame,
                             evaluation_results: Dict, backtest_results: Dict) -> str:
        """
        生成HTML内容
        """
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>因子挖掘分析报告</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1, h2, h3 { color: #333; }
                .section { margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
                .metric { display: inline-block; margin: 10px; padding: 10px; background: #f5f5f5; border-radius: 3px; }
                .positive { color: green; }
                .negative { color: red; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <h1>因子挖掘分析报告</h1>
        """
        
        # 数据概览
        if data is not None:
            html += self._generate_data_overview_html(data)
        
        # 因子分析
        if factors is not None:
            html += self._generate_factor_analysis_html(factors)
        
        # 评估结果
        if evaluation_results is not None:
            html += self._generate_evaluation_html(evaluation_results)
        
        # 回测结果
        if backtest_results is not None:
            html += self._generate_backtest_html(backtest_results)
        
        html += """
        </body>
        </html>
        """
        
        return html
    
    def _generate_data_overview_html(self, data: pd.DataFrame) -> str:
        """
        生成数据概览HTML
        """
        html = """
        <div class="section">
            <h2>数据概览</h2>
            <p><strong>数据时间范围:</strong> {} 至 {}</p>
            <p><strong>数据点数量:</strong> {}</p>
            <p><strong>数据列:</strong> {}</p>
        </div>
        """.format(
            data.index.min().strftime('%Y-%m-%d'),
            data.index.max().strftime('%Y-%m-%d'),
            len(data),
            ', '.join(data.columns)
        )
        return html
    
    def _generate_factor_analysis_html(self, factors: pd.DataFrame) -> str:
        """
        生成因子分析HTML
        """
        html = """
        <div class="section">
            <h2>因子分析</h2>
            <p><strong>因子数量:</strong> {}</p>
            <p><strong>因子列表:</strong> {}</p>
        </div>
        """.format(
            len(factors.columns),
            ', '.join(factors.columns)
        )
        return html
    
    def _generate_evaluation_html(self, evaluation_results: Dict) -> str:
        """
        生成评估结果HTML
        """
        html = '<div class="section"><h2>因子评估结果</h2>'
        
        if 'ic' in evaluation_results:
            html += '<h3>IC分析</h3>'
            for period, data in evaluation_results['ic'].items():
                if isinstance(data, dict) and 'mean' in data:
                    html += """
                    <div class="metric">
                        <strong>{}</strong><br>
                        平均IC: {:.4f}<br>
                        信息比率: {:.4f}<br>
                        正IC比例: {:.2%}
                    </div>
                    """.format(period, data['mean'], data.get('ir', 0), data.get('positive_ratio', 0))
        
        html += '</div>'
        return html
    
    def _generate_backtest_html(self, backtest_results: Dict) -> str:
        """
        生成回测结果HTML
        """
        html = '<div class="section"><h2>回测结果</h2>'
        
        if 'performance' in backtest_results:
            html += '<h3>性能指标</h3><table><tr><th>因子</th><th>年化收益率</th><th>夏普比率</th><th>最大回撤</th><th>胜率</th></tr>'
            
            for factor, metrics in backtest_results['performance'].items():
                html += """
                <tr>
                    <td>{}</td>
                    <td class="{}">{:.2%}</td>
                    <td>{:.4f}</td>
                    <td class="{}">{:.2%}</td>
                    <td>{:.2%}</td>
                </tr>
                """.format(
                    factor,
                    'positive' if metrics['annual_return'] > 0 else 'negative',
                    metrics['annual_return'],
                    metrics['sharpe_ratio'],
                    'negative',
                    metrics['max_drawdown'],
                    metrics['win_rate']
                )
            
            html += '</table>'
        
        html += '</div>'
        return html 