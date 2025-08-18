#!/usr/bin/env python3
"""
数据科学统计模块
包含因子评估的常用统计指标：IC值、IR比率、正比例等
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats
from sklearn.metrics import mutual_info_score
import warnings
warnings.filterwarnings('ignore')


class FactorStatistics:
    """
    因子统计评估类
    提供各种因子评估的统计指标
    """
    
    def __init__(self):
        """初始化统计评估器"""
        pass
    
    def calculate_ic(self, factor: pd.Series, returns: pd.Series, 
                    method: str = 'pearson') -> float:
        """
        计算信息系数(Information Coefficient, IC)
        
        Args:
            factor: 因子值序列
            returns: 收益率序列
            method: 相关系数计算方法 ('pearson', 'spearman', 'kendall')
            
        Returns:
            float: IC值
        """
        # 对齐数据
        factor = factor.dropna()
        returns = returns.loc[factor.index]
        
        if len(factor) < 10:
            return np.nan
        
        # 计算相关系数
        if method == 'pearson':
            correlation = factor.corr(returns)
        elif method == 'spearman':
            correlation = factor.corr(returns, method='spearman')
        elif method == 'kendall':
            correlation = factor.corr(returns, method='kendall')
        else:
            raise ValueError(f"不支持的相关系数方法: {method}")
            
        return correlation
    
    def calculate_rank_ic(self, factor: pd.Series, returns: pd.Series) -> float:
        """
        计算排序IC (Rank IC)
        
        Args:
            factor: 因子值序列
            returns: 收益率序列
            
        Returns:
            float: Rank IC值
        """
        return self.calculate_ic(factor, returns, method='spearman')
    
    def calculate_mutual_information(self, factor: pd.Series, returns: pd.Series, 
                                   bins: int = 10) -> float:
        """
        计算互信息(Mutual Information)
        
        Args:
            factor: 因子值序列
            returns: 收益率序列
            bins: 分箱数量
            
        Returns:
            float: 互信息值
        """
        # 对齐数据
        factor = factor.dropna()
        returns = returns.loc[factor.index]
        
        if len(factor) < 10:
            return np.nan
        
        # 处理NaN值
        factor_clean = factor.dropna()
        returns_clean = returns.loc[factor_clean.index].dropna()
        
        if len(factor_clean) < 10 or len(returns_clean) < 10:
            return np.nan
        
        # 确保两个序列长度一致
        common_index = factor_clean.index.intersection(returns_clean.index)
        if len(common_index) < 10:
            return np.nan
            
        factor_clean = factor_clean.loc[common_index]
        returns_clean = returns_clean.loc[common_index]
        
        try:
            # 分箱处理
            factor_binned = pd.cut(factor_clean, bins=bins, labels=False, duplicates='drop')
            returns_binned = pd.cut(returns_clean, bins=bins, labels=False, duplicates='drop')
            
            # 处理分箱后的NaN值
            valid_mask = ~(factor_binned.isna() | returns_binned.isna())
            if valid_mask.sum() < 10:
                return np.nan
                
            factor_binned = factor_binned[valid_mask]
            returns_binned = returns_binned[valid_mask]
            
            # 计算互信息
            mi = mutual_info_score(factor_binned, returns_binned)
            return mi
        except Exception:
            return np.nan
    
    def calculate_ic_ir(self, ic_series: pd.Series) -> float:
        """
        计算信息比率(Information Ratio, IR)
        
        Args:
            ic_series: IC值序列
            
        Returns:
            float: IR值
        """
        if len(ic_series) == 0:
            return np.nan
        
        ic_mean = ic_series.mean()
        ic_std = ic_series.std()
        
        if ic_std == 0:
            return np.nan
            
        return ic_mean / ic_std
    
    def calculate_ic_positive_ratio(self, ic_series: pd.Series) -> float:
        """
        计算IC正比例
        
        Args:
            ic_series: IC值序列
            
        Returns:
            float: 正IC比例
        """
        if len(ic_series) == 0:
            return np.nan
            
        return (ic_series > 0).mean()
    
    def calculate_ic_stability(self, ic_series: pd.Series, window: int = 20) -> float:
        """
        计算IC稳定性
        
        Args:
            ic_series: IC值序列
            window: 滚动窗口大小
            
        Returns:
            float: IC稳定性指标
        """
        if len(ic_series) < window:
            return np.nan
        
        # 计算滚动IC的标准差
        rolling_ic_std = ic_series.rolling(window=window).std()
        ic_stability = 1 / (1 + rolling_ic_std.mean())
        
        return ic_stability
    
    def calculate_factor_returns(self, factor: pd.Series, returns: pd.Series, 
                               n_groups: int = 5) -> Dict[str, float]:
        """
        计算因子收益率统计
        
        Args:
            factor: 因子值序列
            returns: 收益率序列
            n_groups: 分组数量
            
        Returns:
            Dict: 包含各种收益率统计指标
        """
        # 对齐数据
        factor = factor.dropna()
        returns = returns.loc[factor.index]
        
        if len(factor) < n_groups * 10:
            return {}
        
        # 分组
        try:
            factor_quantiles = pd.qcut(factor, n_groups, labels=False, duplicates='drop')
        except ValueError:
            # 如果唯一值太少，使用cut
            factor_quantiles = pd.cut(factor, n_groups, labels=False, duplicates='drop')
        
        # 计算各组收益率
        group_returns = returns.groupby(factor_quantiles).mean()
        
        # 计算统计指标
        long_short_return = group_returns.iloc[-1] - group_returns.iloc[0]  # 多空收益
        long_return = group_returns.iloc[-1]  # 多头收益
        short_return = group_returns.iloc[0]  # 空头收益
        
        # 计算收益率标准差
        returns_std = returns.std()
        
        # 计算夏普比率
        sharpe_ratio = long_short_return / returns_std if returns_std > 0 else 0
        
        return {
            'long_short_return': long_short_return,
            'long_return': long_return,
            'short_return': short_return,
            'returns_std': returns_std,
            'sharpe_ratio': sharpe_ratio,
            'group_returns': group_returns.to_dict()
        }
    
    def calculate_factor_turnover(self, factor: pd.Series, window: int = 20) -> float:
        """
        计算因子换手率
        
        Args:
            factor: 因子值序列
            window: 滚动窗口大小
            
        Returns:
            float: 换手率
        """
        if len(factor) < window + 1:
            return np.nan
        
        # 计算因子变化
        factor_change = factor.diff().abs()
        
        # 计算滚动平均换手率
        turnover = factor_change.rolling(window=window).mean()
        
        return turnover.mean()
    
    def calculate_factor_decay(self, factor: pd.Series, returns: pd.Series, 
                             max_lag: int = 10) -> Dict[str, float]:
        """
        计算因子衰减特征（修复未来函数问题）
        
        Args:
            factor: 因子值序列
            returns: 收益率序列
            max_lag: 最大滞后期数
            
        Returns:
            Dict: 包含衰减特征
        """
        ic_decay = {}
        
        for lag in range(max_lag + 1):
            if lag == 0:
                # 当期IC：当期因子值 vs 当期收益率
                ic = self.calculate_ic(factor, returns)
            else:
                # 🚨 修复：使用历史因子值预测未来收益率，而不是未来收益率
                # 原来的错误：lagged_returns = returns.shift(-lag)  # 使用未来数据
                # 修复后：使用历史因子值
                lagged_factor = factor.shift(lag)  # 正数shift = 历史数据
                ic = self.calculate_ic(lagged_factor, returns)
            
            ic_decay[f'ic_lag_{lag}'] = ic
        
        # 计算衰减速度
        ic_values = list(ic_decay.values())
        if len(ic_values) > 1:
            decay_rate = (ic_values[0] - ic_values[-1]) / len(ic_values)
        else:
            decay_rate = 0
        
        ic_decay['decay_rate'] = decay_rate
        
        return ic_decay
    
    def calculate_factor_correlation(self, factors: pd.DataFrame) -> pd.DataFrame:
        """
        计算因子间相关性矩阵
        
        Args:
            factors: 因子DataFrame
            
        Returns:
            pd.DataFrame: 相关性矩阵
        """
        return factors.corr()
    
    def calculate_factor_autocorrelation(self, factor: pd.Series, 
                                       max_lag: int = 10) -> Dict[str, float]:
        """
        计算因子自相关性
        
        Args:
            factor: 因子值序列
            max_lag: 最大滞后期数
            
        Returns:
            Dict: 包含各期自相关系数
        """
        autocorr = {}
        
        for lag in range(1, max_lag + 1):
            if len(factor) > lag:
                corr = factor.autocorr(lag=lag)
                autocorr[f'autocorr_lag_{lag}'] = corr
        
        return autocorr
    
    def calculate_factor_skewness_kurtosis(self, factor: pd.Series) -> Dict[str, float]:
        """
        计算因子偏度和峰度
        
        Args:
            factor: 因子值序列
            
        Returns:
            Dict: 包含偏度和峰度
        """
        factor_clean = factor.dropna()
        
        if len(factor_clean) < 10:
            return {'skewness': np.nan, 'kurtosis': np.nan}
        
        skewness = factor_clean.skew()
        kurtosis = factor_clean.kurtosis()
        
        return {
            'skewness': skewness,
            'kurtosis': kurtosis
        }
    
    def calculate_factor_win_rate(self, factor: pd.Series, returns: pd.Series, 
                                threshold: float = 0.5) -> float:
        """
        计算因子胜率（修复：使用历史因子预测当期收益）
        
        Args:
            factor: 因子值序列
            returns: 收益率序列
            threshold: 胜率阈值
            
        Returns:
            float: 胜率
        """
        # 对齐数据
        factor = factor.dropna()
        returns = returns.loc[factor.index]
        
        if len(factor) < 10:
            return np.nan
        
        # 🚨 修复：使用历史因子值预测当期收益率
        # 原来的错误：factor_sign = np.sign(factor)  # 同期数据
        # 修复后：使用t-1时刻的因子值预测t时刻的收益率
        factor_sign = np.sign(factor.shift(1))  # t-1时刻的因子值
        returns_sign = np.sign(returns)         # t时刻的收益率
        
        # 计算胜率（历史因子预测当期收益的正确性）
        win_rate = (factor_sign == returns_sign).mean()
        
        return win_rate
    
    def comprehensive_factor_analysis(self, factor: pd.Series, returns: pd.Series,
                                    factor_name: str = "factor") -> Dict[str, Union[float, Dict]]:
        """
        综合因子分析
        
        Args:
            factor: 因子值序列
            returns: 收益率序列
            factor_name: 因子名称
            
        Returns:
            Dict: 包含所有统计指标的综合分析结果
        """
        analysis = {
            'factor_name': factor_name,
            'data_length': len(factor.dropna()),
            'missing_ratio': factor.isna().mean()
        }
        
        # 基础IC指标
        analysis['ic_pearson'] = self.calculate_ic(factor, returns, 'pearson')
        analysis['ic_spearman'] = self.calculate_ic(factor, returns, 'spearman')
        analysis['ic_kendall'] = self.calculate_ic(factor, returns, 'kendall')
        analysis['mutual_information'] = self.calculate_mutual_information(factor, returns)
        
        # 因子收益率分析
        factor_returns = self.calculate_factor_returns(factor, returns)
        analysis.update(factor_returns)
        
        # 因子特征
        skewness_kurtosis = self.calculate_factor_skewness_kurtosis(factor)
        analysis.update(skewness_kurtosis)
        
        # 因子换手率
        analysis['turnover'] = self.calculate_factor_turnover(factor)
        
        # 因子胜率
        analysis['win_rate'] = self.calculate_factor_win_rate(factor, returns)
        
        # 因子自相关性
        autocorr = self.calculate_factor_autocorrelation(factor)
        analysis.update(autocorr)
        
        return analysis
    
    def batch_factor_analysis(self, factors: pd.DataFrame, returns: pd.Series,
                            progress_bar: bool = True) -> pd.DataFrame:
        """
        批量因子分析
        
        Args:
            factors: 因子DataFrame
            returns: 收益率序列
            progress_bar: 是否显示进度条
            
        Returns:
            pd.DataFrame: 包含所有因子分析结果的DataFrame
        """
        results = []
        
        if progress_bar:
            from tqdm import tqdm
            factor_iterator = tqdm(factors.columns, desc="分析因子")
        else:
            factor_iterator = factors.columns
        
        for factor_name in factor_iterator:
            factor = factors[factor_name]
            analysis = self.comprehensive_factor_analysis(factor, returns, factor_name)
            results.append(analysis)
        
        return pd.DataFrame(results)
    
    def calculate_rolling_ic(self, factor: pd.Series, returns: pd.Series,
                           window: int = 60) -> pd.Series:
        """
        计算滚动IC
        
        Args:
            factor: 因子值序列
            returns: 收益率序列
            window: 滚动窗口大小
            
        Returns:
            pd.Series: 滚动IC序列
        """
        rolling_ic = pd.Series(index=factor.index, dtype=float)
        
        for i in range(window, len(factor)):
            factor_window = factor.iloc[i-window:i]
            returns_window = returns.iloc[i-window:i]
            ic = self.calculate_ic(factor_window, returns_window)
            rolling_ic.iloc[i] = ic
        
        return rolling_ic
    
    def calculate_ic_ir_series(self, rolling_ic: pd.Series, window: int = 20) -> pd.Series:
        """
        计算滚动IR序列
        
        Args:
            rolling_ic: 滚动IC序列
            window: 滚动窗口大小
            
        Returns:
            pd.Series: 滚动IR序列
        """
        rolling_ir = pd.Series(index=rolling_ic.index, dtype=float)
        
        for i in range(window, len(rolling_ic)):
            ic_window = rolling_ic.iloc[i-window:i]
            ir = self.calculate_ic_ir(ic_window)
            rolling_ir.iloc[i] = ir
        
        return rolling_ir
    
    def get_factor_ranking(self, factors: pd.DataFrame, returns: pd.Series,
                          metric: str = 'ic_pearson', top_n: int = 10) -> pd.DataFrame:
        """
        获取因子排名
        
        Args:
            factors: 因子DataFrame
            returns: 收益率序列
            metric: 排序指标
            top_n: 返回前N个因子
            
        Returns:
            pd.DataFrame: 因子排名结果
        """
        # 批量分析
        analysis_df = self.batch_factor_analysis(factors, returns)
        
        # 按指定指标排序
        if metric in analysis_df.columns:
            analysis_df = analysis_df.sort_values(metric, key=abs, ascending=False)
        
        return analysis_df.head(top_n)
    
    def calculate_factor_effectiveness_score(self, factor: pd.Series, returns: pd.Series) -> float:
        """
        计算因子有效性综合评分
        
        Args:
            factor: 因子值序列
            returns: 收益率序列
            
        Returns:
            float: 综合评分 (0-1)
        """
        # 计算各项指标
        ic = abs(self.calculate_ic(factor, returns))
        mi = self.calculate_mutual_information(factor, returns)
        win_rate = self.calculate_factor_win_rate(factor, returns)
        turnover = self.calculate_factor_turnover(factor)
        
        # 标准化指标
        ic_score = min(ic * 10, 1.0)  # IC转换为0-1
        mi_score = min(mi * 5, 1.0)   # 互信息转换为0-1
        win_rate_score = win_rate     # 胜率已经是0-1
        turnover_score = max(0, 1 - turnover)  # 换手率越低越好
        
        # 综合评分 (加权平均)
        effectiveness_score = (
            ic_score * 0.4 +
            mi_score * 0.3 +
            win_rate_score * 0.2 +
            turnover_score * 0.1
        )
        
        return effectiveness_score


class FactorEvaluator:
    """
    因子评估器
    提供高级的因子评估功能
    """
    
    def __init__(self):
        """初始化因子评估器"""
        self.stats = FactorStatistics()
    
    def evaluate_single_factor(self, factor: pd.Series, returns: pd.Series,
                             factor_name: str = "factor") -> Dict:
        """
        评估单个因子
        
        Args:
            factor: 因子值序列
            returns: 收益率序列
            factor_name: 因子名称
            
        Returns:
            Dict: 评估结果
        """
        return self.stats.comprehensive_factor_analysis(factor, returns, factor_name)
    
    def evaluate_multiple_factors(self, factors: pd.DataFrame, returns: pd.Series,
                                metrics: List[str] = None) -> pd.DataFrame:
        """
        评估多个因子
        
        Args:
            factors: 因子DataFrame
            returns: 收益率序列
            metrics: 评估指标列表
            
        Returns:
            pd.DataFrame: 评估结果
        """
        if metrics is None:
            metrics = ['ic_pearson', 'ic_spearman', 'mutual_information', 
                      'long_short_return', 'sharpe_ratio', 'win_rate']
        
        # 批量分析
        analysis_df = self.stats.batch_factor_analysis(factors, returns)
        
        # 选择指定指标
        if metrics:
            available_metrics = [m for m in metrics if m in analysis_df.columns]
            analysis_df = analysis_df[['factor_name'] + available_metrics]
        
        return analysis_df
    
    def get_best_factors(self, factors: pd.DataFrame, returns: pd.Series,
                        metric: str = 'ic_pearson', top_n: int = 10) -> pd.DataFrame:
        """
        获取最佳因子
        
        Args:
            factors: 因子DataFrame
            returns: 收益率序列
            metric: 排序指标
            top_n: 返回前N个因子
            
        Returns:
            pd.DataFrame: 最佳因子列表
        """
        return self.stats.get_factor_ranking(factors, returns, metric, top_n)
    
    def calculate_factor_effectiveness_scores(self, factors: pd.DataFrame, 
                                            returns: pd.Series) -> pd.Series:
        """
        计算所有因子的有效性评分
        
        Args:
            factors: 因子DataFrame
            returns: 收益率序列
            
        Returns:
            pd.Series: 因子有效性评分
        """
        scores = {}
        
        for factor_name in factors.columns:
            factor = factors[factor_name]
            score = self.stats.calculate_factor_effectiveness_score(factor, returns)
            scores[factor_name] = score
        
        return pd.Series(scores)
    
    def generate_factor_report(self, factors: pd.DataFrame, returns: pd.Series,
                             output_file: str = None) -> str:
        """
        生成因子评估报告
        
        Args:
            factors: 因子DataFrame
            returns: 收益率序列
            output_file: 输出文件路径
            
        Returns:
            str: 报告内容
        """
        # 获取最佳因子
        best_factors = self.get_best_factors(factors, returns, 'ic_pearson', 20)
        
        # 计算有效性评分
        effectiveness_scores = self.calculate_factor_effectiveness_scores(factors, returns)
        
        # 生成报告
        report = []
        report.append("=" * 80)
        report.append("因子评估报告")
        report.append("=" * 80)
        report.append(f"评估时间: {pd.Timestamp.now()}")
        report.append(f"因子总数: {len(factors.columns)}")
        report.append(f"数据长度: {len(returns)}")
        report.append("")
        
        # 整体统计
        ic_values = []
        for factor_name in factors.columns:
            factor = factors[factor_name]
            ic = self.stats.calculate_ic(factor, returns)
            if not np.isnan(ic):
                ic_values.append(ic)
        
        if ic_values:
            ic_series = pd.Series(ic_values)
            report.append("整体IC统计:")
            report.append(f"  IC均值: {ic_series.mean():.4f}")
            report.append(f"  IC标准差: {ic_series.std():.4f}")
            report.append(f"  IC IR: {self.stats.calculate_ic_ir(ic_series):.4f}")
            report.append(f"  IC正比例: {self.stats.calculate_ic_positive_ratio(ic_series):.4f}")
            report.append("")
        
        # 最佳因子
        report.append("最佳因子排名 (按IC绝对值):")
        report.append("-" * 60)
        for i, (_, row) in enumerate(best_factors.iterrows()):
            factor_name = row['factor_name']
            ic = row.get('ic_pearson', 0)
            effectiveness = effectiveness_scores.get(factor_name, 0)
            report.append(f"{i+1:2d}. {factor_name:25s} | IC: {ic:6.4f} | 有效性: {effectiveness:.3f}")
        
        report.append("")
        report.append("=" * 80)
        
        report_content = "\n".join(report)
        
        # 保存报告
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
        
        return report_content


# 便捷函数
def calculate_ic(factor: pd.Series, returns: pd.Series, method: str = 'pearson') -> float:
    """便捷函数：计算IC"""
    stats = FactorStatistics()
    return stats.calculate_ic(factor, returns, method)


def calculate_ic_ir(ic_series: pd.Series) -> float:
    """便捷函数：计算IR"""
    stats = FactorStatistics()
    return stats.calculate_ic_ir(ic_series)


def calculate_factor_effectiveness_score(factor: pd.Series, returns: pd.Series) -> float:
    """便捷函数：计算因子有效性评分"""
    stats = FactorStatistics()
    return stats.calculate_factor_effectiveness_score(factor, returns)


def evaluate_factors(factors: pd.DataFrame, returns: pd.Series) -> pd.DataFrame:
    """便捷函数：评估因子"""
    evaluator = FactorEvaluator()
    return evaluator.evaluate_multiple_factors(factors, returns) 