#!/usr/bin/env python3
"""
数据健康度检查器
检查数据的完整性、连续性和质量
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class DataHealthChecker:
    """数据健康度检查器"""
    
    def __init__(self):
        """初始化健康度检查器"""
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
        
        # 健康度阈值配置
        self.health_thresholds = {
            'min_coverage': 95.0,        # 最小覆盖率
            'max_duplicate_ratio': 0.01,  # 最大重复率
            'max_gap_ratio': 0.05,       # 最大断层率
            'max_outlier_ratio': 0.02,   # 最大异常值率
        }
    
    def check_data_health(self, df: pd.DataFrame, expected_timeframe: str, 
                          symbol: str = None) -> Dict:
        """
        检查数据健康度
        
        Args:
            df: 数据DataFrame
            expected_timeframe: 预期的时间框架
            symbol: 交易对符号（可选）
            
        Returns:
            健康度检查报告
        """
        try:
            if df is None or df.empty:
                return self._create_health_report(
                    is_healthy=False,
                    score=0.0,
                    issues=['数据为空或None'],
                    symbol=symbol,
                    timeframe=expected_timeframe
                )
            
            # 基础检查
            basic_checks = self._check_basic_data(df)
            
            # 时间连续性检查
            continuity_checks = self._check_time_continuity(df, expected_timeframe)
            
            # 数据质量检查
            quality_checks = self._check_data_quality(df)
            
            # 计算综合健康度分数
            health_score = self._calculate_health_score(basic_checks, continuity_checks, quality_checks)
            
            # 汇总所有问题
            all_issues = []
            all_issues.extend(basic_checks.get('issues', []))
            all_issues.extend(continuity_checks.get('issues', []))
            all_issues.extend(quality_checks.get('issues', []))
            
            # 判断是否健康 - 只有100分才能保存
            is_healthy = health_score >= 100.0 and len(all_issues) == 0
            
            return self._create_health_report(
                is_healthy=is_healthy,
                score=health_score,
                issues=all_issues,
                symbol=symbol,
                timeframe=expected_timeframe,
                basic_checks=basic_checks,
                continuity_checks=continuity_checks,
                quality_checks=quality_checks
            )
            
        except Exception as e:
            self.logger.error(f"健康度检查失败: {e}")
            return self._create_health_report(
                is_healthy=False,
                score=0.0,
                issues=[f'健康度检查异常: {str(e)}'],
                symbol=symbol,
                timeframe=expected_timeframe
            )
    
    def _check_basic_data(self, df: pd.DataFrame) -> Dict:
        """检查基础数据"""
        issues = []
        warnings = []
        
        # 检查必需的列
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            issues.append(f'缺少必需的列: {missing_columns}')
        
        # 检查数据形状
        if len(df) < 10:
            issues.append(f'数据量过少: {len(df)} 条')
        elif len(df) > 1000000:
            warnings.append(f'数据量过大: {len(df)} 条')
        
        # 检查数据类型
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    issues.append(f'列 {col} 不是数值类型: {df[col].dtype}')
        
        return {
            'status': 'ok' if not issues else 'warning',
            'issues': issues,
            'warnings': warnings,
            'data_shape': df.shape,
            'data_types': {col: str(df[col].dtype) for col in df.columns}
        }
    
    def _check_time_continuity(self, df: pd.DataFrame, timeframe: str) -> Dict:
        """检查时间连续性"""
        issues = []
        warnings = []
        
        # 检查是否有时间列或时间索引
        time_col = None
        if df.index.name in ['date', 'datetime', 'time', 'timestamp']:
            # 如果有时间索引，重置索引
            df_temp = df.reset_index()
            time_col = df.index.name
        else:
            # 检查列中是否有时间列
            for col in ['date', 'datetime', 'time', 'timestamp']:
                if col in df.columns:
                    time_col = col
                    break
        
        if not time_col:
            issues.append('没有找到时间列或时间索引')
            return {
                'status': 'error',
                'issues': issues,
                'coverage': 0.0,
                'gaps_count': 0,
                'gaps': []
            }
        
        try:
            # 转换时间列
            if df.index.name in ['date', 'datetime', 'time', 'timestamp']:
                # 使用重置索引后的数据
                df_time = df_temp.copy()
                df_time[time_col] = pd.to_datetime(df_time[time_col])
                df_time = df_time.set_index(time_col).sort_index()
            else:
                # 使用原始数据
                df_time = df.copy()
                df_time[time_col] = pd.to_datetime(df_time[time_col])
                df_time = df_time.set_index(time_col).sort_index()
            
            # 计算时间跨度
            time_span = df_time.index.max() - df_time.index.min()
            
            # 计算预期数据条数 (修复：包含起始和结束时间点)
            expected_interval = self.timeframe_intervals.get(timeframe, pd.Timedelta('1 minute'))
            
            # 计算预期数据条数
            if timeframe == '1d':
                # 日线：计算天数 + 1
                expected_count = time_span.days + 1
            else:
                # 其他时间框架：计算时间间隔数 + 1
                # 注意：这里需要正确处理时间跨度的计算
                total_seconds = time_span.total_seconds()
                interval_seconds = expected_interval.total_seconds()
                
                if interval_seconds > 0:
                    # 计算完整的间隔数量，然后 +1 (包含起始时间点)
                    expected_count = int(total_seconds / interval_seconds) + 1
                else:
                    expected_count = 1
            

            
            # 计算覆盖率
            actual_count = len(df_time)
            coverage = (actual_count / expected_count * 100) if expected_count > 0 else 0
            
            # 检查时间间隔
            time_diff = df_time.index.to_series().diff()
            expected_interval_td = self.timeframe_intervals.get(timeframe, pd.Timedelta('1 minute'))
            
            # 允许的间隔误差（±10%）
            tolerance = expected_interval_td * 0.1
            min_interval = expected_interval_td - tolerance
            max_interval = expected_interval_td + tolerance
            
            # 检测异常间隔
            abnormal_intervals = time_diff[
                (time_diff < min_interval) | (time_diff > max_interval)
            ]
            
            # 检测大断层
            gap_threshold = expected_interval_td * 3
            large_gaps = time_diff[time_diff > gap_threshold]
            
            # 生成断层信息
            gaps_info = []
            for idx, gap in large_gaps.items():
                gaps_info.append({
                    'position': str(idx),
                    'gap_duration': str(gap),
                    'gap_hours': gap.total_seconds() / 3600,
                    'expected_interval': str(expected_interval_td)
                })
            
            # 判断状态 - 覆盖率过低直接设为error
            if coverage < self.health_thresholds['min_coverage']:
                issues.append(f'数据覆盖率过低: {coverage:.2f}% (期望 >= {self.health_thresholds["min_coverage"]}%)')
                status = 'error'  # 覆盖率过低直接设为error
            elif len(large_gaps) > 0:
                gap_ratio = len(large_gaps) / len(df_time)
                if gap_ratio > self.health_thresholds['max_gap_ratio']:
                    issues.append(f'数据断层过多: {len(large_gaps)} 个断层 ({gap_ratio:.2%})')
                    status = 'error'  # 断层过多直接设为error
                else:
                    warnings.append(f'发现 {len(large_gaps)} 个数据断层')
                    status = 'warning'
            elif len(abnormal_intervals) > 0:
                abnormal_ratio = len(abnormal_intervals) / len(df_time)
                if abnormal_ratio > 0.1:  # 超过10%的异常间隔
                    warnings.append(f'时间间隔异常: {len(abnormal_intervals)} 个异常间隔 ({abnormal_ratio:.2%})')
                    status = 'warning'
                else:
                    status = 'ok'
            else:
                status = 'ok'
            
            return {
                'status': status,
                'issues': issues,
                'warnings': warnings,
                'coverage': round(coverage, 2),
                'expected_count': round(expected_count, 2),
                'actual_count': actual_count,
                'time_span': str(time_span),
                'gaps_count': len(large_gaps),
                'gaps': gaps_info,
                'abnormal_intervals_count': len(abnormal_intervals)
            }
            
        except Exception as e:
            issues.append(f'时间连续性检查失败: {str(e)}')
            return {
                'status': 'error',
                'issues': issues,
                'coverage': 0.0,
                'gaps_count': 0,
                'gaps': []
            }
    
    def _check_data_quality(self, df: pd.DataFrame) -> Dict:
        """检查数据质量"""
        issues = []
        warnings = []
        
        # 检查缺失值
        missing_stats = {}
        for col in df.columns:
            missing_count = df[col].isna().sum()
            missing_ratio = missing_count / len(df)
            missing_stats[col] = {
                'count': missing_count,
                'ratio': missing_ratio
            }
            
            if missing_ratio > 0.1:  # 超过10%的缺失值
                issues.append(f'列 {col} 缺失值过多: {missing_ratio:.2%}')
            elif missing_ratio > 0.01:  # 超过1%的缺失值
                warnings.append(f'列 {col} 有缺失值: {missing_ratio:.2%}')
        
        # 检查异常值（基于OHLCV数据）
        if 'open' in df.columns and 'high' in df.columns and 'low' in df.columns and 'close' in df.columns:
            # 检查OHLC逻辑关系
            invalid_ohlc = (
                (df['high'] < df['low']) |
                (df['open'] > df['high']) |
                (df['close'] < df['low']) |
                (df['open'] < df['low']) |
                (df['close'] > df['high'])
            )
            
            invalid_count = invalid_ohlc.sum() if 'invalid_count' in locals() else 0
            if invalid_count > 0:
                invalid_ratio = invalid_count / len(df)
                if invalid_ratio > self.health_thresholds['max_outlier_ratio']:
                    issues.append(f'OHLC数据逻辑错误: {invalid_count} 条记录 ({invalid_ratio:.2%})')
                else:
                    warnings.append(f'发现 {invalid_count} 条OHLC逻辑错误')
        
        # 检查价格合理性
        if 'close' in df.columns:
            # 检查价格是否为0或负数
            zero_or_negative = (df['close'] <= 0).sum()
            if zero_or_negative > 0:
                issues.append(f'发现 {zero_or_negative} 条价格为0或负数的记录')
            
            # 检查价格变化幅度是否合理
            if len(df) > 1:
                price_changes = df['close'].pct_change().abs()
                extreme_changes = (price_changes > 0.5).sum()  # 超过50%的价格变化
                if extreme_changes > 0:
                    extreme_ratio = extreme_changes / len(df)
                    if extreme_ratio > 0.01:  # 超过1%的极端变化
                        warnings.append(f'发现 {extreme_changes} 条极端价格变化记录 ({extreme_ratio:.2%})')
        
        # 检查成交量合理性
        if 'volume' in df.columns:
            # 检查成交量是否为负数
            negative_volume = (df['volume'] < 0).sum()
            if negative_volume > 0:
                issues.append(f'发现 {negative_volume} 条成交量为负数的记录')
        
        return {
            'status': 'ok' if not issues else 'warning',
            'issues': issues,
            'warnings': warnings,
            'missing_stats': missing_stats,
            'data_quality_score': self._calculate_quality_score(issues, warnings)
        }
    
    def _calculate_quality_score(self, issues: List[str], warnings: List[str]) -> float:
        """计算数据质量分数"""
        base_score = 100.0
        
        # 每个严重问题扣10分
        base_score -= len(issues) * 10
        
        # 每个警告扣2分
        base_score -= len(warnings) * 2
        
        return max(0.0, base_score)
    
    def _calculate_health_score(self, basic_checks: Dict, continuity_checks: Dict, 
                               quality_checks: Dict) -> float:
        """计算综合健康度分数"""
        scores = []
        
        # 基础数据检查分数
        if basic_checks['status'] == 'ok':
            scores.append(100.0)
        elif basic_checks['status'] == 'warning':
            scores.append(70.0)
        else:
            scores.append(30.0)
        
        # 时间连续性分数 - 覆盖率过低直接给0分
        if continuity_checks['status'] == 'ok':
            scores.append(100.0)
        elif continuity_checks['status'] == 'warning':
            coverage = continuity_checks.get('coverage', 0)
            scores.append(max(50.0, coverage))
        elif continuity_checks['status'] == 'error':
            scores.append(0.0)  # error状态直接给0分
        else:
            scores.append(20.0)
        
        # 数据质量分数
        quality_score = quality_checks.get('data_quality_score', 50.0)
        scores.append(quality_score)
        
        # 计算加权平均分数
        weights = [0.3, 0.4, 0.3]  # 基础、连续性、质量的权重
        final_score = sum(score * weight for score, weight in zip(scores, weights))
        
        return round(final_score, 2)
    
    def _create_health_report(self, is_healthy: bool, score: float, issues: List[str],
                             symbol: str = None, timeframe: str = None,
                             basic_checks: Dict = None, continuity_checks: Dict = None,
                             quality_checks: Dict = None) -> Dict:
        """创建健康度报告"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'is_healthy': is_healthy,
            'health_score': score,
            'health_level': self._get_health_level(score),
            'summary': self._generate_summary(is_healthy, score, issues),
            'issues': issues,
            'recommendations': self._generate_recommendations(issues),
            'metadata': {
                'symbol': symbol,
                'timeframe': timeframe,
                'checker_version': '1.0.0'
            }
        }
        
        if basic_checks:
            report['basic_checks'] = basic_checks
        if continuity_checks:
            report['continuity_checks'] = continuity_checks
        if quality_checks:
            report['quality_checks'] = quality_checks
        
        return report
    
    def _get_health_level(self, score: float) -> str:
        """根据分数获取健康等级"""
        if score >= 100:
            return 'excellent'  # 只有100分才是excellent
        elif score >= 90:
            return 'good'
        elif score >= 80:
            return 'fair'
        elif score >= 70:
            return 'poor'
        else:
            return 'critical'
    
    def _generate_summary(self, is_healthy: bool, score: float, issues: List[str]) -> str:
        """生成健康度摘要"""
        if is_healthy:
            if score >= 100:
                return f"数据完美健康，健康度分数: {score} - 可以保存"
            else:
                return f"数据基本健康，健康度分数: {score}，有少量问题需要关注"
        else:
            if score >= 80:
                return f"数据质量一般，健康度分数: {score}，需要修复后才能保存"
            elif score >= 60:
                return f"数据质量较差，健康度分数: {score}，建议重新下载"
            else:
                return f"数据质量很差，健康度分数: {score}，需要立即重新下载"
    
    def _generate_recommendations(self, issues: List[str]) -> List[str]:
        """生成修复建议"""
        recommendations = []
        
        for issue in issues:
            if '缺少必需的列' in issue:
                recommendations.append('检查数据源，确保包含所有必需的列')
            elif '数据量过少' in issue:
                recommendations.append('重新下载数据，确保时间范围完整')
            elif '数据覆盖率过低' in issue:
                recommendations.append('检查数据下载是否完整，可能需要补全缺失时间段')
            elif '数据断层过多' in issue:
                recommendations.append('使用数据补全功能，下载缺失时间段的数据')
            elif 'OHLC数据逻辑错误' in issue:
                recommendations.append('检查数据源质量，可能需要重新下载或清洗数据')
            elif '价格为0或负数' in issue:
                recommendations.append('检查数据源，过滤无效价格数据')
            elif '成交量为负数' in issue:
                recommendations.append('检查数据源，过滤无效成交量数据')
            else:
                recommendations.append('建议进行详细的数据质量分析')
        
        if not recommendations:
            recommendations.append('数据质量良好，无需特殊处理')
        
        return recommendations


# 创建全局实例
health_checker = DataHealthChecker()
