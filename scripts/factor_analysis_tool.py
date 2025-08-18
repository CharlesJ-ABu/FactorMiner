#!/usr/bin/env python3
"""
因子分析工具
分析因子库中的因子质量和性能
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from factor_miner.core.factor_engine import get_global_engine
from factor_miner.core.factor_storage import get_global_storage
from factor_miner.core.data_loader import DataLoader


def analyze_factor_quality():
    """分析因子质量"""
    print("=== 因子质量分析 ===\n")
    
    storage = get_global_storage()
    engine = get_global_engine()
    
    # 获取所有因子
    all_factors = storage.list_factors()
    print(f"总因子数量: {len(all_factors)}")
    
    if not all_factors:
        print("❌ 没有找到任何因子")
        return
    
    # 分析因子类型分布
    factor_types = {}
    factor_categories = {}
    
    for factor_id in all_factors:
        try:
            factor_info = storage.get_factor_info(factor_id)
            if factor_info:
                # 统计计算类型
                comp_type = factor_info.get('computation_type', 'unknown')
                factor_types[comp_type] = factor_types.get(comp_type, 0) + 1
                
                # 统计分类
                category = factor_info.get('category', 'unknown')
                factor_categories[category] = factor_categories.get(category, 0) + 1
        except Exception as e:
            print(f"获取因子 {factor_id} 信息失败: {e}")
    
    print("\n📊 因子类型分布:")
    for comp_type, count in sorted(factor_types.items()):
        print(f"   {comp_type}: {count} 个")
    
    print("\n📊 因子分类分布:")
    for category, count in sorted(factor_categories.items()):
        print(f"   {category}: {count} 个")
    
    return all_factors


def analyze_factor_performance(sample_data=None):
    """分析因子性能"""
    print("\n=== 因子性能分析 ===\n")
    
    if sample_data is None:
        print("❌ 没有样本数据，无法进行性能分析")
        return
    
    storage = get_global_storage()
    engine = get_global_engine()
    
    # 计算收益率作为目标变量
    returns = sample_data['close'].pct_change().shift(-1).dropna()
    
    # 分析每个因子
    performance_results = []
    
    for factor_id in storage.list_factors()[:20]:  # 限制分析前20个因子
        try:
            # 计算因子值
            factor_series = engine.compute_single_factor(factor_id, sample_data)
            
            if factor_series is not None and len(factor_series.dropna()) > 0:
                # 对齐数据
                common_index = factor_series.index.intersection(returns.index)
                if len(common_index) > 10:  # 至少需要10个数据点
                    factor_aligned = factor_series.loc[common_index]
                    returns_aligned = returns.loc[common_index]
                    
                    # 计算性能指标
                    ic = factor_aligned.corr(returns_aligned)
                    ic_abs = abs(ic)
                    
                    # 计算胜率
                    factor_rank = factor_aligned.rank(pct=True)
                    returns_rank = returns_aligned.rank(pct=True)
                    win_rate = (factor_rank > 0.5) == (returns_rank > 0.5)
                    win_rate = win_rate.mean()
                    
                    # 计算稳定性（IC的标准差）
                    ic_rolling = factor_aligned.rolling(20).corr(returns_aligned)
                    ic_stability = 1 / (ic_rolling.std() + 1e-6)
                    
                    performance_results.append({
                        'factor_id': factor_id,
                        'ic': ic,
                        'ic_abs': ic_abs,
                        'win_rate': win_rate,
                        'stability': ic_stability.mean(),
                        'data_points': len(common_index)
                    })
                    
        except Exception as e:
            print(f"分析因子 {factor_id} 失败: {e}")
    
    if performance_results:
        # 转换为DataFrame并排序
        perf_df = pd.DataFrame(performance_results)
        perf_df = perf_df.sort_values('ic_abs', ascending=False)
        
        print("🏆 前10个最佳因子 (按|IC|排序):")
        print(perf_df.head(10).to_string(index=False))
        
        # 保存结果
        exports_dir = Path("factorlib/exports")
        exports_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = exports_dir / f"factor_performance_analysis_{timestamp}.csv"
        perf_df.to_csv(output_file, index=False)
        print(f"\n📁 分析结果已保存到: {output_file}")
        
        return perf_df
    else:
        print("❌ 没有成功的性能分析结果")
        return None


def generate_factor_report():
    """生成因子库报告"""
    print("\n=== 生成因子库报告 ===\n")
    
    storage = get_global_storage()
    
    # 收集报告数据
    report_data = {
        'timestamp': datetime.now().isoformat(),
        'total_factors': len(storage.list_factors()),
        'factor_details': []
    }
    
    for factor_id in storage.list_factors():
        try:
            factor_info = storage.get_factor_info(factor_id)
            if factor_info:
                report_data['factor_details'].append({
                    'factor_id': factor_id,
                    'name': factor_info.get('name', 'Unknown'),
                    'category': factor_info.get('category', 'Unknown'),
                    'computation_type': factor_info.get('computation_type', 'Unknown'),
                    'description': factor_info.get('description', ''),
                    'created_at': factor_info.get('created_at', ''),
                    'updated_at': factor_info.get('updated_at', '')
                })
        except Exception as e:
            print(f"获取因子 {factor_id} 详细信息失败: {e}")
    
    # 保存报告
    exports_dir = Path("factorlib/exports")
    exports_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = exports_dir / f"factor_library_report_{timestamp}.json"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
    
    print(f"📁 因子库报告已保存到: {report_file}")
    
    # 显示摘要
    print(f"\n📋 报告摘要:")
    print(f"   总因子数量: {report_data['total_factors']}")
    print(f"   生成时间: {report_data['timestamp']}")
    
    return report_file


def main():
    """主函数"""
    print("🔍 FactorMiner 因子分析工具")
    print("=" * 50)
    
    try:
        # 1. 分析因子质量
        all_factors = analyze_factor_quality()
        
        # 2. 尝试加载样本数据进行性能分析
        print("\n📊 尝试加载样本数据进行性能分析...")
        data_loader = DataLoader()
        
        # 尝试加载BTC数据
        data_result = data_loader.load_data(
            symbol='BTC_USDT',
            timeframe='1h',
            trade_type='futures',
            start_date='2024-01-01',
            end_date='2024-01-31'
        )
        
        if data_result['success']:
            sample_data = data_result['data']
            print(f"✅ 样本数据加载成功: {sample_data.shape}")
            
            # 进行性能分析
            performance_df = analyze_factor_performance(sample_data)
        else:
            print(f"❌ 样本数据加载失败: {data_result['error']}")
            print("💡 跳过性能分析，仅进行质量分析")
        
        # 3. 生成因子库报告
        report_file = generate_factor_report()
        
        print("\n✅ 因子分析完成!")
        print(f"📁 报告文件: {report_file}")
        
    except Exception as e:
        print(f"❌ 因子分析失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
