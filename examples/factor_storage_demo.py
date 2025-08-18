#!/usr/bin/env python3
"""
因子存储演示
演示因子存储的基本操作
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from factor_miner.core.factor_storage import TransparentFactorStorage


def demo_basic_operations():
    """演示基本存储操作"""
    print("=" * 60)
    print("1. 基本存储操作演示")
    print("=" * 60)
    
    storage = TransparentFactorStorage()
    
    # 创建测试数据
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
    data = pd.DataFrame({
        'open': np.random.randn(len(dates)).cumsum() + 100,
        'high': np.random.randn(len(dates)).cumsum() + 102,
        'low': np.random.randn(len(dates)).cumsum() + 98,
        'close': np.random.randn(len(dates)).cumsum() + 100,
        'volume': np.random.exponential(1000, len(dates))
    }, index=dates)
    
    print(f"测试数据形状: {data.shape}")
    print(f"时间范围: {data.index.min()} 到 {data.index.max()}")
    
    return storage, data


def demo_factor_save_load():
    """演示因子的保存和加载"""
    print("\n" + "=" * 60)
    print("2. 因子保存和加载演示")
    print("=" * 60)
    
    storage, data = demo_basic_operations()
    
    # 保存一个简单的公式因子
    formula = "data['close'].pct_change()"
    success = storage.save_formula_factor(
        factor_id='test_returns',
        name='测试收益率',
        formula=formula,
        description='简单的价格收益率计算',
        category='test',
        parameters={}
    )
    
    if success:
        print("✅ 测试因子保存成功")
        
        # 加载因子定义
        factor_def = storage.load_factor_definition('test_returns')
        if factor_def:
            print("✅ 因子定义加载成功")
            print(f"  名称: {factor_def.name}")
            print(f"  描述: {factor_def.description}")
            print(f"  类别: {factor_def.category}")
            print(f"  计算类型: {factor_def.computation_type}")
            print(f"  参数: {factor_def.parameters}")
        else:
            print("❌ 因子定义加载失败")
    else:
        print("❌ 测试因子保存失败")


def demo_factor_metadata():
    """演示因子元数据管理"""
    print("\n" + "=" * 60)
    print("3. 因子元数据管理演示")
    print("=" * 60)
    
    storage, data = demo_basic_operations()
    
    # 保存带详细元数据的因子
    function_code = '''
def calculate(data, **kwargs):
    """计算移动平均线"""
    period = kwargs.get('period', 20)
    return data['close'].rolling(window=period).mean()
'''
    
    success = storage.save_function_factor(
        factor_id='custom_ma',
        name='自定义移动平均',
        function_code=function_code,
        description='可配置周期的移动平均线指标',
        category='technical',
        parameters={'period': 20},
        imports=['import pandas as pd']
    )
    
    if success:
        print("✅ 自定义因子保存成功")
        
        # 加载并查看元数据
        factor_def = storage.load_factor_definition('custom_ma')
        if factor_def:
            print("✅ 因子元数据:")
            print(f"  创建时间: {factor_def.metadata.get('created_at', 'N/A')}")
            print(f"  校验和: {factor_def.metadata.get('checksum', 'N/A')}")
            print(f"  依赖: {factor_def.dependencies}")
            print(f"  输出类型: {factor_def.output_type}")
        else:
            print("❌ 因子元数据加载失败")
    else:
        print("❌ 自定义因子保存失败")


def demo_storage_status():
    """演示存储状态查询"""
    print("\n" + "=" * 60)
    print("4. 存储状态查询演示")
    print("=" * 60)
    
    storage, data = demo_basic_operations()
    
    # 列出所有因子
    factors = storage.list_factors()
    print(f"因子库中共有 {len(factors)} 个因子")
    
    if factors:
        print("前10个因子:")
        for i, factor_id in enumerate(factors[:10], 1):
            print(f"  {i}. {factor_id}")
        
        if len(factors) > 10:
            print(f"  ... 还有 {len(factors) - 10} 个因子")
        
        # 检查特定因子是否存在
        test_factor = 'test_returns'
        if test_factor in factors:
            print(f"\n✅ 因子 '{test_factor}' 存在于因子库中")
        else:
            print(f"\n❌ 因子 '{test_factor}' 不存在于因子库中")
    else:
        print("因子库为空")


def demo_storage_cleanup():
    """演示存储清理操作"""
    print("\n" + "=" * 60)
    print("5. 存储清理演示")
    print("=" * 60)
    
    storage, data = demo_basic_operations()
    
    # 列出当前因子
    factors_before = storage.list_factors()
    print(f"清理前因子数量: {len(factors_before)}")
    
    # 删除测试因子（如果存在）
    test_factors = ['test_returns', 'custom_ma']
    for factor_id in test_factors:
        if factor_id in factors_before:
            # 注意：这里只是演示，实际删除需要实现delete方法
            print(f"⚠️  测试因子 '{factor_id}' 将被标记为删除（需要实现delete方法）")
    
    print("✅ 存储清理演示完成")
    print("注意：实际的因子删除功能需要在TransparentFactorStorage中实现")


def main():
    """主函数"""
    print("🚀 FactorMiner 因子存储系统演示")
    print("=" * 60)
    
    try:
        # 演示各种存储操作
        demo_factor_save_load()
        demo_factor_metadata()
        demo_storage_status()
        demo_storage_cleanup()
        
        print("\n" + "=" * 60)
        print("✅ 所有演示完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
