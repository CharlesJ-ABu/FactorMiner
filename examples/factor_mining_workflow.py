#!/usr/bin/env python3
"""
FactorMiner 因子挖掘工作流示例
展示如何使用V3架构进行完整的因子挖掘流程
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from factor_miner.core.factor_engine import get_global_engine
from factor_miner.core.factor_storage import get_global_storage
from factor_miner.core.data_loader import DataLoader


def main():
    """主函数：演示完整的因子挖掘工作流"""
    print("=== FactorMiner 因子挖掘工作流示例 ===\n")
    
    # 1. 初始化核心组件
    print("1. 初始化核心组件...")
    storage = get_global_storage()
    engine = get_global_engine()
    data_loader = DataLoader()
    
    print("✅ 核心组件初始化完成")
    
    # 2. 加载市场数据
    print("\n2. 加载市场数据...")
    try:
        # 尝试加载BTC数据
        data_result = data_loader.load_data(
            symbol='BTC_USDT',
            timeframe='1h',
            start_date='2024-01-01',
            end_date='2024-01-31'
        )
        
        if data_result['success']:
            data = data_result['data']
            print(f"✅ 数据加载成功")
            print(f"   数据形状: {data.shape}")
            print(f"   时间范围: {data.index.min()} 到 {data.index.max()}")
            print(f"   数据列: {list(data.columns)}")
        else:
            print(f"❌ 数据加载失败: {data_result['error']}")
            print("使用模拟数据...")
            data = create_sample_data()
    except Exception as e:
        print(f"❌ 数据加载异常: {e}")
        print("使用模拟数据...")
        data = create_sample_data()
    
    # 3. 构建技术因子
    print("\n3. 构建技术因子...")
    try:
        from factor_miner.factors.technical import TechnicalFactorBuilder
        
        technical_builder = TechnicalFactorBuilder()
        technical_factors = technical_builder.calculate_all_factors(data)
        
        print(f"✅ 技术因子构建成功")
        print(f"   因子数量: {len(technical_factors.columns)}")
        print(f"   因子列表: {list(technical_factors.columns)}")
        
        # 保存技术因子
        for factor_name in technical_factors.columns:
            factor_data = technical_factors[factor_name].dropna()
            if len(factor_data) > 0:
                storage.save_formula_factor(
                    factor_id=f"demo_{factor_name}",
                    name=f"Demo {factor_name}",
                    formula=f"# 技术因子: {factor_name}",
                    description=f"演示用技术因子: {factor_name}",
                    category="technical"
                )
        
    except Exception as e:
        print(f"❌ 技术因子构建失败: {e}")
    
    # 4. 构建统计因子
    print("\n4. 构建统计因子...")
    try:
        from factor_miner.factors.statistical import StatisticalFactorBuilder
        
        statistical_builder = StatisticalFactorBuilder()
        statistical_factors = statistical_builder.calculate_all_factors(data)
        
        print(f"✅ 统计因子构建成功")
        print(f"   因子数量: {len(statistical_factors.columns)}")
        print(f"   因子列表: {list(statistical_factors.columns)}")
        
    except Exception as e:
        print(f"❌ 统计因子构建失败: {e}")
    
    # 5. 因子评估
    print("\n5. 因子评估...")
    try:
        # 计算收益率作为目标变量
        returns = data['close'].pct_change().shift(-1).dropna()
        
        # 合并所有因子
        all_factors = pd.concat([technical_factors, statistical_factors], axis=1)
        all_factors = all_factors.dropna()
        
        # 对齐数据
        common_index = all_factors.index.intersection(returns.index)
        all_factors = all_factors.loc[common_index]
        returns = returns.loc[common_index]
        
        print(f"✅ 因子评估准备完成")
        print(f"   评估数据点数: {len(common_index)}")
        print(f"   因子数量: {len(all_factors.columns)}")
        
        # 计算IC值
        ic_values = {}
        for factor_name in all_factors.columns:
            factor_series = all_factors[factor_name]
            if len(factor_series.dropna()) > 0:
                ic = factor_series.corr(returns)
                ic_values[factor_name] = ic
        
        # 显示最佳因子
        sorted_ic = sorted(ic_values.items(), key=lambda x: abs(x[1]), reverse=True)
        print(f"\n前5个最佳因子 (按|IC|排序):")
        for i, (factor_name, ic) in enumerate(sorted_ic[:5]):
            print(f"   {i+1}. {factor_name}: IC = {ic:.4f}")
        
    except Exception as e:
        print(f"❌ 因子评估失败: {e}")
    
    # 6. 因子存储状态
    print("\n6. 因子存储状态...")
    try:
        all_factors = storage.list_factors()
        print(f"✅ 存储中的因子总数: {len(all_factors)}")
        
        if all_factors:
            print(f"   示例因子: {all_factors[:5]}")
        
    except Exception as e:
        print(f"❌ 获取因子存储状态失败: {e}")
    
    print("\n=== 因子挖掘工作流示例完成 ===")


def create_sample_data(periods=1000):
    """创建模拟市场数据"""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=periods, freq='H')
    
    # 生成价格数据
    base_price = 50000
    returns = np.random.normal(0, 0.02, periods)
    prices = [base_price]
    
    for i in range(1, periods):
        new_price = prices[-1] * (1 + returns[i])
        prices.append(new_price)
    
    # 创建OHLCV数据
    data = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': np.random.uniform(1000000, 10000000, periods)
    }, index=dates)
    
    # 确保OHLC逻辑
    data['high'] = data[['open', 'close', 'high']].max(axis=1)
    data['low'] = data[['open', 'close', 'low']].min(axis=1)
    
    return data


if __name__ == "__main__":
    main()
