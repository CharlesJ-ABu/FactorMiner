#!/usr/bin/env python3
"""
透明因子存储演示
展示V3架构的透明因子存储系统
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from factor_miner.core.factor_storage import TransparentFactorStorage
from factor_miner.core.factor_storage import FactorDefinition


def demo_formula_factor():
    """演示公式类因子存储"""
    print("=" * 60)
    print("1. 公式类因子存储演示")
    print("=" * 60)
    
    storage = TransparentFactorStorage()
    
    # 保存公式因子
    formula = "data['close'] / data['close'].shift(20) - 1"
    success = storage.save_formula_factor(
        factor_id='price_momentum_20',
        name='价格动量(20)',
        formula=formula,
        description='20期价格动量指标',
        category='technical',
        parameters={'period': 20}
    )
    
    if success:
        print("✅ 公式因子保存成功")
        print(f"公式: {formula}")
    else:
        print("❌ 公式因子保存失败")


def demo_function_factor():
    """演示函数类因子存储"""
    print("\n" + "=" * 60)
    print("2. 函数类因子存储演示")
    print("=" * 60)
    
    storage = TransparentFactorStorage()
    
    # 函数代码
    function_code = '''
def calculate(data, **kwargs):
    """计算RSI指标"""
    period = kwargs.get('period', 14)
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    return rsi
'''
    
    success = storage.save_function_factor(
        factor_id='custom_rsi',
        name='自定义RSI',
        function_code=function_code,
        description='自定义RSI指标计算',
        category='technical',
        parameters={'period': 14},
        imports=['import pandas as pd', 'import numpy as np']
    )
    
    if success:
        print("✅ 函数因子保存成功")
        print("函数代码已保存到functions目录")
    else:
        print("❌ 函数因子保存失败")


def demo_pipeline_factor():
    """演示ML流水线因子存储"""
    print("\n" + "=" * 60)
    print("3. ML流水线因子存储演示")
    print("=" * 60)
    
    storage = TransparentFactorStorage()
    
    # 流水线步骤
    pipeline_steps = [
        {
            "type": "feature_engineering",
            "code": """
# 特征工程
features = pd.DataFrame(index=data.index)
features['returns'] = data['close'].pct_change()
features['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
features['volatility'] = features['returns'].rolling(10).std()
""",
            "outputs": ["returns", "volume_ratio", "volatility"]
        },
        {
            "type": "model",
            "algorithm": "LinearRegression",
            "parameters": {"fit_intercept": True},
            "features": ["returns", "volume_ratio", "volatility"],
            "target": "next_return"
        }
    ]
    
    success = storage.save_pipeline_factor(
        factor_id='ml_prediction_pipeline',
        name='ML预测流水线',
        pipeline_steps=pipeline_steps,
        description='机器学习预测流水线',
        category='ml',
        parameters={'window': 20}
    )
    
    if success:
        print("✅ ML流水线因子保存成功")
        print("流水线定义已保存到pipelines目录")
    else:
        print("❌ ML流水线因子保存失败")


def demo_ml_model_factor():
    """演示ML模型因子存储"""
    print("\n" + "=" * 60)
    print("4. ML模型因子存储演示")
    print("=" * 60)
    
    storage = TransparentFactorStorage()
    
    # 保存ML模型因子
    success = storage.save_ml_model_factor(
        factor_id='ensemble_random_forest',
        name='集成随机森林',
        artifact_filename='ensemble_random_forest.pkl',
        description='预训练的随机森林模型',
        category='ml',
        parameters={'n_estimators': 100},
        feature_set='basic_v1'
    )
    
    if success:
        print("✅ ML模型因子保存成功")
        print("模型引用已保存，artifact文件应放在models目录")
    else:
        print("❌ ML模型因子保存失败")


def demo_factor_loading():
    """演示因子加载功能"""
    print("\n" + "=" * 60)
    print("5. 因子加载演示")
    print("=" * 60)
    
    storage = TransparentFactorStorage()
    
    # 列出所有因子
    factors = storage.list_factors()
    print(f"因子库中共有 {len(factors)} 个因子:")
    for factor_id in factors[:10]:  # 显示前10个
        print(f"  - {factor_id}")
    
    if len(factors) > 10:
        print(f"  ... 还有 {len(factors) - 10} 个因子")
    
    # 加载特定因子定义
    if factors:
        factor_id = factors[0]
        factor_def = storage.load_factor_definition(factor_id)
        if factor_def:
            print(f"\n因子 {factor_id} 的定义:")
            print(f"  名称: {factor_def.name}")
            print(f"  类别: {factor_def.category}")
            print(f"  计算类型: {factor_def.computation_type}")
        else:
            print(f"无法加载因子定义: {factor_id}")


def main():
    """主函数"""
    print("🚀 FactorMiner V3 透明因子存储系统演示")
    print("=" * 60)
    
    try:
        # 演示各种因子存储方式
        demo_formula_factor()
        demo_function_factor()
        demo_pipeline_factor()
        demo_ml_model_factor()
        demo_factor_loading()
        
        print("\n" + "=" * 60)
        print("✅ 所有演示完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
