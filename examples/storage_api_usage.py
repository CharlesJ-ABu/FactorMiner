#!/usr/bin/env python3
"""
因子存储API使用示例

展示如何使用新的简洁API接口来保存不同类型的因子
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from factor_miner.core.factor_storage import get_global_storage

def example_save_technical_factor():
    """示例：保存技术指标因子"""
    print("=== 保存技术指标因子 ===")
    
    storage = get_global_storage()
    
    # 定义因子代码
    function_code = '''
def calculate(data: pd.DataFrame, **kwargs) -> pd.Series:
    """计算简单移动平均因子"""
    return data['close'].rolling(20).mean()
'''
    
    # 保存技术指标因子
    success = storage.save_technical_factor(
        factor_id="sma_20",
        name="20日简单移动平均",
        function_code=function_code,
        description="计算20日简单移动平均线",
        category="technical",
        imports=["import pandas as pd"]
    )
    
    print(f"保存结果: {'成功' if success else '失败'}")
    return success

def example_save_minactor_factor():
    """示例：保存挖掘因子"""
    print("\n=== 保存挖掘因子 ===")
    
    storage = get_global_storage()
    
    # 保存挖掘因子定义
    success = storage.save_minactor_factor(
        factor_id="ml_momentum_5d",
        name="ML动量因子5日",
        algorithm_name="ml_momentum_factor",
        model_file="ml_momentum_5d.pkl",
        description="基于机器学习的5日动量因子",
        category="ml",
        performance_metrics={
            "ic": 0.15,
            "sharpe": 1.2,
            "win_rate": 0.65
        }
    )
    
    print(f"保存结果: {'成功' if success else '失败'}")
    return success

def example_save_model():
    """示例：保存模型文件"""
    print("\n=== 保存模型文件 ===")
    
    storage = get_global_storage()
    
    # 模拟模型数据（实际使用中应该是真实的模型数据）
    import pickle
    model_data = pickle.dumps({"model": "dummy_model", "version": "1.0"})
    
    # 保存模型文件
    success = storage.save_model(
        factor_id="ml_momentum_5d",
        model_data=model_data,
        model_type="pkl"
    )
    
    print(f"保存结果: {'成功' if success else '失败'}")
    return success

def example_save_evaluation():
    """示例：保存评估结果"""
    print("\n=== 保存评估结果 ===")
    
    storage = get_global_storage()
    
    # 评估数据
    evaluation_data = {
        "ic_pearson": 0.15,
        "ic_spearman": 0.12,
        "sharpe_ratio": 1.2,
        "win_rate": 0.65,
        "long_short_return": 0.08
    }
    
    # 保存评估结果
    success = storage.save_evaluation(
        factor_id="ml_momentum_5d",
        evaluation_data=evaluation_data,
        source="minactors"  # 或 "technicals"
    )
    
    print(f"保存结果: {'成功' if success else '失败'}")
    return success

def example_save_mining_history():
    """示例：保存挖掘历史"""
    print("\n=== 保存挖掘历史 ===")
    
    storage = get_global_storage()
    
    # 挖掘会话数据
    session_data = {
        "session_id": "test_session_001",
        "config": {
            "symbols": ["BTCUSDT"],
            "timeframes": ["1h"],
            "algorithms": ["ml_momentum_factor"]
        },
        "results": {
            "total_factors": 3,
            "algorithms_used": ["ml_momentum_factor"],
            "factors": {
                "ml_momentum_5d": "factor_data_here",
                "ml_momentum_10d": "factor_data_here",
                "ml_momentum_20d": "factor_data_here"
            }
        },
        "status": "completed",
        "completed_time": "2024-01-01T12:00:00"
    }
    
    # 保存挖掘历史
    success = storage.save_mining_history(
        session_id="test_session_001",
        session_data=session_data
    )
    
    print(f"保存结果: {'成功' if success else '失败'}")
    return success

def main():
    """主函数：运行所有示例"""
    print("因子存储API使用示例")
    print("=" * 50)
    
    try:
        # 运行所有示例
        results = []
        results.append(example_save_technical_factor())
        results.append(example_save_minactor_factor())
        results.append(example_save_model())
        results.append(example_save_evaluation())
        results.append(example_save_mining_history())
        
        # 总结
        print("\n" + "=" * 50)
        print("示例运行总结:")
        print(f"总示例数: {len(results)}")
        print(f"成功数: {sum(results)}")
        print(f"失败数: {len(results) - sum(results)}")
        
        if all(results):
            print("✅ 所有示例都运行成功！")
        else:
            print("❌ 部分示例运行失败，请检查错误信息")
            
    except Exception as e:
        print(f"❌ 运行示例时出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
