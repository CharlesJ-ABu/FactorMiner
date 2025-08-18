#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子挖掘主脚本 V3.0
使用新的V3透明因子存储系统进行因子挖掘
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
import json

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from factor_miner.api.factor_mining_api import FactorMiningAPI
from config import settings


def create_mining_config():
    """创建默认挖掘配置"""
    return {
        'factor_types': ['technical', 'statistical', 'advanced', 'ml', 'crypto', 'pattern', 'composite', 'sentiment'],
        'factor_params': {
            'save_to_storage': True,
            'auto_optimize': True
        },
        'optimization': {
            'method': 'greedy',
            'max_factors': 15,
            'min_ic': 0.02,
            'min_ir': 0.1
        },
        'evaluation': {
            'min_sample_size': 30,
            'metrics': ['ic_pearson', 'ic_spearman', 'sharpe_ratio', 'win_rate', 'factor_decay']
        }
    }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='FactorMiner V3.0 因子挖掘工具')
    parser.add_argument('--symbol', type=str, default='BTC_USDT', help='交易对')
    parser.add_argument('--timeframe', type=str, default='1h', help='时间框架')
    parser.add_argument('--factor-types', nargs='+', 
                       default=['technical', 'statistical', 'advanced', 'ml', 'crypto', 'pattern', 'composite', 'sentiment'],
                       help='因子类型')
    parser.add_argument('--start-date', type=str, default=None, help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None, help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--config', type=str, default=None, help='配置文件路径')
    parser.add_argument('--output', type=str, default=None, help='输出文件路径')
    parser.add_argument('--history', action='store_true', help='显示挖掘历史')
    parser.add_argument('--load-result', type=str, default=None, help='加载指定的挖掘结果')
    
    args = parser.parse_args()
    
    print("=== FactorMiner V3.0 因子挖掘工具 ===")
    print(f"交易对: {args.symbol}")
    print(f"时间框架: {args.timeframe}")
    print(f"因子类型: {args.factor_types}")
    
    # 设置默认时间范围（如果没有指定）
    if not args.start_date:
        args.start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
    if not args.end_date:
        args.end_date = datetime.now().strftime('%Y-%m-%d')
    
    print(f"时间范围: {args.start_date} 到 {args.end_date}")
    print()
    
    # 初始化API
    api = FactorMiningAPI()
    
    # 处理特殊命令
    if args.history:
        print("📚 挖掘历史:")
        history = api.get_mining_history()
        if history:
            for i, record in enumerate(history[:10], 1):  # 显示最近10条
                print(f"{i}. {record['symbol']} - {record['timestamp']} - {record['factors_count']} 个因子")
        else:
            print("暂无挖掘历史")
        return
    
    if args.load_result:
        print(f"📂 加载挖掘结果: {args.load_result}")
        result = api.load_mining_result(args.load_result)
        if result.get('success'):
            print("✅ 结果加载成功")
            print(f"因子数量: {result.get('factors_info', {}).get('total_factors', 0)}")
            print(f"报告长度: {len(result.get('report', ''))}")
        else:
            print(f"❌ 结果加载失败: {result.get('error')}")
        return
    
    # 加载配置
    if args.config and Path(args.config).exists():
        with open(args.config, 'r', encoding='utf-8') as f:
            mining_config = json.load(f)
        print(f"📋 从配置文件加载配置: {args.config}")
    else:
        mining_config = create_mining_config()
        # 更新因子类型
        mining_config['factor_types'] = args.factor_types
        print("📋 使用默认挖掘配置")
    
    print(f"挖掘配置: {json.dumps(mining_config, indent=2, ensure_ascii=False)}")
    print()
    
    # 运行完整挖掘流程
    print("🚀 开始因子挖掘分析...")
    results = api.run_complete_mining(
        symbol=args.symbol,
        timeframe=args.timeframe,
        factor_types=args.factor_types,
        start_date=args.start_date,
        end_date=args.end_date,
        mining_config=mining_config
    )
    
    if results['success']:
        print("✅ 挖掘分析完成！")
        print()
        
        # 显示结果摘要
        print("=== 挖掘结果摘要 ===")
        print(f"数据点数量: {results['data_info']['shape'][0]:,}")
        print(f"因子数量: {results['factors_info']['total_factors']}")
        print(f"评估完成: {len(results['evaluation'])} 个因子")
        
        if results['optimization']['success']:
            print(f"优化方法: {results['optimization']['method']}")
            print(f"选择因子数: {len(results['optimization']['selected_factors'])}")
            print(f"优化得分: {results['optimization']['score']:.4f}")
        
        # 显示最佳因子
        if results['evaluation']:
            print("\n🏆 最佳因子 (按IC排序):")
            best_factors = sorted(
                [(name, data.get('ic_pearson', 0)) for name, data in results['evaluation'].items()],
                key=lambda x: x[1] if not np.isnan(x[1]) else 0,
                reverse=True
            )[:5]
            
            for i, (name, ic) in enumerate(best_factors, 1):
                print(f"{i}. {name}: IC = {ic:.4f}")
        
        # 保存结果
        if args.output:
            output_path = Path(args.output)
            # 这里可以添加自定义保存逻辑
            print(f"\n结果已保存到: {results['output_path']}")
        else:
            print(f"\n结果已保存到: {results['output_path']}")
        
        # 显示报告
        print("\n=== 详细挖掘报告 ===")
        print(results['report'])
        
    else:
        print("❌ 挖掘分析失败！")
        print(f"错误信息: {results['error']}")


def demo_mining():
    """演示因子挖掘功能"""
    print("🎯 因子挖掘演示模式")
    print("=" * 50)
    
    # 创建演示配置
    demo_config = {
        'factor_types': ['technical', 'statistical'],
        'factor_params': {
            'save_to_storage': True
        },
        'optimization': {
            'method': 'greedy',
            'max_factors': 10
        }
    }
    
    # 初始化API
    api = FactorMiningAPI()
    
    # 运行演示挖掘
    print("🚀 开始演示挖掘...")
    results = api.run_complete_mining(
        symbol='BTC_USDT',
        timeframe='1h',
        start_date='2024-01-01',
        end_date='2024-01-31',
        mining_config=demo_config
    )
    
    if results['success']:
        print("✅ 演示挖掘完成！")
        print(f"生成了 {results['factors_info']['total_factors']} 个因子")
        print(f"评估了 {len(results['evaluation'])} 个因子")
        
        if results['optimization']['success']:
            print(f"选择了 {len(results['optimization']['selected_factors'])} 个最优因子")
    else:
        print(f"❌ 演示挖掘失败: {results['error']}")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # 如果没有参数，运行演示模式
        demo_mining()
    else:
        main() 