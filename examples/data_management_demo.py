#!/usr/bin/env python3
"""
FactorMiner 数据管理示例
展示如何使用数据下载、加载和管理功能
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from factor_miner.core.data_downloader import DataDownloader
from factor_miner.core.data_loader import DataLoader


def main():
    """主函数：演示数据管理功能"""
    print("=== FactorMiner 数据管理示例 ===\n")
    
    # 1. 初始化数据组件
    print("1. 初始化数据组件...")
    downloader = DataDownloader()
    data_loader = DataLoader()
    
    print("✅ 数据组件初始化完成")
    
    # 2. 检查现有数据
    print("\n2. 检查现有数据...")
    try:
        # 检查data目录结构
        data_dir = Path("data")
        if data_dir.exists():
            print("✅ data目录存在")
            
            # 检查binance子目录
            binance_dir = data_dir / "binance"
            if binance_dir.exists():
                print("✅ binance目录存在")
                
                # 检查现货和期货数据
                for trade_type in ["spot", "futures"]:
                    type_dir = binance_dir / trade_type
                    if type_dir.exists():
                        files = list(type_dir.glob("*.feather"))
                        print(f"   {trade_type}: {len(files)} 个数据文件")
                        
                        if files:
                            # 显示一些示例文件
                            sample_files = [f.name for f in files[:3]]
                            print(f"     示例文件: {sample_files}")
                    else:
                        print(f"   {trade_type}: 目录不存在")
            else:
                print("❌ binance目录不存在")
        else:
            print("❌ data目录不存在")
            
    except Exception as e:
        print(f"❌ 检查现有数据失败: {e}")
    
    # 3. 数据下载演示（仅演示，不实际下载）
    print("\n3. 数据下载功能演示...")
    try:
        # 获取可用的交易所信息
        print("   支持的交易所: Binance")
        print("   支持的数据类型: 现货(spot), 期货(futures)")
        print("   支持的时间框架: 1m, 5m, 15m, 1h, 4h, 1d")
        
        # 演示下载参数
        download_params = {
            'symbol': 'BTC_USDT',
            'timeframe': '1h',
            'trade_type': 'futures',
            'start_date': '2024-01-01',
            'end_date': '2024-01-31'
        }
        
        print(f"   示例下载参数: {download_params}")
        print("   💡 注意: 这是演示，不会实际下载数据")
        
    except Exception as e:
        print(f"❌ 数据下载演示失败: {e}")
    
    # 4. 数据加载演示
    print("\n4. 数据加载功能演示...")
    try:
        # 尝试加载一些现有数据
        sample_symbols = ['BTC_USDT', 'ETH_USDT']
        sample_timeframes = ['1h', '4h']
        
        for symbol in sample_symbols:
            for timeframe in sample_timeframes:
                print(f"   尝试加载 {symbol} {timeframe} 数据...")
                
                # 尝试现货数据
                spot_result = data_loader.load_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    trade_type='spot',
                    start_date='2024-01-01',
                    end_date='2024-01-31'
                )
                
                if spot_result['success']:
                    data = spot_result['data']
                    print(f"     ✅ 现货数据加载成功: {data.shape}")
                else:
                    print(f"     ❌ 现货数据加载失败: {spot_result['error']}")
                
                # 尝试期货数据
                futures_result = data_loader.load_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    trade_type='futures',
                    start_date='2024-01-01',
                    end_date='2024-01-31'
                )
                
                if futures_result['success']:
                    data = futures_result['data']
                    print(f"     ✅ 期货数据加载成功: {data.shape}")
                else:
                    print(f"     ❌ 期货数据加载失败: {futures_result['error']}")
                
                print()  # 空行分隔
                
    except Exception as e:
        print(f"❌ 数据加载演示失败: {e}")
    
    # 5. 数据质量检查
    print("\n5. 数据质量检查...")
    try:
        # 尝试加载一个数据文件进行质量检查
        data_result = data_loader.load_data(
            symbol='BTC_USDT',
            timeframe='1h',
            trade_type='futures',
            start_date='2024-01-01',
            end_date='2024-01-31'
        )
        
        if data_result['success']:
            data = data_result['data']
            print("✅ 数据质量检查:")
            print(f"   数据形状: {data.shape}")
            print(f"   时间范围: {data.index.min()} 到 {data.index.max()}")
            print(f"   数据列: {list(data.columns)}")
            print(f"   缺失值统计:")
            for col in data.columns:
                missing_count = data[col].isna().sum()
                missing_pct = (missing_count / len(data)) * 100
                print(f"     {col}: {missing_count} ({missing_pct:.2f}%)")
            
            # 检查数据类型
            print(f"   数据类型:")
            for col in data.columns:
                print(f"     {col}: {data[col].dtype}")
                
        else:
            print("❌ 无法加载数据进行质量检查")
            
    except Exception as e:
        print(f"❌ 数据质量检查失败: {e}")
    
    # 6. 数据管理建议
    print("\n6. 数据管理建议...")
    print("   📥 数据下载:")
    print("      - 使用 webui 的数据下载页面")
    print("      - 支持智能合并，避免重复数据")
    print("      - 自动处理时区和数据格式")
    
    print("   📊 数据查看:")
    print("      - 使用 webui 的数据查看页面")
    print("      - 支持多时间框架和多交易类型")
    print("      - 实时数据覆盖情况查询")
    
    print("   🔧 数据维护:")
    print("      - 定期清理临时文件")
    print("      - 监控磁盘空间使用")
    print("      - 备份重要数据文件")
    
    print("\n=== 数据管理示例完成 ===")


if __name__ == "__main__":
    main()
