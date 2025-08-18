#!/usr/bin/env python3
"""
分批下载功能演示
展示如何使用智能分批下载器下载不同时间框架的数据
"""

import sys
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from factor_miner.core.batch_downloader import batch_downloader


def progress_callback(progress, message):
    """进度回调函数"""
    print(f"[{progress:3d}%] {message}")


def demo_batch_download():
    """演示分批下载功能"""
    print("=== FactorMiner 智能分批下载演示 ===\n")
    
    # 测试参数
    symbol = 'BTC_USDT'
    start_date = '2024-01-01'
    end_date = '2024-01-31'
    
    print(f"下载参数:")
    print(f"  交易对: {symbol}")
    print(f"  时间范围: {start_date} 到 {end_date}")
    print(f"  总天数: {(datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')).days} 天")
    print()
    
    # 测试不同时间框架的分批下载
    timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
    
    for timeframe in timeframes:
        print(f"🔍 测试 {timeframe} 时间框架的分批下载...")
        
        # 获取分批配置
        batch_days, total_batches = batch_downloader.calculate_optimal_batch_size(
            timeframe, start_date, end_date
        )
        
        print(f"  推荐批次大小: {batch_days} 天")
        print(f"  预计批次数: {total_batches}")
        
        # 获取详细配置
        config = batch_downloader.get_batch_config(timeframe)
        print(f"  每批最大K线数: {config.max_candles_per_batch}")
        print(f"  批次间延迟: {config.delay_seconds} 秒")
        print(f"  重试次数: {config.retry_attempts}")
        print()
        
        # 询问是否实际下载
        response = input(f"是否下载 {timeframe} 数据？(y/n): ").lower().strip()
        
        if response == 'y':
            print(f"🚀 开始下载 {timeframe} 数据...")
            
            try:
                result = batch_downloader.download_ohlcv_batch(
                    symbol=symbol.replace('_', '/'),
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    trade_type='futures',
                    progress_callback=progress_callback
                )
                
                if result['success']:
                    print(f"✅ {timeframe} 数据下载成功!")
                    print(f"   总记录数: {result['total_records']}")
                    print(f"   实际批次数: {result['batch_info']['actual_batches']}")
                    print(f"   消息: {result['message']}")
                else:
                    print(f"❌ {timeframe} 数据下载失败: {result['error']}")
                    
            except Exception as e:
                print(f"❌ {timeframe} 下载异常: {e}")
            
            print()
        else:
            print(f"⏭️  跳过 {timeframe} 数据下载\n")


def demo_custom_batch_config():
    """演示自定义分批配置"""
    print("=== 自定义分批配置演示 ===\n")
    
    # 创建自定义配置
    custom_config = batch_downloader.batch_configs['1m'].copy()
    custom_config.batch_days = 2  # 每2天一批
    custom_config.delay_seconds = 2.0  # 延迟2秒
    custom_config.retry_attempts = 5  # 重试5次
    
    print("自定义配置:")
    print(f"  时间框架: {custom_config.timeframe}")
    print(f"  每批天数: {custom_config.batch_days}")
    print(f"  最大K线数: {custom_config.max_candles_per_batch}")
    print(f"  延迟秒数: {custom_config.delay_seconds}")
    print(f"  重试次数: {custom_config.retry_attempts}")
    print()
    
    # 计算特定时间范围的分批信息
    start_date = '2024-01-01'
    end_date = '2024-01-15'
    
    batch_days, total_batches = batch_downloader.calculate_optimal_batch_size(
        '1m', start_date, end_date
    )
    
    print(f"时间范围 {start_date} 到 {end_date} 的分批策略:")
    print(f"  推荐批次大小: {batch_days} 天")
    print(f"  总批次数: {total_batches}")
    print(f"  预计下载时间: {total_batches * custom_config.delay_seconds:.1f} 秒")


def main():
    """主函数"""
    print("🔍 选择演示模式:")
    print("1. 分批下载功能演示")
    print("2. 自定义配置演示")
    print("3. 退出")
    
    while True:
        choice = input("\n请输入选择 (1-3): ").strip()
        
        if choice == '1':
            demo_batch_download()
            break
        elif choice == '2':
            demo_custom_batch_config()
            break
        elif choice == '3':
            print("👋 再见!")
            break
        else:
            print("❌ 无效选择，请重新输入")


if __name__ == "__main__":
    main()
