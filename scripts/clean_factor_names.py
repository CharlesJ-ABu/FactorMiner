#!/usr/bin/env python3
"""
因子名称清理脚本
移除因子库中不合理的币种后缀
"""

import json
import os
import shutil
from pathlib import Path
from datetime import datetime

def clean_factor_name(factor_name):
    """清理因子名称，移除不合理的币种后缀"""
    # 需要移除的币种后缀列表
    crypto_symbols = ['BTC', 'ETH', 'BNB', 'SOL', 'ADA', 'DOGE', 'LINK', 'LPT', 'MOVR', 'PEOPLE', 'SUI', 'FIL']
    
    original_name = factor_name
    
    # 移除因子名称末尾的币种后缀
    for symbol in crypto_symbols:
        # 移除 "_SYMBOL" 格式的后缀
        if factor_name.endswith(f'_{symbol}'):
            factor_name = factor_name[:-len(symbol)-1]
        # 移除 "_SYMBOL_USDT" 格式的后缀  
        if factor_name.endswith(f'_{symbol}_USDT'):
            factor_name = factor_name[:-len(symbol)-6]
        # 移除 "_SYMBOL_timeframe" 格式的后缀
        for tf in ['1h', '4h', '1m', '5m', '15m', '1d']:
            if factor_name.endswith(f'_{symbol}_{tf}'):
                factor_name = factor_name[:-len(symbol)-len(tf)-2]
    
    if original_name != factor_name:
        print(f"清理因子名称: {original_name} -> {factor_name}")
    
    return factor_name

def clean_deep_alpha_factor_records():
    """清理Deep Alpha因子记录文件"""
    project_root = Path(__file__).parent.parent
    factor_records_file = project_root / "factorlib" / "deep_alpha" / "values" / "factor_records.json"
    
    if not factor_records_file.exists():
        print(f"因子记录文件不存在: {factor_records_file}")
        return False
    
    # 备份原文件
    backup_file = factor_records_file.with_suffix(f'.json.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    shutil.copy2(factor_records_file, backup_file)
    print(f"已备份原文件到: {backup_file}")
    
    try:
        # 读取原文件
        with open(factor_records_file, 'r', encoding='utf-8') as f:
            factor_records = json.load(f)
        
        # 清理记录
        cleaned_records = {}
        changes_made = 0
        
        for factor_id, record in factor_records.items():
            cleaned_record = record.copy()
            
            # 移除或标准化symbol字段 - 将特定币种标记为Universal
            if 'symbol' in cleaned_record:
                original_symbol = cleaned_record['symbol']
                if original_symbol in ['BTC', 'ETH', 'BNB', 'SOL', 'ADA', 'DOGE', 'LINK', 'LPT', 'MOVR', 'PEOPLE', 'SUI', 'FIL']:
                    cleaned_record['symbol'] = 'Universal'
                    print(f"更新因子 {factor_id} 的symbol: {original_symbol} -> Universal")
                    changes_made += 1
            
            # 如果因子公式是描述性文字而非代码，保持原样但添加通用标记
            if 'formula' in cleaned_record:
                formula = cleaned_record['formula']
                if any(keyword in formula for keyword in ['因子', '波动率', '动量', '趋势', '成交量']):
                    # 这是中文描述，添加通用标记
                    if not formula.endswith(' (通用因子)'):
                        cleaned_record['formula'] = formula + ' (通用因子)'
                        changes_made += 1
            
            cleaned_records[factor_id] = cleaned_record
        
        # 写入清理后的文件
        if changes_made > 0:
            with open(factor_records_file, 'w', encoding='utf-8') as f:
                json.dump(cleaned_records, f, indent=2, ensure_ascii=False)
            
            print(f"已清理Deep Alpha因子记录，共修改了 {changes_made} 个条目")
            return True
        else:
            print("Deep Alpha因子记录无需清理")
            return False
            
    except Exception as e:
        print(f"清理Deep Alpha因子记录失败: {e}")
        # 恢复备份文件
        if backup_file.exists():
            shutil.copy2(backup_file, factor_records_file)
            print("已恢复原文件")
        return False

def rename_alpha101_files():
    """重命名Alpha101文件，使其更通用"""
    project_root = Path(__file__).parent.parent
    alpha101_dir = project_root / "factorlib" / "alpha101" / "values"
    
    if not alpha101_dir.exists():
        print(f"Alpha101目录不存在: {alpha101_dir}")
        return False
    
    files_renamed = 0
    
    for file_path in alpha101_dir.glob("*.pkl"):
        filename = file_path.name
        
        # 检查是否是包含币种后缀的文件名
        if filename.startswith("alpha101_results_") and "_USDT_" in filename:
            # 从 alpha101_results_BTC_USDT_1h.pkl 改为 alpha101_universal_1h.pkl
            parts = filename.replace('.pkl', '').split('_')
            if len(parts) >= 4:
                timeframe = parts[3]
                new_filename = f"alpha101_universal_{timeframe}.pkl"
                new_file_path = alpha101_dir / new_filename
                
                # 如果新文件名已存在，添加计数器
                counter = 1
                while new_file_path.exists():
                    new_filename = f"alpha101_universal_{timeframe}_{counter}.pkl"
                    new_file_path = alpha101_dir / new_filename
                    counter += 1
                
                print(f"重命名文件: {filename} -> {new_filename}")
                file_path.rename(new_file_path)
                files_renamed += 1
    
    if files_renamed > 0:
        print(f"已重命名 {files_renamed} 个Alpha101文件")
        return True
    else:
        print("Alpha101文件无需重命名")
        return False

def main():
    """主函数"""
    print("开始清理因子库中的不合理后缀...")
    print("=" * 50)
    
    # 1. 清理Deep Alpha因子记录
    print("\n1. 清理Deep Alpha因子记录...")
    clean_deep_alpha_factor_records()
    
    # 2. 重命名Alpha101文件 (可选，可能会影响现有引用)
    print("\n2. 检查Alpha101文件...")
    # rename_alpha101_files()  # 暂时注释掉，避免破坏现有功能
    print("Alpha101文件暂时保持原名，API层面会进行名称清理")
    
    print("\n" + "=" * 50)
    print("因子库清理完成！")
    print("\n主要变更:")
    print("- Deep Alpha因子的symbol字段已标准化为'Universal'")
    print("- 因子描述添加了通用标记")
    print("- API层面会自动清理因子名称中的币种后缀")
    print("- 传统技术因子标记为通用因子")

if __name__ == "__main__":
    main()
