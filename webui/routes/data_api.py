"""
数据管理API路由
"""

from flask import Blueprint, request, jsonify, current_app, session
import os
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
import json
import ccxt  # 添加 CCXT 库
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
import sys

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent.parent))

bp = Blueprint('data_api', __name__)

# 全局下载任务存储（线程安全）
DOWNLOADS = {}
DOWNLOADS_LOCK = threading.Lock()

@bp.route('/exchanges', methods=['GET'])
def get_exchanges():
    """获取支持的交易所列表"""
    exchanges = [
        {
            'id': 'binance',
            'name': 'Binance',
            'type': 'cryptocurrency',
            'description': '全球最大的加密货币交易所'
        },
        {
            'id': 'okx',
            'name': 'OKX',
            'type': 'cryptocurrency',
            'description': '领先的加密货币交易平台'
        },
        {
            'id': 'bybit',
            'name': 'Bybit',
            'type': 'cryptocurrency',
            'description': '专业的加密货币衍生品交易所'
        }
    ]
    return jsonify({'success': True, 'data': exchanges})

def get_exchange_instance(exchange_id, is_futures=False):
    """获取交易所实例"""
    exchange_class = getattr(ccxt, exchange_id)
    
    # 基础配置 - 检测环境变量中的代理配置
    http_proxy = os.getenv('HTTP_PROXY')
    https_proxy = os.getenv('HTTPS_PROXY')
    proxies = {}
    if http_proxy:
        proxies['http'] = http_proxy
    if https_proxy:
        proxies['https'] = https_proxy

    options = {
        'enableRateLimit': True,
        'timeout': 30000,
        'proxies': proxies if proxies else None
    }
    
    # 为不同交易所配置期货市场选项
    if is_futures:
        if exchange_id == 'binance':
            options.update({
                'defaultType': 'future',
                'urls': {
                    'api': {
                        'public': 'https://fapi.binance.com/fapi/v1',
                        'private': 'https://fapi.binance.com/fapi/v1',
                    }
                }
            })
        elif exchange_id == 'okx':
            options['defaultType'] = 'swap'
        elif exchange_id == 'bybit':
            options['defaultType'] = 'linear'
    else:
        if exchange_id == 'binance':
            options.update({
                'defaultType': 'spot',
                'urls': {
                    'api': {
                        'public': 'https://api.binance.com/api/v3',
                        'private': 'https://api.binance.com/api/v3',
                    }
                }
            })
        else:
            options['defaultType'] = 'spot'
    
    exchange = exchange_class(options)
    
    # 设置请求头
    exchange.headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'
    }
    
    return exchange

def format_symbol(market, market_type='spot', exchange_id='binance'):
    """格式化交易对信息"""
    base = market['base']
    quote = market['quote']
    
    # 获取合约到期日（如果有）
    contract_type = market.get('info', {}).get('contractType', '')
    delivery_date = market.get('info', {}).get('deliveryDate', '')
    
    # 根据不同交易所格式化交易对名称
    if exchange_id == 'binance':
        # 对于永续合约，使用基础名称
        if contract_type == 'PERPETUAL':
            symbol = f"{base}_{quote}"
        # 对于交割合约，添加到期日
        elif delivery_date:
            symbol = f"{base}_{quote}_{delivery_date}"
        else:
            symbol = f"{base}_{quote}"
    elif exchange_id == 'okx':
        symbol = f"{base}-{quote}"
    elif exchange_id == 'bybit':
        symbol = f"{base}{quote}"
    else:
        symbol = f"{base}_{quote}"
    
    result = {
        'symbol': symbol,
        'name': f"{base}/{quote}",
        'type': market_type,
        'base': base,
        'quote': quote,
        'active': market.get('active', True)
    }
    
    # 添加合约信息（如果有）
    if contract_type:
        result['contract_type'] = contract_type
    if delivery_date:
        result['delivery_date'] = delivery_date
    
    return result

@bp.route('/symbols/<exchange>', methods=['GET'])
def get_symbols(exchange):
    """获取指定交易所的交易对列表"""
    try:
        spot_markets = []
        perpetual_markets = []
        delivery_markets = []
        
        # 获取现货市场
        try:
            print(f"\n开始获取 {exchange} 现货市场数据...")
            spot_instance = get_exchange_instance(exchange, is_futures=False)
            spot_markets_data = spot_instance.load_markets()
            print(f"成功获取现货市场数据，共 {len(spot_markets_data)} 个交易对")
            
            # 用于去重的集合
            seen_symbols = set()
            
            for symbol, market in spot_markets_data.items():
                try:
                    # 调试每个市场的数据结构
                    print(f"\n处理现货交易对 {symbol}:")
                    print(f"  base: {market.get('base', 'N/A')}")
                    print(f"  quote: {market.get('quote', 'N/A')}")
                    print(f"  active: {market.get('active', 'N/A')}")
                    
                    if market.get('quote') == 'USDT' and market.get('active', True):
                        formatted = format_symbol(market, 'spot', exchange)
                        
                        # 检查是否已经添加过这个交易对
                        if formatted['symbol'] not in seen_symbols:
                            seen_symbols.add(formatted['symbol'])
                            spot_markets.append(formatted)
                            print(f"  ✅ 添加现货交易对: {formatted['symbol']}")
                        else:
                            print(f"  ⚠️ 跳过重复的现货交易对: {formatted['symbol']}")
                    else:
                        print(f"  ❌ 跳过现货交易对: quote={market.get('quote')}, active={market.get('active')}")
                except Exception as e:
                    print(f"  ❌ 处理现货交易对 {symbol} 时出错: {str(e)}")
                    continue
                    
        except Exception as e:
            print(f"获取现货市场失败: {str(e)}")
            print(f"错误类型: {type(e)}")
            import traceback
            print(f"错误堆栈: {traceback.format_exc()}")
        
        # 获取期货市场
        try:
            print(f"\n开始获取 {exchange} 期货市场数据...")
            futures_instance = get_exchange_instance(exchange, is_futures=True)
            futures_markets_data = futures_instance.load_markets()
            print(f"成功获取期货市场数据，共 {len(futures_markets_data)} 个交易对")
            
            # 只获取永续合约，跳过交割合约
            seen_perpetual_symbols = set()
            
            for symbol, market in futures_markets_data.items():
                try:
                    # 调试每个市场的数据结构
                    print(f"\n处理期货交易对 {symbol}:")
                    print(f"  base: {market.get('base', 'N/A')}")
                    print(f"  quote: {market.get('quote', 'N/A')}")
                    print(f"  active: {market.get('active', 'N/A')}")
                    print(f"  contract_type: {market.get('info', {}).get('contractType', 'N/A')}")
                    
                    if market.get('quote') == 'USDT' and market.get('active', True):
                        contract_type = market.get('info', {}).get('contractType', '')
                        
                        # 只处理永续合约，跳过交割合约
                        if contract_type == 'PERPETUAL':
                            formatted = format_symbol(market, 'futures', exchange)
                            if formatted['symbol'] not in seen_perpetual_symbols:
                                seen_perpetual_symbols.add(formatted['symbol'])
                                perpetual_markets.append(formatted)
                                print(f"  ✅ 添加永续合约: {formatted['symbol']}")
                            else:
                                print(f"  ⚠️ 跳过重复的永续合约: {formatted['symbol']}")
                        else:
                            print(f"  ⏭️ 跳过交割合约: {symbol} (contract_type: {contract_type})")
                    else:
                        print(f"  ❌ 跳过期货交易对: quote={market.get('quote')}, active={market.get('active')}")
                except Exception as e:
                    print(f"  ❌ 处理期货交易对 {symbol} 时出错: {str(e)}")
                    continue
                    
        except Exception as e:
            print(f"获取期货市场失败: {str(e)}")
            print(f"错误类型: {type(e)}")
            import traceback
            print(f"错误堆栈: {traceback.format_exc()}")
        
        # 按交易对名称排序
        spot_markets.sort(key=lambda x: x['symbol'])
        perpetual_markets.sort(key=lambda x: x['symbol'])
        
        print(f"\n最终结果:")
        print(f"✅ 获取到 {len(spot_markets)} 个现货交易对")
        print(f"✅ 获取到 {len(perpetual_markets)} 个永续合约")
        
        return jsonify({
            'success': True,
            'data': {
                'spot': spot_markets,
                'futures': perpetual_markets  # 期货直接返回永续合约
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'获取交易对失败: {str(e)}'
        }), 500

@bp.route('/timeframes', methods=['GET'])
def get_timeframes():
    """获取支持的时间框架"""
    timeframes = [
        {'value': '1m', 'name': '1分钟', 'description': '1分钟K线数据'},
        {'value': '3m', 'name': '3分钟', 'description': '3分钟K线数据'},
        {'value': '5m', 'name': '5分钟', 'description': '5分钟K线数据'},
        {'value': '15m', 'name': '15分钟', 'description': '15分钟K线数据'},
        {'value': '1h', 'name': '1小时', 'description': '1小时K线数据'},
        {'value': '2h', 'name': '2小时', 'description': '2小时K线数据'},
        {'value': '4h', 'name': '4小时', 'description': '4小时K线数据'},
        {'value': '6h', 'name': '6小时', 'description': '6小时K线数据'},
        {'value': '8h', 'name': '8小时', 'description': '8小时K线数据'},
        {'value': '12h', 'name': '12小时', 'description': '12小时K线数据'},
        {'value': '1d', 'name': '1天', 'description': '1天K线数据'}
    ]
    return jsonify({'success': True, 'data': timeframes})

@bp.route('/local-data', methods=['GET'])
def get_local_data():
    """获取本地存储的数据信息"""
    try:
        # 获取查询参数
        exchange = request.args.get('exchange', 'binance')
        trade_type = request.args.get('trade_type', '')  # 空字符串表示所有类型
        
        # 构建数据目录路径
        configured_data_dir = current_app.config.get('DATA_DIR', 'data')
        print(f"配置的DATA_DIR: {configured_data_dir}")
        
        # 如果配置的路径已经指向具体目录，则使用其父目录
        if 'binance' in str(configured_data_dir) and ('futures' in str(configured_data_dir) or 'spot' in str(configured_data_dir)):
            base_data_dir = Path(configured_data_dir).parent.parent
        else:
            base_data_dir = Path(configured_data_dir)
        
        local_data = []
        
        # 如果指定了特定类型，只扫描该类型目录
        if trade_type:
            search_dirs = [base_data_dir / exchange / trade_type]
        else:
            # 扫描所有类型目录
            search_dirs = [
                base_data_dir / exchange / 'futures',
                base_data_dir / exchange / 'spot',
                base_data_dir / exchange / 'perpetual',
                base_data_dir / exchange / 'delivery'
            ]
        
        print(f"基础数据目录: {base_data_dir}")
        print(f"扫描目录: {[str(d) for d in search_dirs]}")
        
        for data_dir in search_dirs:
            if not data_dir.exists():
                print(f"目录不存在: {data_dir}")
                continue
                
            print(f"扫描目录: {data_dir}")
            print(f"目录内容: {list(data_dir.glob('*.feather'))}")
            
            for file_path in data_dir.glob('*.feather'):
                try:
                    # 解析文件名获取信息
                    filename = file_path.stem
                    
                    # 检测数据类型和解析文件名
                    data_type = 'unknown'
                    base_name = filename
                    timeframe_part = 'unknown'
                    
                    # 处理不同的文件名格式
                    if filename.endswith('-futures'):
                        data_type = 'futures'
                        base_name = filename[:-8]
                    elif filename.endswith('-spot'):
                        data_type = 'spot'
                        base_name = filename[:-5]
                    elif filename.endswith('-perpetual'):
                        data_type = 'perpetual'
                        base_name = filename[:-10]
                    elif filename.endswith('-delivery'):
                        data_type = 'delivery'
                        base_name = filename[:-9]
                    
                    # 查找最后一个连字符的位置（分隔交易对和时间框架）
                    last_hyphen = base_name.rfind('-')
                    if last_hyphen != -1:
                        symbol_part = base_name[:last_hyphen]
                        timeframe_part = base_name[last_hyphen + 1:]
                        
                        print(f"解析文件名: {filename}")
                        print(f"  data_type: {data_type}")
                        print(f"  base_name: {base_name}")
                        print(f"  symbol_part: {symbol_part}")
                        print(f"  timeframe_part: {timeframe_part}")
                        
                        # 解析交易对 (例如: BTC_USDT_USDT -> BTC_USDT)
                        symbol_parts = symbol_part.split('_')
                        if len(symbol_parts) >= 2:
                            symbol = f"{symbol_parts[0]}_{symbol_parts[1]}"
                            
                            print(f"  最终解析结果: symbol={symbol}, timeframe={timeframe_part}, type={data_type}")
                            
                            # 读取数据获取基本信息
                            df = pd.read_feather(file_path)
                            print(f"文件 {filename} 的列名: {list(df.columns)}")
                            
                            # 推断时间范围：优先列，其次索引
                            def to_datetime_series(series):
                                try:
                                    if pd.api.types.is_datetime64_any_dtype(series):
                                        return series
                                    if pd.api.types.is_numeric_dtype(series):
                                        # 判断毫秒/秒级
                                        s = series.dropna()
                                        if len(s) == 0:
                                            return pd.to_datetime(series, errors='coerce', unit='s')
                                        sample = s.iloc[0]
                                        unit = 'ms' if sample > 10_000_000_000 else 's'
                                        return pd.to_datetime(series, errors='coerce', unit=unit)
                                    # 字符串
                                    return pd.to_datetime(series, errors='coerce')
                                except Exception:
                                    return pd.to_datetime(series, errors='coerce')

                            start_ts = None
                            end_ts = None
                            cols_lower = {c.lower(): c for c in df.columns}
                            # 常见列名
                            open_cols = [name for key, name in cols_lower.items() if key in ['open_time', 'opentime', 'start_time', 'time', 'timestamp', 'datetime', 'date']]
                            close_cols = [name for key, name in cols_lower.items() if key in ['close_time', 'closetime', 'end_time', 'time', 'timestamp', 'datetime', 'date']]
                            cand_open = open_cols[0] if open_cols else None
                            cand_close = close_cols[0] if close_cols else None
                            if cand_open is not None:
                                s = to_datetime_series(df[cand_open])
                                if s.notna().any():
                                    start_ts = s.min()
                            if cand_close is not None:
                                s = to_datetime_series(df[cand_close])
                                if s.notna().any():
                                    end_ts = s.max()
                            # 若仍为空，尝试索引
                            if (start_ts is None or pd.isna(start_ts)) and hasattr(df.index, 'min'):
                                idx = df.index
                                try:
                                    if not pd.api.types.is_datetime64_any_dtype(idx):
                                        idx = to_datetime_series(pd.Series(idx))
                                    start_ts = idx.min()
                                except Exception:
                                    start_ts = None
                            if (end_ts is None or pd.isna(end_ts)) and hasattr(df.index, 'max'):
                                idx = df.index
                                try:
                                    if not pd.api.types.is_datetime64_any_dtype(idx):
                                        idx = to_datetime_series(pd.Series(idx))
                                    end_ts = idx.max()
                                except Exception:
                                    end_ts = None

                            # 格式化为 ISO 字符串
                            if start_ts is not None and not pd.isna(start_ts):
                                try:
                                    start_str = pd.to_datetime(start_ts).strftime('%Y-%m-%d')
                                except Exception:
                                    start_str = str(start_ts)
                            else:
                                start_str = ""
                            if end_ts is not None and not pd.isna(end_ts):
                                try:
                                    end_str = pd.to_datetime(end_ts).strftime('%Y-%m-%d')
                                except Exception:
                                    end_str = str(end_ts)
                            else:
                                end_str = ""
                            
                            data_info = {
                                'exchange': exchange,
                                'symbol': symbol,
                                'timeframe': timeframe_part,
                                'data_type': data_type,
                                'file_path': str(file_path),
                                'file_size': f"{file_path.stat().st_size / 1024 / 1024:.2f} MB",
                                'data_points': len(df),
                                'date_range': {
                                    'start': start_str,
                                    'end': end_str
                                },
                                'last_modified': datetime.fromtimestamp(file_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                            }
                            local_data.append(data_info)
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")
                    continue
        
        print(f"返回数据条数: {len(local_data)}")
        print(f"返回数据结构示例: {local_data[0] if local_data else 'No data'}")
        return jsonify({'success': True, 'data': local_data})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@bp.route('/view-data', methods=['POST'])
def view_data():
    """查看数据文件内容"""
    print("🔍 view_data API 开始执行")
    try:
        data = request.get_json()
        print(f"🔍 接收到的请求数据: {data}")
        
        file_path = data.get('file_path')
        print(f"🔍 文件路径: {file_path}")
        
        if not file_path:
            print("❌ 文件路径为空")
            return jsonify({'success': False, 'error': '文件路径不能为空'})
        
        # 检查文件是否存在
        print(f"🔍 检查文件是否存在: {os.path.exists(file_path)}")
        if not os.path.exists(file_path):
            print(f"❌ 文件不存在: {file_path}")
            return jsonify({'success': False, 'error': '文件不存在'})
        
        # 读取Feather文件
        print("🔍 开始读取Feather文件...")
        df = pd.read_feather(file_path)
        print(f"🔍 文件读取成功，数据形状: {df.shape}")
        print(f"🔍 列名: {list(df.columns)}")
        print(f"🔍 数据类型: {df.dtypes.to_dict()}")
        
        # 解析文件名获取基本信息
        filename = Path(file_path).stem
        
        # 解析交易对和时间框架
        if filename.endswith('-futures'):
            base_name = filename[:-8]
            last_hyphen = base_name.rfind('-')
            if last_hyphen != -1:
                symbol_part = base_name[:last_hyphen]
                timeframe_part = base_name[last_hyphen + 1:]
                symbol_parts = symbol_part.split('_')
                if len(symbol_parts) >= 2:
                    symbol = f"{symbol_parts[0]}_{symbol_parts[1]}"
                    timeframe = timeframe_part
                else:
                    symbol = filename
                    timeframe = 'unknown'
            else:
                symbol = filename
                timeframe = 'unknown'
        else:
            symbol = filename
            timeframe = 'unknown'
        
        # 准备OHLCV数据
        print("🔍 开始准备OHLCV数据...")
        ohlcv_data = []
        
        # 确定时间列和OHLCV列
        time_col = None
        ohlcv_cols = {}
        
        # 查找时间列
        print("🔍 开始查找时间列...")
        for col in df.columns:
            col_lower = col.lower()
            print(f"🔍 检查列: {col} (小写: {col_lower})")
            if any(keyword in col_lower for keyword in ['time', 'date', 'timestamp', 'datetime']):
                time_col = col
                print(f"✅ 找到时间列: {col}")
                break
        
        print(f"🔍 最终确定的时间列: {time_col}")
        
        # 如果没有找到时间列，尝试使用索引
        if time_col is None and df.index.name:
            time_col = df.index.name
            df = df.reset_index()
        
        # 查找OHLCV列
        print("🔍 开始查找OHLCV列...")
        for col in df.columns:
            col_lower = col.lower()
            print(f"🔍 检查列: {col} (小写: {col_lower})")
            if 'open' in col_lower:
                ohlcv_cols['open'] = col
                print(f"✅ 找到开盘价列: {col}")
            elif 'high' in col_lower:
                ohlcv_cols['high'] = col
                print(f"✅ 找到最高价列: {col}")
            elif 'low' in col_lower:
                ohlcv_cols['low'] = col
                print(f"✅ 找到最低价列: {col}")
            elif 'close' in col_lower:
                ohlcv_cols['close'] = col
                print(f"✅ 找到收盘价列: {col}")
            elif 'volume' in col_lower:
                ohlcv_cols['volume'] = col
                print(f"✅ 找到成交量列: {col}")
        
        print(f"🔍 找到的OHLCV列映射: {ohlcv_cols}")
        
        # 如果没有找到标准列名，尝试其他常见列名
        if not ohlcv_cols.get('open'):
            for col in df.columns:
                if any(keyword in col.lower() for keyword in ['o', 'op', 'open']):
                    ohlcv_cols['open'] = col
                    break
        
        if not ohlcv_cols.get('high'):
            for col in df.columns:
                if any(keyword in col.lower() for keyword in ['h', 'hi', 'high']):
                    ohlcv_cols['high'] = col
                    break
        
        if not ohlcv_cols.get('low'):
            for col in df.columns:
                if any(keyword in col.lower() for keyword in ['l', 'lo', 'low']):
                    ohlcv_cols['low'] = col
                    break
        
        if not ohlcv_cols.get('close'):
            for col in df.columns:
                if any(keyword in col.lower() for keyword in ['c', 'cl', 'close']):
                    ohlcv_cols['close'] = col
                    break
        
        if not ohlcv_cols.get('volume'):
            for col in df.columns:
                if any(keyword in col.lower() for keyword in ['v', 'vol', 'volume']):
                    ohlcv_cols['volume'] = col
                    break
        
        # 构建OHLCV数据
        print("🔍 开始构建OHLCV数据...")
        print(f"🔍 数据行数: {len(df)}")
        print(f"🔍 时间列: {time_col}")
        print(f"🔍 OHLCV列映射: {ohlcv_cols}")
        
        for idx, row in df.iterrows():
            if idx < 5:  # 只打印前5行的调试信息
                print(f"🔍 处理第{idx}行数据...")
            
            # 处理时间
            timestamp = None
            if time_col and time_col in df.columns:
                try:
                    if pd.api.types.is_datetime64_any_dtype(df[time_col]):
                        timestamp = df[time_col].iloc[idx]
                        if idx < 5:
                            print(f"🔍 第{idx}行时间(原始): {df[time_col].iloc[idx]}")
                    elif pd.api.types.is_numeric_dtype(df[time_col]):
                        # 判断是秒还是毫秒
                        sample = df[time_col].iloc[0]
                        unit = 'ms' if sample > 10_000_000_000 else 's'
                        timestamp = pd.to_datetime(df[time_col].iloc[idx], unit=unit)
                        if idx < 5:
                            print(f"🔍 第{idx}行时间(转换): {timestamp}, 单位: {unit}")
                    else:
                        timestamp = pd.to_datetime(df[time_col].iloc[idx])
                        if idx < 5:
                            print(f"🔍 第{idx}行时间(字符串): {timestamp}")
                except Exception as e:
                    print(f"❌ 第{idx}行时间处理失败: {e}")
                    timestamp = pd.Timestamp.now()
            else:
                timestamp = pd.Timestamp.now()
                if idx < 5:
                    print(f"🔍 第{idx}行使用默认时间: {timestamp}")
            
            # 处理OHLCV数据
            ohlcv_item = {}
            
            # 开盘价
            if ohlcv_cols.get('open') and ohlcv_cols['open'] in df.columns:
                try:
                    ohlcv_item['open'] = float(row[ohlcv_cols['open']])
                    if idx < 5:
                        print(f"🔍 第{idx}行开盘价: {ohlcv_item['open']}")
                except:
                    ohlcv_item['open'] = None
            else:
                ohlcv_item['open'] = None
            
            # 最高价
            if ohlcv_cols.get('high') and ohlcv_cols['high'] in df.columns:
                try:
                    ohlcv_item['high'] = float(row[ohlcv_cols['high']])
                    if idx < 5:
                        print(f"🔍 第{idx}行最高价: {ohlcv_item['high']}")
                except:
                    ohlcv_item['high'] = None
            else:
                ohlcv_item['high'] = None
            
            # 最低价
            if ohlcv_cols.get('low') and ohlcv_cols['low'] in df.columns:
                try:
                    ohlcv_item['low'] = float(row[ohlcv_cols['low']])
                    if idx < 5:
                        print(f"🔍 第{idx}行最低价: {ohlcv_item['low']}")
                except:
                    ohlcv_item['low'] = None
            else:
                ohlcv_item['low'] = None
            
            # 收盘价
            if ohlcv_cols.get('close') and ohlcv_cols['close'] in df.columns:
                try:
                    ohlcv_item['close'] = float(row[ohlcv_cols['close']])
                    if idx < 5:
                        print(f"🔍 第{idx}行收盘价: {ohlcv_item['close']}")
                except:
                    ohlcv_item['close'] = None
            else:
                ohlcv_item['close'] = None
            
            # 成交量
            if ohlcv_cols.get('volume') and ohlcv_cols['volume'] in df.columns:
                try:
                    ohlcv_item['volume'] = float(row[ohlcv_cols['volume']])
                    if idx < 5:
                        print(f"🔍 第{idx}行成交量: {ohlcv_item['volume']}")
                except:
                    ohlcv_item['volume'] = None
            else:
                ohlcv_item['volume'] = None
            
            # 添加时间戳 - 确保使用UTC时区
            if timestamp:
                # 如果时间有时区信息，转换为UTC
                if timestamp.tz is not None:
                    timestamp_utc = timestamp.tz_convert('UTC')
                else:
                    # 如果没有时区信息，假设是UTC
                    timestamp_utc = timestamp.tz_localize('UTC')
                
                ohlcv_item['timestamp'] = timestamp_utc.isoformat()
                if idx < 5:
                    print(f"🔍 第{idx}行时间(UTC): {ohlcv_item['timestamp']}")
            else:
                ohlcv_item['timestamp'] = None
            
            if idx < 5:
                print(f"🔍 第{idx}行完整OHLCV数据: {ohlcv_item}")
            
            ohlcv_data.append(ohlcv_item)
        
        print(f"🔍 构建完成，共{len(ohlcv_data)}条OHLCV数据")
        if len(ohlcv_data) > 0:
            print(f"🔍 第一条数据示例: {ohlcv_data[0]}")
            print(f"🔍 最后一条数据示例: {ohlcv_data[-1]}")
        
        # 按时间排序
        ohlcv_data.sort(key=lambda x: x['timestamp'] if x['timestamp'] else '')
        
        # 准备返回数据
        print("🔍 准备返回数据...")
        result_data = {
            'symbol': symbol,
            'timeframe': timeframe,
            'file_path': file_path,
            'file_size': f"{Path(file_path).stat().st_size / 1024 / 1024:.2f} MB",
            'last_modified': datetime.fromtimestamp(Path(file_path).stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
            'ohlcv_data': ohlcv_data,
            'columns_found': list(df.columns),
            'ohlcv_columns_mapped': ohlcv_cols
        }
        
        print(f"🔍 返回数据摘要:")
        print(f"  - 交易对: {result_data['symbol']}")
        print(f"  - 时间框架: {result_data['timeframe']}")
        print(f"  - OHLCV数据条数: {len(result_data['ohlcv_data'])}")
        print(f"  - 找到的列: {result_data['columns_found']}")
        print(f"  - OHLCV列映射: {result_data['ohlcv_columns_mapped']}")
        
        print("✅ view_data API 执行成功，准备返回数据")
        return jsonify({'success': True, 'data': result_data})
        
    except Exception as e:
        import traceback
        print(f"查看数据失败: {e}")
        print(f"错误详情: {traceback.format_exc()}")
        return jsonify({'success': False, 'error': f'查看数据失败: {str(e)}'})

@bp.route('/download', methods=['POST'])
def start_download():
    """开始下载数据"""
    try:
        data = request.get_json()
        exchange = data.get('exchange')
        symbol = data.get('symbol')
        timeframe = data.get('timeframe')
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        trade_type = data.get('trade_type', 'spot')  # 默认为现货
        
        # 生成下载ID
        download_id = f"{exchange}_{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # 初始化下载状态到全局存储
        with DOWNLOADS_LOCK:
            DOWNLOADS[download_id] = {
                'id': download_id,
            'exchange': exchange,
            'symbol': symbol,
            'timeframe': timeframe,
            'start_date': start_date,
            'end_date': end_date,
                'trade_type': trade_type,
                'status': 'starting',
                'progress': 0,
                'message': '正在初始化下载...',
                'start_time': datetime.now().isoformat()
            }
        
        # 启动后台下载任务
        def download_task():
            try:
                from factor_miner.core.batch_downloader import batch_downloader
                
                # 更新状态
                with DOWNLOADS_LOCK:
                    if download_id in DOWNLOADS:
                        DOWNLOADS[download_id]['status'] = 'downloading'
                        DOWNLOADS[download_id]['message'] = '正在初始化分批下载...'
                
                # 设置交易类型
                batch_downloader.trade_type = trade_type
                
                # 开始分批下载 - 使用智能分批下载器
                # 修复交易对格式：BTC_USDT -> BTC/USDT
                formatted_symbol = symbol.replace('_', '/')
                
                # 计算分批信息
                from datetime import datetime
                start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                total_days = (end_dt - start_dt).days
                
                with DOWNLOADS_LOCK:
                    if download_id in DOWNLOADS:
                        DOWNLOADS[download_id]['message'] = f'开始分批下载，总天数: {total_days} 天'
                
                result = batch_downloader.download_ohlcv_batch(
                    config_id=None,  # 不使用配置文件，直接创建实例
                    symbol=formatted_symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    trade_type=trade_type,
                    progress_callback=lambda progress, message: update_download_progress(download_id, progress, message)
                )
                
                if result.get('success'):
                    with DOWNLOADS_LOCK:
                        if download_id in DOWNLOADS:
                            DOWNLOADS[download_id]['status'] = 'completed'
                            DOWNLOADS[download_id]['progress'] = 100
                            DOWNLOADS[download_id]['message'] = f'下载完成！共 {result.get("total_records", 0)} 条数据'
                            DOWNLOADS[download_id]['file_path'] = result.get('file_path', '')
                else:
                    with DOWNLOADS_LOCK:
                        if download_id in DOWNLOADS:
                            DOWNLOADS[download_id]['status'] = 'failed'
                            DOWNLOADS[download_id]['message'] = f'下载失败: {result.get("error", "未知错误")}'
                    
            except Exception as e:
                with DOWNLOADS_LOCK:
                    if download_id in DOWNLOADS:
                        DOWNLOADS[download_id]['status'] = 'failed'
                        DOWNLOADS[download_id]['message'] = f'下载异常: {str(e)}'
                print(f"下载任务异常: {e}")
        
        # 启动后台线程
        thread = threading.Thread(target=download_task)
        thread.daemon = True
        thread.start()
        
        with DOWNLOADS_LOCK:
            return jsonify({'success': True, 'data': DOWNLOADS[download_id]})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@bp.route('/downloads', methods=['GET'])
def get_all_downloads():
    """获取所有下载任务的状态"""
    try:
        with DOWNLOADS_LOCK:
            # 返回所有下载任务的状态
            downloads = list(DOWNLOADS.values())
            return jsonify({'success': True, 'data': downloads})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@bp.route('/batch-download', methods=['POST'])
def start_batch_download():
    """开始批量下载数据"""
    try:
        data = request.get_json()
        exchange = data.get('exchange')
        symbols = data.get('symbols', [])  # 交易对列表
        timeframes = data.get('timeframes', [])  # 时间框架列表
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        trade_type = data.get('trade_type', 'spot')  # 默认为现货
        
        if not symbols or not timeframes:
            return jsonify({'success': False, 'error': '请选择交易对和时间框架'})
        
        # 计算总任务数
        total_tasks = len(symbols) * len(timeframes)
        print(f"开始批量下载: {len(symbols)} 个交易对 × {len(timeframes)} 个时间框架 = {total_tasks} 个任务")
        
        # 创建批量下载任务ID
        batch_id = f"batch_{exchange}_{trade_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 初始化批量下载状态
        with DOWNLOADS_LOCK:
            DOWNLOADS[batch_id] = {
                'id': batch_id,
                'type': 'batch',
                'exchange': exchange,
                'symbols': symbols,
                'timeframes': timeframes,
                'start_date': start_date,
                'end_date': end_date,
                'trade_type': trade_type,
                'status': 'starting',
            'progress': 0,
                'message': f'正在初始化批量下载，共 {total_tasks} 个任务...',
                'start_time': datetime.now().isoformat(),
                'total_tasks': total_tasks,
                'completed_tasks': 0,
                'failed_tasks': 0,
                'task_results': []
            }
        
        # 启动后台批量下载任务
        def batch_download_task():
            try:
                from factor_miner.core.batch_downloader import batch_downloader
                
                # 更新状态为下载中
                with DOWNLOADS_LOCK:
                    if batch_id in DOWNLOADS:
                        DOWNLOADS[batch_id]['status'] = 'downloading'
                        DOWNLOADS[batch_id]['message'] = f'开始批量下载，共 {total_tasks} 个任务'
                        print(f"批量下载状态更新: {batch_id} -> downloading")
                
                # 设置交易类型
                batch_downloader.trade_type = trade_type
                
                completed_count = 0
                failed_count = 0
                
                # 遍历所有交易对和时间框架组合
                for symbol in symbols:
                    for timeframe in timeframes:
                        task_id = f"{symbol}_{timeframe}"
                        
                        try:
                            # 更新当前任务状态
                            with DOWNLOADS_LOCK:
                                if batch_id in DOWNLOADS:
                                    current_task = completed_count + failed_count + 1
                                    progress = int((current_task - 1) / total_tasks * 100)
                                    DOWNLOADS[batch_id]['progress'] = progress
                                    DOWNLOADS[batch_id]['message'] = f'正在下载 {symbol} {timeframe} ({current_task}/{total_tasks})'
                                    print(f"批量下载进度更新: {batch_id} -> {progress}% - {symbol} {timeframe}")
                            
                            # 修复交易对格式：BTC_USDT -> BTC/USDT
                            formatted_symbol = symbol.replace('_', '/')
                            
                            # 使用现有的下载逻辑
                            result = batch_downloader.download_ohlcv_batch(
                                config_id=None,
                                symbol=formatted_symbol,
                                timeframe=timeframe,
                                start_date=start_date,
                                end_date=end_date,
                                trade_type=trade_type,
                                progress_callback=None  # 批量下载时不使用进度回调
                            )
                            
                            # 记录任务结果
                            task_result = {
                                'task_id': task_id,
                                'symbol': symbol,
                                'timeframe': timeframe,
                                'success': result.get('success', False),
                                'message': result.get('message', '未知状态'),
                                'records': result.get('total_records', 0),
                                'file_path': result.get('file_path', '')
                            }
                            
                            if result.get('success'):
                                completed_count += 1
                                task_result['status'] = 'completed'
                                print(f"✅ 任务完成: {symbol} {timeframe}")
                            else:
                                failed_count += 1
                                task_result['status'] = 'failed'
                                print(f"❌ 任务失败: {symbol} {timeframe} - {result.get('error', '未知错误')}")
                            
                            # 更新批量下载状态
                            with DOWNLOADS_LOCK:
                                if batch_id in DOWNLOADS:
                                    DOWNLOADS[batch_id]['task_results'].append(task_result)
                                    DOWNLOADS[batch_id]['completed_tasks'] = completed_count
                                    DOWNLOADS[batch_id]['failed_tasks'] = failed_count
                                    progress = int((completed_count + failed_count) / total_tasks * 100)
                                    DOWNLOADS[batch_id]['progress'] = progress
                                    print(f"批量下载任务完成: {batch_id} -> 进度 {progress}% - 完成 {completed_count}, 失败 {failed_count}")
                            
                        except Exception as e:
                            failed_count += 1
                            print(f"❌ 任务异常: {symbol} {timeframe} - {str(e)}")
                            
                            # 记录失败的任务
                            task_result = {
                                'task_id': task_id,
                                'symbol': symbol,
                                'timeframe': timeframe,
                                'success': False,
                                'message': f'任务异常: {str(e)}',
                                'records': 0,
                                'file_path': '',
                                'status': 'failed'
                            }
                            
                            with DOWNLOADS_LOCK:
                                if batch_id in DOWNLOADS:
                                    DOWNLOADS[batch_id]['task_results'].append(task_result)
                                    DOWNLOADS[batch_id]['failed_tasks'] = failed_count
                                    progress = int((completed_count + failed_count) / total_tasks * 100)
                                    DOWNLOADS[batch_id]['progress'] = progress
                                    print(f"批量下载任务异常: {batch_id} -> 进度 {progress}% - 完成 {completed_count}, 失败 {failed_count}")
                
                # 批量下载完成
                with DOWNLOADS_LOCK:
                    if batch_id in DOWNLOADS:
                        DOWNLOADS[batch_id]['status'] = 'completed'
                        DOWNLOADS[batch_id]['progress'] = 100
                        DOWNLOADS[batch_id]['message'] = f'批量下载完成！成功: {completed_count}, 失败: {failed_count}'
                        print(f"批量下载最终完成: {batch_id} -> completed - 成功 {completed_count}, 失败 {failed_count}")
                
                print(f"批量下载完成: 成功 {completed_count}, 失败 {failed_count}")
                
            except Exception as e:
                with DOWNLOADS_LOCK:
                    if batch_id in DOWNLOADS:
                        DOWNLOADS[batch_id]['status'] = 'failed'
                        DOWNLOADS[batch_id]['message'] = f'批量下载异常: {str(e)}'
                print(f"批量下载任务异常: {e}")
        
        # 启动后台线程
        thread = threading.Thread(target=batch_download_task)
        thread.daemon = True
        thread.start()
        
        with DOWNLOADS_LOCK:
            return jsonify({'success': True, 'data': DOWNLOADS[batch_id]})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def update_download_progress(download_id, progress, message):
    """更新下载进度（线程安全）"""
    with DOWNLOADS_LOCK:
        if download_id in DOWNLOADS:
            DOWNLOADS[download_id]['progress'] = progress
            DOWNLOADS[download_id]['message'] = message

@bp.route('/download-status/<download_id>', methods=['GET'])
def get_download_status(download_id):
    """获取下载状态"""
    try:
        # 从全局存储获取下载状态
        with DOWNLOADS_LOCK:
            if download_id not in DOWNLOADS:
                return jsonify({
                    'success': False,
                    'error': '下载任务不存在'
                }), 404
            download_info = dict(DOWNLOADS[download_id])
        
        # 计算下载速度（如果正在下载）
        if download_info['status'] == 'downloading' and 'start_time' in download_info:
            start_time = datetime.fromisoformat(download_info['start_time'])
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed > 0:
                speed = f"{download_info['progress'] / elapsed * 100:.1f} %/s"
            else:
                speed = "计算中..."
        else:
            speed = "N/A"
        
        status_info = {
            'id': download_id,
            'progress': download_info['progress'],
            'status': download_info['status'],
            'message': download_info['message'],
            'exchange': download_info.get('exchange', ''),
            'symbol': download_info.get('symbol', ''),
            'timeframe': download_info.get('timeframe', ''),
            'start_date': download_info.get('start_date', ''),
            'end_date': download_info.get('end_date', ''),
            'trade_type': download_info.get('trade_type', ''),
            'file_path': download_info.get('file_path', ''),
            'download_speed': speed
        }
        
        return jsonify({'success': True, 'data': status_info})
        
    except Exception as e:
        return jsonify({
            'success': False, 
            'error': f'获取下载状态失败: {str(e)}'
        }), 500

@bp.route('/download-suggestions', methods=['POST'])
def get_download_suggestions():
    """获取下载建议"""
    try:
        data = request.get_json()
        exchange = data.get('exchange')
        trade_type = data.get('trade_type')
        symbol = data.get('symbol')
        timeframe = data.get('timeframe')
        
        # 获取本地数据
        local_data = []
        
        # 构建数据目录路径
        search_dirs = []
        
        if trade_type == 'futures':
            # 期货类型：检查 DATA_DIR 是否已经是 futures 目录
            base_dir = Path(current_app.config.get('DATA_DIR', 'data'))
            print(f"DATA_DIR 配置: {base_dir}")
            
            # 如果 DATA_DIR 已经是 futures 目录，直接使用
            if base_dir.name == 'futures':
                search_dirs.append(base_dir)
                print(f"期货类型：DATA_DIR 已是 futures 目录，使用 {base_dir}")
            else:
                # 否则构建标准路径
                futures_dir = base_dir / exchange / 'futures'
                search_dirs.append(futures_dir)
                print(f"期货类型：构建标准路径 {futures_dir}")
        elif trade_type in ['perpetual', 'delivery']:
            # 永续或交割合约：先在新目录中查找，再在旧的 futures 目录中查找
            data_dir = Path(current_app.config.get('DATA_DIR', 'data')) / exchange / trade_type
            search_dirs.append(data_dir)
            
            futures_dir = Path(current_app.config.get('DATA_DIR', 'data')) / exchange / 'futures'
            if futures_dir.exists():
                search_dirs.append(futures_dir)
                print(f"向后兼容：同时在 {futures_dir} 目录中查找数据")
        else:
            # 现货等其他类型
            data_dir = Path(current_app.config.get('DATA_DIR', 'data')) / exchange / trade_type
            search_dirs.append(data_dir)
        
        # 在所有相关目录中查找数据
        for search_dir in search_dirs:
            if search_dir.exists():
                print(f"在目录 {search_dir} 中查找数据文件...")
                # 调试：打印匹配模式
                glob_pattern = f"{symbol}*{timeframe}*.feather"
                print(f"使用匹配模式: {glob_pattern}")
                
                # 使用更精确的匹配模式，避免部分匹配
                # 例如：2h 不应该匹配到 12h
                for file_path in search_dir.glob(f"{symbol}*{timeframe}*.feather"):
                    # 验证文件名是否真正匹配指定的 timeframe
                    filename = file_path.name
                    if timeframe not in filename or f"-{timeframe}-" not in filename:
                        # 跳过不匹配的文件
                        print(f"跳过不匹配的文件: {filename} (期望: {timeframe})")
                        continue
                    try:
                        print(f"找到数据文件: {file_path}")
                        df = pd.read_feather(file_path)
                        # 获取时间范围 - 支持多种时间列名
                        time_col = None
                        for col in df.columns:
                            if col.lower() in ['date', 'time', 'datetime', 'timestamp']:
                                time_col = col
                                break
                        
                        if time_col:
                            start_date = pd.to_datetime(df[time_col].min()).strftime('%Y-%m-%d')
                            end_date = pd.to_datetime(df[time_col].max()).strftime('%Y-%m-%d')
                            local_data.append({
                                'data_type': trade_type,
                                'start_date': start_date,
                                'end_date': end_date,
                                'data_points': len(df),
                                'file_size': f"{file_path.stat().st_size / 1024 / 1024:.2f} MB",
                                'file_path': str(file_path)
                            })
                            print(f"成功读取数据文件: {file_path}, 数据点数: {len(df)}")
                    except Exception as e:
                        print(f"读取文件 {file_path} 失败: {str(e)}")
            else:
                print(f"目录不存在: {search_dir}")
        
        print(f"总共找到 {len(local_data)} 个数据文件")
        
        # 生成下载建议
        recommended_downloads = []
        
        # 如果没有本地数据，建议下载最近一个月的数据
        if not local_data:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            recommended_downloads.append({
                'data_type': trade_type,
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d'),
                'reason': '建议从最近一个月开始下载数据'
            })
        else:
            # 检查数据是否需要更新到最新
            latest_data = max(local_data, key=lambda x: x['end_date'])
            latest_date = datetime.strptime(latest_data['end_date'], '%Y-%m-%d')
            if (datetime.now() - latest_date).days > 1:
                recommended_downloads.append({
                    'data_type': trade_type,
                    'start_date': latest_data['end_date'],
                    'end_date': datetime.now().strftime('%Y-%m-%d'),
                    'reason': '更新数据至最新'
                })
            
            # 检查数据是否有空缺
            sorted_data = sorted(local_data, key=lambda x: x['start_date'])
            for i in range(len(sorted_data) - 1):
                current_end = datetime.strptime(sorted_data[i]['end_date'], '%Y-%m-%d')
                next_start = datetime.strptime(sorted_data[i + 1]['start_date'], '%Y-%m-%d')
                if (next_start - current_end).days > 1:
                    recommended_downloads.append({
                        'data_type': trade_type,
                        'start_date': sorted_data[i]['end_date'],
                        'end_date': sorted_data[i + 1]['start_date'],
                        'reason': '补充数据空缺'
                    })
        
        return jsonify({
            'success': True,
            'data': {
                'exchange': exchange,
                'symbol': symbol,
                'timeframe': timeframe,
                'trade_type': trade_type,
                'existing_data': local_data,
                'recommended_downloads': recommended_downloads
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'获取下载建议失败: {str(e)}'
        }), 500

@bp.route('/data-health', methods=['POST'])
def check_data_health():
    """检查数据健康度"""
    try:
        data = request.get_json()
        file_path = data.get('file_path')
        symbol = data.get('symbol')
        timeframe = data.get('timeframe')
        
        if not file_path or not os.path.exists(file_path):
            return jsonify({'success': False, 'error': '文件不存在'})
        
        # 导入健康度检查器
        from factor_miner.core.data_health_checker import health_checker
        
        # 读取数据文件
        df = pd.read_feather(file_path)
        
        # 检查数据健康度
        health_report = health_checker.check_data_health(df, timeframe, symbol)
        
        return jsonify({
            'success': True,
            'data': health_report
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@bp.route('/auto-fill-gaps', methods=['POST'])
def auto_fill_gaps():
    """自动补全数据断层"""
    try:
        data = request.get_json()
        symbol = data.get('symbol')
        timeframe = data.get('timeframe')
        trade_type = data.get('trade_type', 'futures')
        data_dir = data.get('data_dir')
        
        if not symbol or not timeframe:
            return jsonify({'success': False, 'error': '缺少必要参数'})
        
        # 导入断层补全器
        from factor_miner.core.data_gap_filler import gap_filler
        
        # 执行自动补全
        result = gap_filler.auto_fill_gaps(symbol, timeframe, trade_type, data_dir)
        
        return jsonify({
            'success': True,
            'data': result
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@bp.route('/scan-gaps', methods=['POST'])
def scan_data_gaps():
    """扫描数据断层"""
    try:
        data = request.get_json()
        data_dir = data.get('data_dir', 'data/binance/futures')
        symbol = data.get('symbol')
        timeframe = data.get('timeframe')
        
        # 导入断层补全器
        from factor_miner.core.data_gap_filler import gap_filler
        
        # 扫描断层
        gaps = gap_filler.scan_for_gaps(data_dir, symbol, timeframe)
        
        return jsonify({
            'success': True,
            'data': {
                'gaps': gaps,
                'total_gaps': len(gaps),
                'data_dir': data_dir
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@bp.route('/delete-data', methods=['POST'])
def delete_data():
    """删除本地数据"""
    try:
        data = request.get_json()
        file_path = data.get('file_path')
        
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
            return jsonify({'success': True, 'message': '数据删除成功'})
        else:
            return jsonify({'success': False, 'error': '文件不存在'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}) 