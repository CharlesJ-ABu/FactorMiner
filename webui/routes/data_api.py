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
    
    # 基础配置
    options = {
        'enableRateLimit': True,
        'timeout': 30000,
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
        trade_type = request.args.get('trade_type', 'futures')
        
        # 构建数据目录路径
        configured_data_dir = current_app.config.get('DATA_DIR', 'data')
        print(f"配置的DATA_DIR: {configured_data_dir}")
        
        # 如果配置的路径已经指向具体目录，则使用其父目录
        if 'binance' in str(configured_data_dir) and 'futures' in str(configured_data_dir):
            base_data_dir = Path(configured_data_dir).parent.parent
        else:
            base_data_dir = Path(configured_data_dir)
            
        data_dir = base_data_dir / exchange / trade_type
        local_data = []
        
        print(f"基础数据目录: {base_data_dir}")
        print(f"扫描目录: {data_dir}")
        print(f"目录是否存在: {data_dir.exists()}")
        
        if data_dir.exists():
            print(f"目录内容: {list(data_dir.glob('*.feather'))}")
            for file_path in data_dir.glob('*.feather'):
                try:
                    # 解析文件名获取信息
                    filename = file_path.stem
                    
                    # 文件名格式: SYMBOL_USDT_USDT-TIMEFRAME-futures
                    if filename.endswith('-futures'):
                        # 移除 '-futures' 后缀
                        base_name = filename[:-8]
                        
                        # 查找最后一个连字符的位置（分隔交易对和时间框架）
                        last_hyphen = base_name.rfind('-')
                        if last_hyphen != -1:
                            symbol_part = base_name[:last_hyphen]
                            timeframe_part = base_name[last_hyphen + 1:]
                            
                            print(f"解析文件名: {filename}")
                            print(f"  base_name: {base_name}")
                            print(f"  symbol_part: {symbol_part}")
                            print(f"  timeframe_part: {timeframe_part}")
                            
                            # 解析交易对 (例如: BTC_USDT_USDT -> BTC_USDT)
                            symbol_parts = symbol_part.split('_')
                            if len(symbol_parts) >= 2:
                                symbol = f"{symbol_parts[0]}_{symbol_parts[1]}"
                                timeframe = timeframe_part
                                
                                print(f"  最终解析结果: symbol={symbol}, timeframe={timeframe}")
                                
                                print(f"解析文件名: {filename} -> symbol: {symbol}, timeframe: {timeframe}")
                                
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
                                    'exchange': 'binance',
                                    'symbol': symbol,
                                    'timeframe': timeframe,
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
                from factor_miner.core.data_downloader import DataDownloader
                
                # 更新状态
                with DOWNLOADS_LOCK:
                    if download_id in DOWNLOADS:
                        DOWNLOADS[download_id]['status'] = 'downloading'
                        DOWNLOADS[download_id]['message'] = '正在下载数据...'
                
                # 创建下载器实例
                downloader = DataDownloader()
                
                # 设置交易类型
                downloader.trade_type = trade_type
                
                # 开始下载 - 使用默认的 Binance 配置
                # 修复交易对格式：BTC_USDT -> BTC/USDT
                formatted_symbol = symbol.replace('_', '/')
                
                result = downloader.download_ohlcv(
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
                            DOWNLOADS[download_id]['message'] = f'下载完成！共 {result.get("data_points", 0)} 条数据'
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