"""
因子库相关路由（V3架构）
使用透明因子存储与统一引擎
"""

import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from flask import Blueprint, request, jsonify
import math
from factor_miner.core.factor_evaluator import FactorEvaluator
from factor_miner.core.factor_engine import get_global_engine
from factor_miner.core.evaluation_io import (
    save_evaluation_results as core_save_evaluation_results,
    load_evaluations as core_load_evaluations,
)
# from factor_miner.core.factor_storage import get_global_storage

bp = Blueprint('factors', __name__)

# 因子库路径（V3 扁平化）
FACTOR_LIBRARY_DIR = Path(__file__).parent.parent.parent / "factorlib"

# Alpha101因子公式映射
ALPHA101_FORMULAS = {
    'alpha001': '(rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5)',
    'alpha002': '(-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))',
    'alpha003': '(-1 * correlation(rank(open), rank(volume), 10))',
    'alpha004': '(-1 * Ts_Rank(rank(low), 9))',
    'alpha005': '(rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap)))))',
    'alpha006': '(-1 * correlation(open, volume, 10))',
    'alpha007': '((adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7))) : (-1))',
    'alpha008': '(-1 * rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)), 10))))',
    'alpha009': '((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) : ((ts_max(delta(close, 1), 5) < 0) ? delta(close, 1) : (-1 * delta(close, 1))))',
    'alpha010': 'rank(((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1) : ((ts_max(delta(close, 1), 4) < 0) ? delta(close, 1) : (-1 * delta(close, 1)))))',
    # 可以继续添加更多Alpha101因子公式...
}

# 页面路由 - 已迁移到main.py

# API路由
@bp.route('/list')
def list_factors():
    """获取因子列表（V4：从新的文件夹结构读取）"""
    try:
        # 检查两个文件夹：technicals 和 minactors
        technicals_dir = FACTOR_LIBRARY_DIR / "technicals" / "definitions"
        minactors_dir = FACTOR_LIBRARY_DIR / "minactors" / "definitions"
        
        factors = []
        
        # 读取 technicals 文件夹
        if technicals_dir.exists():
            for file in technicals_dir.glob("*.json"):
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    data['source'] = 'technicals'  # 标记来源
                    factors.append(data)
                except Exception as e:
                    print(f"❌ 读取因子文件失败 {file}: {e}")
                    continue
        
        # 读取 minactors 文件夹
        if minactors_dir.exists():
            for file in minactors_dir.glob("*.json"):
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    data['source'] = 'minactors'  # 标记来源
                    factors.append(data)
                except Exception as e:
                    print(f"❌ 读取因子文件失败 {file}: {e}")
                    continue
        
        # 处理因子数据
        processed_factors = []
        for data in factors:
            try:
                comp = data.get('computation_data', {})
                formula_preview = None
                if data.get('computation_type') == 'formula':
                    formula_preview = (comp.get('formula') or '').split('\n')[0][:120]
                
                # 聚合评估均值（核心IO）
                evaluated = False
                eval_count = 0
                avg_metrics = {}
                last_evaluated_at = None
                try:
                    eval_payload = core_load_evaluations(data.get('factor_id'))
                    evaluations = (eval_payload or {}).get('evaluations') or []
                    eval_count = len(evaluations)
                    if eval_count > 0:
                        evaluated = True
                        keys = ['ic_pearson', 'ic_spearman', 'win_rate', 'sharpe_ratio', 'long_short_return']
                        sums = {k: 0.0 for k in keys}
                        counts = {k: 0 for k in keys}
                        for ev in evaluations:
                            res = (ev or {}).get('results') or {}
                            for k in keys:
                                v = res.get(k)
                                if isinstance(v, (int, float)):
                                    sums[k] += float(v)
                                    counts[k] += 1
                            last_evaluated_at = (ev or {}).get('evaluated_at') or last_evaluated_at
                        for k in keys:
                            avg_metrics[k] = (sums[k] / counts[k]) if counts[k] > 0 else None
                except Exception:
                    pass
                
                # 数值安全处理，避免NaN进入JSON
                def _safe_num(x):
                    return float(x) if isinstance(x, (int, float)) and math.isfinite(x) else None
                avg_metrics_clean = {}
                for k, v in (avg_metrics or {}).items():
                    avg_metrics_clean[k] = _safe_num(v)
                
                processed_factors.append({
                    'id': data.get('factor_id'),
                    'name': data.get('name'),
                    'description': data.get('description'),
                    'type': data.get('category'),
                    'source': data.get('source'),  # 添加来源信息
                    'created_at': data.get('metadata', {}).get('created_at'),
                    'computation_type': data.get('computation_type'),
                    'formula': formula_preview,
                    'evaluated': evaluated,
                    'evaluations_count': eval_count,
                    'avg_metrics': avg_metrics_clean,
                    'last_evaluated_at': last_evaluated_at,
                    'ic': _safe_num((avg_metrics or {}).get('ic_pearson')),
                    'ir': _safe_num((avg_metrics or {}).get('sharpe_ratio')),
                })
            except Exception as e:
                print(f"❌ 处理因子数据失败 {data.get('factor_id', 'unknown')}: {e}")
                continue
        
        return jsonify({'success': True, 'factors': processed_factors, 'total': len(processed_factors)})
    except Exception as e:
        return jsonify({'success': False, 'message': f'获取因子列表失败: {str(e)}'})

def get_traditional_factor_formula(_):
    """V3不再从旧CSV推导传统指标公式，保留占位。"""
    return None

@bp.route('/evaluate', methods=['POST'])
def evaluate_factor():
    """评估因子"""
    try:
        data = request.get_json()
        
        factor_id = data.get('factor_id')
        symbol = data.get('symbol')
        timeframe = data.get('timeframe')
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        exchange = data.get('exchange', 'binance')
        trade_type = data.get('trade_type', 'futures')
        
        if not all([factor_id, symbol, timeframe, start_date, end_date]):
            return jsonify({
                'success': False,
                'message': '缺少必要参数'
            })
        
        # 使用V3引擎直接计算
        engine = get_global_engine()
        # 加载本地数据
        market_data = load_local_market_data(symbol, timeframe, start_date, end_date, exchange, trade_type)
        
        if market_data is None or market_data.empty:
            return jsonify({
                'success': False,
                'message': '无法加载市场数据'
            })
        
        # 计算因子值（V3）
        factor_values = engine.compute_single_factor(factor_id, market_data)
        
        if factor_values is None:
            return jsonify({
                'success': False,
                'message': '因子计算失败'
            })
        
        # 评估因子
        evaluator = FactorEvaluator()
        
        # 简化对齐策略：强制以 market_data.index 为准
        market_data = market_data.sort_index()
        market_data['returns'] = market_data['close'].pct_change()
        
        # 确保因子为Series
        if hasattr(factor_values, 'columns'):
            try:
                factor_values = factor_values.iloc[:, 0]
            except Exception:
                factor_values = factor_values.squeeze()

        # 直接按市场数据索引重建，窗口期产生的NaN后续一起过滤
        factor_values = factor_values.reindex(market_data.index)
        returns = market_data['returns']

        # 同步过滤非空并保证样本数量
        mask = factor_values.notna() & returns.notna()
        factor_values = factor_values[mask]
        returns = returns[mask]

        if len(factor_values) < 30:
            return jsonify({
                'success': False,
                'message': f'数据不足：样本数 {len(factor_values)} < 30，请扩大时间范围或选择更低频时间框架'
            })
        
        # 评估因子
        evaluation_results = evaluator.evaluate_single_factor(
            factor=factor_values,
            returns=returns,
            factor_name=factor_id
        )
        
        # 保存评估结果
        save_evaluation_results(factor_id, evaluation_results, {
            'symbol': symbol,
            'timeframe': timeframe,
            'start_date': start_date,
            'end_date': end_date
        })
        
        return jsonify({
            'success': True,
            'message': '因子评估完成',
            'results': evaluation_results
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'因子评估失败: {str(e)}'
        })

@bp.route('/detail/<factor_id>')
def get_factor_detail(factor_id):
    """获取因子详情"""
    try:
        factor_info = parse_factor_id(factor_id)
        if not factor_info:
            return jsonify({
                'success': False,
                'message': '无效的因子ID'
            })
        
        # 获取因子详细信息
        factor_detail = get_factor_detail_info(factor_info)
        
        return jsonify({
            'success': True,
            'factor': factor_detail
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'获取因子详情失败: {str(e)}'
        })

@bp.route('/export/<factor_id>')
def export_factor(factor_id):
    """导出因子"""
    try:
        # V4导出：从新的文件夹结构导出
        export_dir = FACTOR_LIBRARY_DIR / "minactors"  # 导出到minactors目录
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # 先在minactors中查找
        definition_file = FACTOR_LIBRARY_DIR / "minactors" / "definitions" / f"{factor_id}.json"
        if not definition_file.exists():
            # 再在technicals中查找
            definition_file = FACTOR_LIBRARY_DIR / "technicals" / "definitions" / f"{factor_id}.json"
            if not definition_file.exists():
                return jsonify({'success': False, 'message': '因子定义不存在'})
        
        export_path = export_dir / f"{factor_id}_definition.json"
        with open(definition_file, 'r', encoding='utf-8') as src, open(export_path, 'w', encoding='utf-8') as dst:
            dst.write(src.read())
        
        return jsonify({'success': True, 'message': '因子导出成功', 'export_path': str(export_path)})
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'因子导出失败: {str(e)}'
        })

@bp.route('/evaluations/<factor_id>')
def get_evaluations(factor_id: str):
    """获取某因子的历史评估记录（多结果结构）"""
    try:
        payload = core_load_evaluations(factor_id)
        return jsonify({'success': True, 'factor_id': payload.get('factor_id', factor_id), 'evaluations': payload.get('evaluations', [])})
    except Exception as e:
        return jsonify({'success': False, 'message': f'获取评估历史失败: {str(e)}'})

@bp.route('/batch_evaluate', methods=['POST'])
def batch_evaluate():
    """批量评估：多个因子*多个交易对*多个时间框架"""
    try:
        payload = request.get_json() or {}
        factor_ids = payload.get('factor_ids') or []
        symbols = payload.get('symbols') or []
        timeframes = payload.get('timeframes') or []
        start_date = payload.get('start_date')
        end_date = payload.get('end_date')
        exchange = payload.get('exchange', 'binance')
        trade_type = payload.get('trade_type', 'futures')

        if not factor_ids or not symbols or not timeframes or not start_date or not end_date:
            return jsonify({'success': False, 'message': '缺少必要参数（factor_ids/symbols/timeframes/start_date/end_date）'})

        engine = get_global_engine()
        evaluator = FactorEvaluator()
        results = []

        for factor_id in factor_ids:
            for symbol in symbols:
                for timeframe in timeframes:
                    market_data = load_local_market_data(symbol, timeframe, start_date, end_date, exchange, trade_type)
                    if market_data is None or market_data.empty:
                        results.append({'factor_id': factor_id, 'symbol': symbol, 'timeframe': timeframe, 'success': False, 'message': '无法加载市场数据'})
                        continue
                    try:
                        factor_values = engine.compute_single_factor(factor_id, market_data)
                        if factor_values is None:
                            results.append({'factor_id': factor_id, 'symbol': symbol, 'timeframe': timeframe, 'success': False, 'message': '因子计算失败'})
                            continue
                        market_data = market_data.sort_index()
                        market_data['returns'] = market_data['close'].pct_change()
                        if hasattr(factor_values, 'columns'):
                            try:
                                factor_values = factor_values.iloc[:, 0]
                            except Exception:
                                factor_values = factor_values.squeeze()
                        factor_values = factor_values.reindex(market_data.index)
                        returns = market_data['returns']
                        mask = factor_values.notna() & returns.notna()
                        factor_values = factor_values[mask]
                        returns = returns[mask]
                        if len(factor_values) < 30:
                            results.append({'factor_id': factor_id, 'symbol': symbol, 'timeframe': timeframe, 'success': False, 'message': f'数据不足：样本数 {len(factor_values)} < 30'})
                            continue
                        eval_res = evaluator.evaluate_single_factor(factor=factor_values, returns=returns, factor_name=factor_id)
                        results.append({'factor_id': factor_id, 'symbol': symbol, 'timeframe': timeframe, 'success': True, 'results': eval_res})
                    except Exception as ex:
                        results.append({'factor_id': factor_id, 'symbol': symbol, 'timeframe': timeframe, 'success': False, 'message': str(ex)})
        return jsonify({'success': True, 'results': results})
    except Exception as e:
        return jsonify({'success': False, 'message': f'批量评估失败: {str(e)}'})

def parse_alpha101_filename(filename):
    """解析Alpha101文件名"""
    # 格式: alpha101_results_SYMBOL_TIMEFRAME.pkl
    parts = filename.replace('.pkl', '').split('_')
    if len(parts) >= 4:
        symbol = parts[2]
        timeframe = parts[3]
        return symbol, timeframe
    return 'Unknown', 'Unknown'

def clean_factor_name(factor_name, factor_type=''):
    """清理因子名称，移除不合理的币种后缀"""
    # 需要移除的币种后缀列表
    crypto_symbols = ['BTC', 'ETH', 'BNB', 'SOL', 'ADA', 'DOGE', 'LINK', 'LPT', 'MOVR', 'PEOPLE', 'SUI', 'FIL']
    
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
    
    return factor_name

def parse_factor_id(factor_id):
    """解析因子ID"""
    parts = factor_id.split('_', 2)
    if len(parts) >= 2:
        return {
            'type': parts[0],
            'subtype': parts[1] if len(parts) > 2 else None,
            'identifier': parts[2] if len(parts) > 2 else parts[1]
        }
    return None

def calculate_factor_values(factor_info, market_data):
    """计算因子值"""
    # V3已不再使用该函数，保留占位
    return None

def save_evaluation_results(factor_id, results, metadata):
    """转调核心层的评估结果保存"""
    try:
        core_save_evaluation_results(factor_id, results, metadata)
    except Exception as e:
        print(f"保存评估结果失败: {e}")

def get_factor_detail_info(factor_info):
    """获取因子详细信息"""
    # 这里实现获取因子详细信息的逻辑
    return {
        'id': factor_info.get('identifier'),
        'type': factor_info.get('type'),
        'description': '因子详细信息',
        'formula': '因子计算公式',
        'parameters': {},
        'evaluation_history': []
    }

def export_factor_data(factor_info):
    """导出因子数据"""
    export_dir = FACTOR_LIBRARY_DIR / "minactors"  # 导出到minactors目录
    export_dir.mkdir(parents=True, exist_ok=True)
    
    identifier = factor_info.get('identifier') or factor_info.get('id') or 'factor'
    export_file = export_dir / f"{identifier}_export.csv"
    
    # 创建示例导出文件
    pd.DataFrame({
        'factor_name': [factor_info['identifier']],
        'type': [factor_info['type']],
        'exported_at': [datetime.now().isoformat()]
    }).to_csv(export_file, index=False)
    
    return str(export_file)

def load_local_market_data(symbol, timeframe, start_date, end_date, exchange='binance', trade_type='futures'):
    """从本地加载市场数据"""
    try:
        # 构建文件路径
        data_dir = Path(__file__).parent.parent.parent / "data" / exchange / trade_type
        
        # 解析交易对格式
        # API返回的交易对格式是 BTC_USDT，但实际文件名需要 BTC_USDT_USDT
        if '_' in symbol:
            parts = symbol.split('_')
            if len(parts) >= 2:
                # 如果输入是 BTC_USDT，我们需要构建 BTC_USDT_USDT
                base_symbol = parts[0]  # BTC
                filename = f"{base_symbol}_USDT_USDT-{timeframe}-{trade_type}.feather"
            else:
                base_symbol = symbol
                filename = f"{base_symbol}_USDT_USDT-{timeframe}-{trade_type}.feather"
        else:
            base_symbol = symbol
            filename = f"{base_symbol}_USDT_USDT-{timeframe}-{trade_type}.feather"
            
        print(f"解析交易对: {symbol} -> {base_symbol}")
        print(f"时间框架参数: {timeframe}")
        print(f"构建文件名: {filename}")
        file_path = data_dir / filename
        
        if not file_path.exists():
            print(f"数据文件不存在: {file_path}")
            return None
        
        # 读取feather文件
        data = pd.read_feather(file_path)
        print(f"原始数据形状: {data.shape}")
        print(f"原始数据列名: {list(data.columns)}")
        print(f"原始数据索引: {data.index[:5] if len(data) > 0 else '空'}")
        
        # 确保数据有必要的列
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_columns):
            print(f"数据文件缺少必要的列: {required_columns}")
            return None
        
        # 统一的时间列解析（自动识别秒/毫秒/字符串）
        def parse_time_series(series):
            try:
                if pd.api.types.is_datetime64_any_dtype(series):
                    return series
                if pd.api.types.is_numeric_dtype(series):
                    s = series.dropna()
                    unit = 's'
                    if len(s) > 0:
                        sample = s.iloc[0]
                        unit = 'ms' if sample > 10_000_000_000 else 's'
                    return pd.to_datetime(series, errors='coerce', unit=unit, utc=True)
                return pd.to_datetime(series, errors='coerce', utc=True)
            except Exception:
                return pd.to_datetime(series, errors='coerce', utc=True)
        
        # 如果有时间列，设置为索引
        time_col = None
        for cand in ['timestamp', 'datetime', 'time', 'date']:
            if cand in data.columns:
                time_col = cand
                break
        
        if time_col is not None:
            # 找到时间列，正确设置索引
            print(f"找到时间列: {time_col}")
            data[time_col] = parse_time_series(data[time_col])
            data.set_index(time_col, inplace=True)
        else:
            # 没有找到时间列，检查数据是否已经有正确的索引
            if isinstance(data.index, pd.DatetimeIndex):
                print("数据已有正确的时间索引，无需设置")
            else:
                # 尝试从文件名推断时间信息
                print("没有找到时间列，且数据没有时间索引")
                print("尝试从文件名推断时间信息...")
                
                # 检查数据是否已经是正确的格式（feather文件通常保持索引）
                if len(data) > 0:
                    print(f"数据行数: {len(data)}")
                    print(f"数据列: {list(data.columns)}")
                    print(f"当前索引类型: {type(data.index)}")
                    
                    # 如果数据量很大且没有时间索引，可能是索引丢失了
                    if len(data) > 1000 and not isinstance(data.index, pd.DatetimeIndex):
                        print("警告：数据量大但缺少时间索引，可能无法正确加载")
                        return None
                else:
                    print("数据为空，无法处理")
                    return None
        
        print(f"设置索引后的数据形状: {data.shape}")
        print(f"设置索引后的索引类型: {type(data.index)}")
        print(f"设置索引后的前几行索引: {data.index[:5] if len(data) > 0 else '空'}")
        
        # 过滤日期范围
        if start_date and end_date:
            try:
                # 统一到UTC
                start_dt = pd.to_datetime(start_date, utc=True)
                end_dt = pd.to_datetime(end_date, utc=True)
                
                # 确保索引是datetime类型
                if not isinstance(data.index, pd.DatetimeIndex):
                    print("警告：数据索引不是datetime类型，尝试转换...")
                    try:
                        data.index = pd.to_datetime(data.index, utc=True)
                    except Exception as conv_err:
                        print(f"索引转换失败: {conv_err}")
                        return None
                
                # 统一索引到UTC（若无tz则本地化为UTC）
                try:
                    if data.index.tz is None:
                        data.index = data.index.tz_localize('UTC')
                    else:
                        data.index = data.index.tz_convert('UTC')
                except Exception as tz_err:
                    print(f"时区统一失败: {tz_err}")
                    # 如果时区处理失败，尝试移除时区信息
                    try:
                        data.index = data.index.tz_localize(None)
                        print("已移除时区信息，使用本地时间")
                    except Exception:
                        print("时区处理完全失败")
                        return None
                
                # 执行日期过滤
                original_count = len(data)
                data = data[(data.index >= start_dt) & (data.index <= end_dt)]
                filtered_count = len(data)
                print(f"日期过滤: {original_count} -> {filtered_count} 条记录")
                
                if filtered_count == 0:
                    print("警告：过滤后数据为空，请检查日期范围")
                    print(f"数据时间范围: {data.index.min()} 到 {data.index.max()}")
                    print(f"请求时间范围: {start_dt} 到 {end_dt}")
                    
            except Exception as e:
                print(f"日期过滤失败: {e}")
                # 如果日期过滤失败，返回None而不是继续使用原始数据
                return None
        
        # 确保数据按时间排序
        data.sort_index(inplace=True)
        
        # 最终数据验证
        if len(data) == 0:
            print("❌ 最终数据为空，无法使用")
            return None
        
        # 验证数据完整性
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            print(f"❌ 数据缺少必要列: {missing_columns}")
            return None
        
        # 验证时间索引
        if not isinstance(data.index, pd.DatetimeIndex):
            print("❌ 数据索引不是时间类型")
            return None
        
        # 检查时间范围
        time_range = data.index.max() - data.index.min()
        if time_range.total_seconds() < 60:  # 少于1分钟
            print("❌ 数据时间范围过短")
            return None
        
        print(f"✅ 成功加载数据: {len(data)} 条记录")
        print(f"   时间范围: {data.index.min()} 到 {data.index.max()}")
        print(f"   数据列: {list(data.columns)}")
        return data
        
    except Exception as e:
        print(f"加载本地数据失败: {e}")
        return None 

# 新增：因子计算API
@bp.route('/calculate', methods=['POST'])
def calculate_factor():
    """计算因子值"""
    try:
        data = request.get_json()
        factor_id = data.get('factor_id')
        market_data = data.get('data', {})
        parameters = data.get('parameters', {})
        
        print(f"🔍 收到因子计算请求: {factor_id}")
        print(f"🔍 市场数据长度: {len(market_data.get('close', []))}")
        print(f"🔍 参数: {parameters}")
        
        if not factor_id:
            return jsonify({'success': False, 'error': '缺少factor_id参数'})
        
        # 查找因子定义（先在minactors中查找，再在technicals中查找）
        factor_file = FACTOR_LIBRARY_DIR / "minactors" / "definitions" / f"{factor_id}.json"
        if not factor_file.exists():
            factor_file = FACTOR_LIBRARY_DIR / "technicals" / "definitions" / f"{factor_id}.json"
            if not factor_file.exists():
                print(f"❌ 因子定义文件不存在: {factor_id}")
                return jsonify({'success': False, 'error': f'因子 {factor_id} 不存在'})
        
        with open(factor_file, 'r', encoding='utf-8') as f:
            factor_info = json.load(f)
        
        print(f"🔍 因子信息: {factor_info.get('name', factor_id)}")
        
        # 检查因子类型
        computation_type = factor_info.get('computation_type')
        
        if computation_type == 'formula':
            # 公式因子
            factor_values = calculate_formula_factor(factor_info, market_data, parameters)
        elif computation_type == 'ml':
            # ML因子
            factor_values = calculate_ml_factor(factor_info, market_data, parameters)
        else:
            # 默认使用函数计算
            factor_values = calculate_function_factor(factor_info, market_data, parameters)
        
        if factor_values is not None:
            print(f"✅ 因子计算成功，返回 {len(factor_values)} 个值")
            return jsonify({
                'success': True,
                'factor_values': factor_values,
                'factor_name': factor_info.get('name', factor_id)
            })
        else:
            print(f"❌ 因子计算失败")
            return jsonify({'success': False, 'error': '因子计算失败'})
            
    except Exception as e:
        print(f"❌ 因子计算API异常: {e}")
        return jsonify({'success': False, 'error': str(e)})

def calculate_formula_factor(factor_info, market_data, parameters):
    """计算公式因子"""
    try:
        print(f"🔍 计算公式因子: {factor_info.get('name')}")
        # 这里可以实现公式解析和计算
        # 暂时返回简单的移动平均线作为示例
        close_prices = market_data.get('close', [])
        if not close_prices:
            print("❌ 没有收盘价数据")
            return None
        
        period = parameters.get('period', 20)
        if len(close_prices) < period:
            print(f"❌ 数据长度 {len(close_prices)} 小于周期 {period}")
            return None
        
        # 计算简单移动平均线
        factor_values = []
        for i in range(len(close_prices)):
            if i < period - 1:
                factor_values.append(None)
            else:
                window = close_prices[i-period+1:i+1]
                avg = sum(window) / len(window)
                factor_values.append(avg)
        
        print(f"✅ 公式因子计算完成，返回 {len(factor_values)} 个值")
        return factor_values
    except Exception as e:
        print(f"❌ 公式因子计算失败: {e}")
        return None

def calculate_ml_factor(factor_info, market_data, parameters):
    """计算ML因子"""
    try:
        print(f"🔍 计算ML因子: {factor_info.get('name')}")
        # 这里可以实现ML模型预测
        # 暂时返回随机值作为示例
        close_prices = market_data.get('close', [])
        if not close_prices:
            print("❌ 没有收盘价数据")
            return None
        
        import random
        factor_values = [random.uniform(-1, 1) for _ in range(len(close_prices))]
        print(f"✅ ML因子计算完成，返回 {len(factor_values)} 个值")
        return factor_values
    except Exception as e:
        print(f"❌ ML因子计算失败: {e}")
        return None

def calculate_function_factor(factor_info, market_data, parameters):
    """计算函数因子"""
    try:
        print(f"🔍 计算函数因子: {factor_info.get('name')}")
        # 尝试导入并调用因子函数
        factor_name = factor_info.get('factor_id', '')
        if not factor_name:
            print("❌ 因子ID为空")
            return None
        
        # 构建函数文件路径（先在technicals中查找，再在minactors中查找）
        function_file = FACTOR_LIBRARY_DIR / "technicals" / "functions" / f"{factor_name}.py"
        if not function_file.exists():
            function_file = FACTOR_LIBRARY_DIR / "minactors" / "functions" / f"{factor_name}.py"
            if not function_file.exists():
                print(f"❌ 因子函数文件不存在: {factor_name}")
                return None
        
        print(f"🔍 找到因子函数文件: {function_file}")
        
        # 动态导入因子函数
        import importlib.util
        spec = importlib.util.spec_from_file_location(factor_name, function_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # 准备数据
        df_data = pd.DataFrame({
            'open': market_data.get('open', []),
            'high': market_data.get('high', []),
            'low': market_data.get('low', []),
            'close': market_data.get('close', []),
            'volume': market_data.get('volume', [])
        })
        
        print(f"🔍 准备数据DataFrame: {df_data.shape}")
        
        # 调用calculate函数
        if hasattr(module, 'calculate'):
            print(f"🔍 调用因子函数: {factor_name}.calculate()")
            factor_values = module.calculate(df_data, **parameters)
            
            if isinstance(factor_values, pd.Series):
                result = factor_values.tolist()
            elif isinstance(factor_values, (list, tuple)):
                result = list(factor_values)
            else:
                print(f"❌ 因子函数返回类型不支持: {type(factor_values)}")
                return None
            
            print(f"✅ 函数因子计算完成，返回 {len(result)} 个值")
            return result
        else:
            print(f"❌ 因子函数 {factor_name} 没有calculate函数")
            return None
            
    except Exception as e:
        print(f"❌ 函数因子计算失败: {e}")
        import traceback
        traceback.print_exc()
        return None 