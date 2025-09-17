"""
因子挖掘API路由
"""

import sys
import os
import json
import uuid
import threading
from datetime import datetime
from pathlib import Path
from flask import Blueprint, request, jsonify

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 现在再导入新的API
from factor_miner.api.factor_mining_api import FactorMiningAPI

# 创建蓝图
bp = Blueprint('mining_api', __name__, url_prefix='/api/mining')

# 全局变量
_mining_api = None

# 挖掘会话管理
mining_sessions = {}
mining_progress = {}

def get_mining_api():
    """获取因子挖掘API实例"""
    global _mining_api
    if _mining_api is None:
        _mining_api = FactorMiningAPI()
    return _mining_api

@bp.route('/start', methods=['POST'])
def start_mining():
    """启动因子挖掘"""
    try:
        data = request.get_json()
        print(f"收到挖掘请求: {data}")
        
        # 验证必要参数
        if not data.get('symbols') or not data.get('timeframes') or not data.get('selected_algorithms'):
            return jsonify({
                'success': False,
                'error': '缺少必要参数：symbols, timeframes, selected_algorithms'
            }), 400
        
        # 生成会话ID
        session_id = str(uuid.uuid4())
        
        # 创建挖掘配置
        mining_config = {
            'symbols': data['symbols'],
            'timeframes': data['timeframes'],
            'selected_algorithms': data['selected_algorithms'],
            'start_date': data.get('start_date'),
            'end_date': data.get('end_date'),
            'session_id': session_id
        }
        
        # 初始化会话
        mining_sessions[session_id] = {
            'status': 'running',
            'progress': 0,
            'current_step': 'initializing',
            'message': '正在初始化挖掘任务...',
            'start_time': datetime.now().isoformat(),
            'config': mining_config
        }
        
        # 启动后台任务
        thread = threading.Thread(
            target=_run_mining_background,
            args=(session_id, data, mining_config)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'message': '挖掘任务已启动'
        })
        
    except Exception as e:
        print(f"启动挖掘失败: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'启动挖掘失败: {str(e)}'
        }), 500

def _run_mining_background(session_id, data, mining_config):
    """后台运行挖掘任务"""
    print(f"开始后台挖掘任务: {session_id}")
    try:
        # 更新进度
        mining_sessions[session_id]['progress'] = 10
        mining_sessions[session_id]['current_step'] = 'data_loading'
        mining_sessions[session_id]['message'] = '正在加载市场数据...'
        
        # 获取挖掘API
        mining_api = get_mining_api()
        
        # 加载数据
        print(f"加载数据: {data['symbols'][0]}, {data['timeframes'][0]}")
        market_data = mining_api.load_data(
            symbol=data['symbols'][0],
            timeframe=data['timeframes'][0],
            start_date=data.get('start_date'),
            end_date=data.get('end_date')
        )
        
        if market_data is None or len(market_data) == 0:
            raise ValueError("数据加载失败或数据为空")
        
        print(f"数据加载成功，共 {len(market_data)} 条记录")
        
        # 更新进度
        mining_sessions[session_id]['progress'] = 30
        mining_sessions[session_id]['current_step'] = 'factor_building'
        mining_sessions[session_id]['message'] = '正在构建因子...'
        
        # 构建因子
        from factor_miner.core.factor_builder import FactorBuilder
        factor_builder = FactorBuilder()
        
        result = factor_builder.build_all_factors(
            data=market_data,
            selected_algorithms=data['selected_algorithms'],
            save_to_storage=True
        )
        
        if not result['success']:
            raise ValueError("因子构建失败")
        
        print(f"因子构建成功，共生成 {result['total_factors']} 个因子")
        
        # 更新进度
        mining_sessions[session_id]['progress'] = 60
        mining_sessions[session_id]['current_step'] = 'evaluation'
        mining_sessions[session_id]['message'] = '正在评估因子...'
        
        # 评估因子
        from factor_miner.core.factor_evaluator import FactorEvaluator
        evaluator = FactorEvaluator()
        
        evaluation_results = {}
        for factor_name, factor_series in result['factors'].items():
            try:
                eval_result = evaluator.evaluate_factor(
                    factor_series, 
                    market_data['close'].pct_change().shift(-1)
                )
                evaluation_results[factor_name] = eval_result
            except Exception as e:
                print(f"评估因子 {factor_name} 失败: {e}")
                continue
        
        print(f"因子评估完成，共评估 {len(evaluation_results)} 个因子")
        
        # 更新进度
        mining_sessions[session_id]['progress'] = 80
        mining_sessions[session_id]['current_step'] = 'optimization'
        mining_sessions[session_id]['message'] = '正在优化因子...'
        
        # 因子优化
        from factor_miner.core.factor_optimizer import FactorOptimizer
        optimizer = FactorOptimizer()
        
        optimization_result = optimizer.optimize_factors(
            result['factors_df'],
            market_data['close'].pct_change().shift(-1)
        )
        
        print(f"因子优化完成，选择了 {len(optimization_result.get('selected_factors', []))} 个因子")
        
        # 保存结果
        mining_sessions[session_id]['progress'] = 90
        mining_sessions[session_id]['current_step'] = 'saving'
        mining_sessions[session_id]['message'] = '正在保存结果...'
        
        # 构建最终结果
        final_result = {
            'factors': result['factors'],
            'factors_df': result['factors_df'].to_dict('records'),
            'total_factors': result['total_factors'],
            'algorithms_used': result['algorithms_used'],
            'evaluation': evaluation_results,
            'optimization': optimization_result
        }
        
        # 保存到文件
        from factor_miner.core.factor_storage import get_global_storage
        storage = get_global_storage()
        storage.save_mining_history(session_id, {
            'session_id': session_id,
            'config': mining_config,
            'results': final_result,
            'status': 'completed',
            'completed_time': datetime.now().isoformat()
        })
        
        # 更新会话状态
        mining_sessions[session_id]['status'] = 'completed'
        mining_sessions[session_id]['progress'] = 100
        mining_sessions[session_id]['current_step'] = 'completed'
        mining_sessions[session_id]['message'] = '挖掘任务完成'
        mining_sessions[session_id]['completed_time'] = datetime.now().isoformat()
        
        print(f"挖掘任务完成: {session_id}")
        
    except Exception as e:
        print(f"挖掘任务失败: {e}")
        import traceback
        traceback.print_exc()
        
        # 更新会话状态为失败
        mining_sessions[session_id]['status'] = 'failed'
        mining_sessions[session_id]['progress'] = 0
        mining_sessions[session_id]['current_step'] = 'failed'
        mining_sessions[session_id]['message'] = f'挖掘失败: {str(e)}'
        mining_sessions[session_id]['error'] = str(e)

@bp.route('/status/<session_id>', methods=['GET'])
def get_mining_status(session_id):
    """获取挖掘状态"""
    if session_id not in mining_sessions:
        return jsonify({'success': False, 'error': '会话不存在'}), 404
    
    session = mining_sessions[session_id]
    return jsonify({
        'success': True,
        'status': session['status'],
        'progress': session['progress'],
        'current_step': session['current_step'],
        'message': session['message'],
        'start_time': session.get('start_time'),
        'completed_time': session.get('completed_time')
    })

@bp.route('/progress/<session_id>', methods=['GET'])
def get_mining_progress(session_id):
    """获取挖掘进度（SSE）"""
    def generate_progress():
        while True:
            if session_id not in mining_sessions:
                yield f"data: {json.dumps({'error': '会话不存在'})}\n\n"
                break
            
            session = mining_sessions[session_id]
            progress_data = {
                'status': session['status'],
                'progress': session['progress'],
                'current_step': session['current_step'],
                'message': session['message']
            }
            
            yield f"data: {json.dumps(progress_data)}\n\n"
            
            if session['status'] in ['completed', 'failed']:
                break
            
            import time
            time.sleep(1)
    
    from flask import Response
    return Response(generate_progress(), mimetype='text/event-stream')

@bp.route('/algorithms', methods=['GET'])
def get_algorithms():
    """获取可用算法列表"""
    try:
        from factor_miner.core.factor_builder import FactorBuilder
        builder = FactorBuilder()
        algorithms = builder.scan_all_algorithms()
        return jsonify({'success': True, 'algorithms': algorithms})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@bp.route('/algorithms/<algorithm_id>', methods=['GET'])
def get_algorithm_info(algorithm_id):
    """获取算法详细信息"""
    try:
        from factor_miner.core.factor_builder import FactorBuilder
        builder = FactorBuilder()
        algorithm_info = builder.get_algorithm_info(algorithm_id)
        if algorithm_info:
            return jsonify({'success': True, 'algorithm': algorithm_info})
        else:
            return jsonify({'success': False, 'error': '算法不存在'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@bp.route('/history', methods=['GET'])
def get_mining_history():
    """获取挖掘历史"""
    try:
        sessions = load_completed_mining_sessions()
        history = []
        
        for session_id, session_data in sessions.items():
            history.append({
                'session_id': session_id,
                'config': session_data.get('config', {}),
                'total_factors': session_data.get('results', {}).get('total_factors', 0),
                'algorithms_used': session_data.get('results', {}).get('algorithms_used', []),
                'completed_time': session_data.get('completed_time'),
                'status': session_data.get('status', 'unknown')
            })
        
        # 按完成时间倒序排列
        history.sort(key=lambda x: x.get('completed_time', ''), reverse=True)
        
        return jsonify({'success': True, 'history': history})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def load_completed_mining_sessions():
    """加载已完成的挖掘会话"""
    try:
        # 优先从mining_sessions.json读取
        sessions_file = Path(__file__).parent.parent.parent / "factorlib" / "minactors" / "mining_history" / "mining_sessions.json"
        
        if sessions_file.exists():
            with open(sessions_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    return json.loads(content)
        
        # 如果mining_sessions.json为空或不存在，返回空字典
        return {}
        
    except Exception as e:
        print(f"加载挖掘会话失败: {e}")
        return {}

@bp.route('/result/<session_id>', methods=['GET'])
def get_mining_result(session_id):
    """获取挖掘结果"""
    try:
        # 先从内存中查找
        if session_id in mining_sessions:
            session = mining_sessions[session_id]
            if session['status'] == 'completed':
                return jsonify({
                    'success': True,
                    'session_id': session_id,
                    'results': session.get('results', {}),
                    'config': session.get('config', {}),
                    'completed_time': session.get('completed_time')
                })
        
        # 从文件中加载
        sessions = load_completed_mining_sessions()
        if session_id in sessions:
            session_data = sessions[session_id]
            return jsonify({
                'success': True,
                'session_id': session_id,
                'results': session_data.get('results', {}),
                'config': session_data.get('config', {}),
                'completed_time': session_data.get('completed_time')
            })
        
        return jsonify({'success': False, 'error': '会话不存在'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def save_mining_result_to_file(session_id, session_data):
    """保存挖掘结果到文件"""
    try:
        # 确保目录存在
        history_dir = Path(__file__).parent.parent.parent / "factorlib" / "minactors" / "mining_history"
        history_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存到mining_sessions.json
        sessions_file = history_dir / "mining_sessions.json"
        
        # 加载现有会话
        try:
            if sessions_file.exists():
                with open(sessions_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        sessions = json.loads(content)
                    else:
                        sessions = {}
            else:
                sessions = {}
        except:
            sessions = {}
        
        # 添加新会话
        sessions[session_id] = session_data
        
        # 保存到文件
        with open(sessions_file, 'w', encoding='utf-8') as f:
            json.dump(sessions, f, ensure_ascii=False, indent=2)
        
        # 保存详细结果到单独文件
        result_file = history_dir / f"mining_results_{session_id}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)
        
        print(f"挖掘结果已保存: {session_id}")
        
    except Exception as e:
        print(f"保存挖掘结果失败: {e}")

def _clean_evaluation_data(evaluation_result):
    """清理评估数据，只保留必要信息"""
    if not isinstance(evaluation_result, dict):
        return evaluation_result
    
    # 只保留这些字段
    cleaned = {}
    for key in ['factor_name', 'ic_pearson', 'ic_spearman', 'sharpe_ratio', 'win_rate', 'long_short_return']:
        if key in evaluation_result:
            cleaned[key] = evaluation_result[key]
    
    return cleaned

def _clean_optimization_data(optimization_result):
    """清理优化数据，只保留必要信息"""
    if not isinstance(optimization_result, dict):
        return optimization_result
    
    cleaned = {}
    for key, value in optimization_result.items():
        if key == 'selected_factors' and isinstance(value, list):
            # 只保留因子名称列表
            cleaned[key] = value
        elif key in ['method', 'score', 'total_factors']:
            # 保留这些重要字段
            cleaned[key] = value
        # 其他字段可能包含大量数据，暂时不保留
    
    return cleaned

@bp.route('/save_selected_factors', methods=['POST'])
def save_selected_factors():
    """保存选中的因子到存储系统"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        factor_ids = data.get('factor_ids', [])
        
        if not session_id:
            return jsonify({'success': False, 'message': '缺少session_id'})
        
        if not factor_ids:
            return jsonify({'success': False, 'message': '没有选择要保存的因子'})
        
        # 从挖掘结果中获取因子定义
        session_data = load_completed_mining_sessions().get(session_id)
        if not session_data:
            return jsonify({'success': False, 'message': '挖掘会话不存在'})
        
        results = session_data.get('results', {})
        factors = results.get('factors', {})
        
        saved_count = 0
        for factor_id in factor_ids:
            if factor_id in factors:
                # 因子已经通过factor_builder保存到存储系统
                # 这里只需要确认保存成功
                saved_count += 1
        
        return jsonify({
            'success': True, 
            'saved_count': saved_count,
            'message': f'成功保存 {saved_count} 个因子'
        })
        
    except Exception as e:
        print(f"❌ 保存选中因子失败: {e}")
        return jsonify({'success': False, 'message': f'保存失败: {str(e)}'})
