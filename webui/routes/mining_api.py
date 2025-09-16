"""
因子挖掘API路由 - 修复版本
提供因子挖掘相关的API接口，包含实时进度反馈
"""

from flask import Blueprint, request, jsonify, current_app, Response
import pandas as pd
import numpy as np
from datetime import datetime
import json
from pathlib import Path
import sys
import os
import time
import threading
from queue import Queue
import uuid
import psutil
import gc

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from factor_miner.api.factor_mining_api import FactorMiningAPI

bp = Blueprint('mining_api', __name__, url_prefix='/api/mining')

# 全局因子挖掘API实例
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
        
        # 验证必要参数
        required_fields = ['symbols', 'timeframes', 'selected_algorithms', 'start_date', 'end_date']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'缺少必要参数: {field}'
                })
        
        # 创建会话
        session_id = str(uuid.uuid4())
        
        # 初始化会话状态
        mining_sessions[session_id] = {
            'status': 'running',
            'start_time': datetime.now().isoformat(),
            'config': data,
            'progress': 0,
            'current_step': 'data_loading',
            'messages': []
        }
        
        # 构建挖掘配置
        mining_config = {
            'selected_algorithms': data.get('selected_algorithms', []),
            'optimization': {
                'method': data.get('optimization_method', 'greedy'),
                'max_factors': data.get('max_factors', 15),
                'min_ic': data.get('min_ic', 0.02),
                'min_ir': data.get('min_ir', 0.1)
            },
            'evaluation': {
                'min_sample_size': data.get('min_sample_size', 30),
                'metrics': ['ic_pearson', 'ic_spearman', 'sharpe_ratio', 'win_rate', 'factor_decay']
            }
        }
        
        # 启动后台挖掘
        thread = threading.Thread(
            target=_run_mining_background,
            args=(session_id, data, mining_config)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'message': '因子挖掘已启动'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

def _run_mining_background(session_id, data, mining_config):
    """后台运行挖掘任务"""
    try:
        # 更新进度
        mining_sessions[session_id]['progress'] = 10
        mining_sessions[session_id]['current_step'] = 'data_loading'
        mining_sessions[session_id]['messages'].append('开始加载数据...')
        
        # 1. 加载数据
        mining_api = get_mining_api()
        data_result = mining_api.load_data(
            symbols=data['symbols'],
            timeframes=data['timeframes'],
            start_date=data['start_date'],
            end_date=data['end_date']
        )
        
        if not data_result['success']:
            mining_sessions[session_id]['status'] = 'failed'
            mining_sessions[session_id]['error'] = data_result['error']
            return
        
        # 更新进度
        mining_sessions[session_id]['progress'] = 30
        mining_sessions[session_id]['current_step'] = 'factor_building'
        mining_sessions[session_id]['messages'].append('开始构建因子...')
        
        # 2. 构建因子
        selected_algorithms = data.get('selected_algorithms', [])
        if not selected_algorithms:
            mining_sessions[session_id]['status'] = 'failed'
            mining_sessions[session_id]['error'] = '未选择任何算法'
            return
        
        factors = mining_api.factor_builder.build_all_factors(
            data_result['data'],
            selected_algorithms=selected_algorithms,
            save_to_storage=False
        )
        
        if factors.empty:
            mining_sessions[session_id]['status'] = 'failed'
            mining_sessions[session_id]['error'] = '未生成任何因子'
            return
        
        # 更新进度
        mining_sessions[session_id]['progress'] = 60
        mining_sessions[session_id]['current_step'] = 'factor_evaluation'
        mining_sessions[session_id]['messages'].append(f'开始评估因子，共 {len(factors.columns)} 个...')
        
        # 3. 评估因子
        evaluation_result = mining_api.evaluate_factors(factors, data_result['data'], mining_config)
        
        # 更新进度
        mining_sessions[session_id]['progress'] = 80
        mining_sessions[session_id]['current_step'] = 'factor_optimization'
        mining_sessions[session_id]['messages'].append('开始优化因子组合...')
        
        # 4. 优化因子
        optimization_result = mining_api.optimize_factor_combination(factors, data_result['data'], mining_config)
        
        # 更新进度
        mining_sessions[session_id]['progress'] = 100
        mining_sessions[session_id]['current_step'] = 'completed'
        mining_sessions[session_id]['status'] = 'completed'
        mining_sessions[session_id]['messages'].append('因子挖掘完成！')
        
        # 保存结果
        mining_sessions[session_id]['results'] = {
            'factors': factors.to_dict('records'),
            'evaluation': evaluation_result,
            'optimization': optimization_result,
            'data_info': data_result['info']
        }
        
    except Exception as e:
        mining_sessions[session_id]['status'] = 'failed'
        mining_sessions[session_id]['error'] = str(e)
        print(f"挖掘任务失败: {e}")

@bp.route('/status/<session_id>', methods=['GET'])
def get_mining_status(session_id):
    """获取挖掘状态"""
    if session_id not in mining_sessions:
        return jsonify({
            'success': False,
            'error': '会话不存在'
        })
    
    session = mining_sessions[session_id]
    return jsonify({
        'success': True,
        'session': session
    })

@bp.route('/algorithms', methods=['GET'])
def get_algorithms():
    """获取所有算法列表"""
    try:
        mining_api = get_mining_api()
        algorithms = mining_api.factor_builder.scan_all_algorithms()
        return jsonify({'success': True, 'algorithms': algorithms})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@bp.route('/algorithms/<algorithm_id>', methods=['GET'])
def get_algorithm_info(algorithm_id):
    """获取特定算法信息"""
    try:
        mining_api = get_mining_api()
        algorithm_info = mining_api.factor_builder.get_algorithm_info(algorithm_id)
        if algorithm_info:
            return jsonify({'success': True, 'algorithm': algorithm_info})
        else:
            return jsonify({'success': False, 'error': '算法未找到'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
