"""
因子挖掘API路由
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
from factor_miner.factors.ml_factors import MLFactorBuilder
from factor_miner.factors.technical import FactorCalculator
from factor_miner.factors.statistical import StatisticalFactorBuilder
from factor_miner.factors.advanced import AdvancedFactorBuilder

bp = Blueprint('mining_api', __name__, url_prefix='/api/mining')

# 全局因子挖掘API实例
_mining_api = None
_ml_factor_builder = None
_technical_factor_builder = None
_statistical_factor_builder = None
_advanced_factor_builder = None

# 挖掘会话管理
mining_sessions = {}
mining_progress = {}

# 进度估算配置 - 基于真实算法调整
PROGRESS_ESTIMATES = {
    'data_loading': {
        'base_time': 5,  # 基础时间（秒）
        'time_per_gb': 2,  # 每GB数据额外时间
        'max_progress_steps': 10
    },
    'factor_building': {
        'base_time': 25,  # 基础时间（秒）
        'time_per_factor_type': 12,  # 每种因子类型额外时间
        'time_per_symbol': 5,  # 每个交易对额外时间
        'time_per_ml_algorithm': 25,  # 每个ML算法额外时间
        'time_per_technical_factor': 8,  # 每个技术因子额外时间
        'time_per_statistical_factor': 10,  # 每个统计因子额外时间
        'time_per_advanced_factor': 15,  # 每个高级因子额外时间
        'max_progress_steps': 35
    },
    'factor_evaluation': {
        'base_time': 25,  # 基础时间（秒）
        'time_per_factor': 0.8,  # 每个因子额外时间
        'time_per_symbol': 5,  # 每个交易对额外时间
        'max_progress_steps': 30
    },
    'factor_optimization': {
        'base_time': 15,  # 基础时间（秒）
        'time_per_factor': 0.5,  # 每个因子额外时间
        'max_progress_steps': 20
    },
    'result_saving': {
        'base_time': 5,  # 基础时间（秒）
        'max_progress_steps': 8
    }
}

def get_mining_api():
    """获取因子挖掘API实例"""
    global _mining_api
    if _mining_api is None:
        _mining_api = FactorMiningAPI()
    return _mining_api

def get_ml_factor_builder():
    """获取ML因子构建器实例"""
    global _ml_factor_builder
    if _ml_factor_builder is None:
        _ml_factor_builder = MLFactorBuilder()
    return _ml_factor_builder

def get_technical_factor_builder():
    """获取技术因子构建器实例"""
    global _technical_factor_builder
    if _technical_factor_builder is None:
        _technical_factor_builder = FactorCalculator()
    return _technical_factor_builder

def get_statistical_factor_builder():
    """获取统计因子构建器实例"""
    global _statistical_factor_builder
    if _statistical_factor_builder is None:
        _statistical_factor_builder = StatisticalFactorBuilder()
    return _statistical_factor_builder

def get_advanced_factor_builder():
    """获取高级因子构建器实例"""
    global _advanced_factor_builder
    if _advanced_factor_builder is None:
        _advanced_factor_builder = AdvancedFactorBuilder()
    return _advanced_factor_builder

def estimate_step_time(step_name, config, data_info=None):
    """估算步骤执行时间"""
    if step_name not in PROGRESS_ESTIMATES:
        return 10, 10  # 默认时间和进度步数
    
    estimates = PROGRESS_ESTIMATES[step_name]
    base_time = estimates['base_time']
    max_steps = estimates['max_progress_steps']
    
    # 根据配置调整时间
    if step_name == 'data_loading':
        # 数据加载时间基于数据大小
        if data_info and 'data_size_mb' in data_info:
            data_size_gb = data_info['data_size_mb'] / 1024
            estimated_time = base_time + (data_size_gb * estimates['time_per_gb'])
        else:
            estimated_time = base_time + 5  # 默认额外5秒
        progress_steps = min(max_steps, int(estimated_time / 2))
        
    elif step_name == 'factor_building':
        # 因子构建时间基于因子类型和交易对数量
        factor_types = config.get('factor_types', [])
        symbols = len(config.get('symbols', []))
        
        # 不同类型因子的时间估算
        total_factor_time = 0
        for factor_type in factor_types:
            if factor_type == 'ml':
                total_factor_time += estimates['time_per_ml_algorithm']
            elif factor_type == 'technical':
                total_factor_time += estimates['time_per_technical_factor']
            elif factor_type == 'statistical':
                total_factor_time += estimates['time_per_statistical_factor']
            elif factor_type == 'advanced':
                total_factor_time += estimates['time_per_advanced_factor']
            else:
                total_factor_time += estimates['time_per_factor_type']
        
        estimated_time = base_time + total_factor_time + (symbols * estimates['time_per_symbol'])
        progress_steps = min(max_steps, int(estimated_time / 3))
        
    elif step_name == 'factor_evaluation':
        # 因子评估时间基于因子数量和交易对数量
        factor_types = config.get('factor_types', [])
        symbols = len(config.get('symbols', []))
        
        # 不同类型因子生成的数量估算
        estimated_factors = 0
        for factor_type in factor_types:
            if factor_type == 'ml':
                estimated_factors += 35  # ML类型生成更多因子
            elif factor_type == 'technical':
                estimated_factors += 25  # 技术因子数量
            elif factor_type == 'statistical':
                estimated_factors += 30  # 统计因子数量
            elif factor_type == 'advanced':
                estimated_factors += 20  # 高级因子数量
            else:
                estimated_factors += 20
        
        estimated_time = (base_time + 
                         estimated_factors * estimates['time_per_factor'] +
                         symbols * estimates['time_per_symbol'])
        progress_steps = min(max_steps, int(estimated_time / 4))
        
    elif step_name == 'factor_optimization':
        # 因子优化时间基于因子数量
        factor_types = config.get('factor_types', [])
        estimated_factors = 0
        for factor_type in factor_types:
            if factor_type == 'ml':
                estimated_factors += 35
            elif factor_type == 'technical':
                estimated_factors += 25
            elif factor_type == 'statistical':
                estimated_factors += 30
            elif factor_type == 'advanced':
                estimated_factors += 20
            else:
                estimated_factors += 20
        
        estimated_time = base_time + estimated_factors * estimates['time_per_factor']
        progress_steps = min(max_steps, int(estimated_time / 2))
        
    else:  # result_saving
        estimated_time = base_time
        progress_steps = max_steps
    
    return max(estimated_time, 1), progress_steps

def get_system_info():
    """获取系统信息用于进度估算"""
    try:
        cpu_count = psutil.cpu_count()
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        
        return {
            'cpu_count': cpu_count,
            'memory_gb': round(memory_gb, 1),
            'memory_percent': memory.percent
        }
    except:
        return {
            'cpu_count': 4,
            'memory_gb': 8.0,
            'memory_percent': 50
        }

@bp.route('/start', methods=['POST'])
def start_mining():
    """启动因子挖掘"""
    try:
        data = request.get_json()
        
        # 验证必要参数
        required_fields = ['symbols', 'timeframes', 'factor_types', 'start_date', 'end_date']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'缺少必要参数: {field}'
                }), 400
        
        # 创建挖掘会话
        session_id = str(uuid.uuid4())
        
        # 获取系统信息
        system_info = get_system_info()
        
        # 初始化进度信息
        progress_info = {}
        total_estimated_time = 0
        
        for step_name in ['data_loading', 'factor_building', 'factor_evaluation', 'factor_optimization', 'result_saving']:
            estimated_time, progress_steps = estimate_step_time(step_name, data)
            progress_info[step_name] = {
                'estimated_time': estimated_time,
                'progress_steps': progress_steps,
                'current_progress': 0,
                'start_time': None,
                'current_step_start': None
            }
            total_estimated_time += estimated_time
        
        mining_sessions[session_id] = {
            'status': 'pending',
            'start_time': datetime.now().isoformat(),
            'config': data,
            'progress': {
                'data_loading': 0,
                'factor_building': 0,
                'factor_evaluation': 0,
                'factor_optimization': 0,
                'result_saving': 0
            },
            'progress_info': progress_info,
            'total_estimated_time': total_estimated_time,
            'current_step': 'data_loading',
            'messages': [],
            'system_info': system_info
        }
        
        # 构建挖掘配置
        mining_config = {
            'factor_types': data['factor_types'],
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
        
        # 在后台启动挖掘
        def run_mining_background():
            try:
                api = get_mining_api()
                
                # 更新状态为运行中
                mining_sessions[session_id]['status'] = 'running'
                mining_sessions[session_id]['current_step'] = 'data_loading'
                
                # 步骤1: 数据加载
                step_start_time = time.time()
                mining_sessions[session_id]['progress_info']['data_loading']['start_time'] = step_start_time
                mining_sessions[session_id]['progress_info']['data_loading']['current_step_start'] = step_start_time
                
                update_session_progress(session_id, 'data_loading', 0, '开始加载市场数据...')
                
                # 获取数据大小信息
                data_info = get_data_info(data['symbols'][0], data['timeframes'][0], data['start_date'], data['end_date'])
                
                data_result = api.load_data(
                    data['symbols'][0], 
                    data['timeframes'][0], 
                    data['start_date'], 
                    data['end_date']
                )
                
                if not data_result['success']:
                    raise Exception(f"数据加载失败: {data_result['error']}")
                
                # 更新数据信息
                if 'data_info' not in mining_sessions[session_id]:
                    mining_sessions[session_id]['data_info'] = {}
                mining_sessions[session_id]['data_info'].update(data_info)
                
                update_session_progress(session_id, 'data_loading', 100, '数据加载完成')
                
                # 步骤2: 因子构建
                step_start_time = time.time()
                mining_sessions[session_id]['progress_info']['factor_building']['start_time'] = step_start_time
                mining_sessions[session_id]['progress_info']['factor_building']['current_step_start'] = step_start_time
                # 初始化子进度容器，确保前端可见
                mining_sessions[session_id]['progress_info'].setdefault('sub_progress', {})
                mining_sessions[session_id]['progress_info'].setdefault('sub_messages', {})
                mining_sessions[session_id]['progress_info']['sub_progress'].setdefault('ml', 0)
                
                update_session_progress(session_id, 'factor_building', 0, '开始构建因子...')
                
                # 根据选择的因子类型构建真实因子
                factor_types = data.get('factor_types', [])
                total_types = len(factor_types)
                
                # 因子类型显示名称映射
                factor_type_names = {
                    'technical': '技术因子',
                    'statistical': '统计因子',
                    'advanced': '高级因子',
                    'ml': '机器学习因子',
                    'crypto': '加密因子',
                    'pattern': '形态因子',
                    'composite': '复合因子',
                    'sentiment': '情感因子'
                }
                
                # 构建因子
                all_factors = pd.DataFrame(index=data_result['data'].index)
                
                for i, factor_type in enumerate(factor_types):
                    progress_percent = int((i / total_types) * 80) + 10  # 10% - 90%
                    factor_type_name = factor_type_names.get(factor_type, factor_type)
                    try:
                        if factor_type == 'ml':
                            # 使用真实的ML因子构建算法
                            update_session_progress(session_id, 'factor_building', progress_percent, f'正在构建机器学习因子...')
                            
                            ml_builder = get_ml_factor_builder()
                            
                            # 开始ML因子构建
                            
                            # 执行ML因子构建，带进度更新
                            try:
                                # 开始构建集成学习因子
                                update_session_progress(session_id, 'factor_building', progress_percent + 5, '正在构建集成学习因子...')
                                print(f"开始构建集成学习因子...")
                                
                                # 定义普通挖掘的ML进度回调
                                def ml_progress_callback_normal(stage: str, progress: int, message: str = ""):
                                    try:
                                        pi = mining_sessions[session_id].setdefault('progress_info', {})
                                        sp = pi.setdefault('sub_progress', {})
                                        sm = pi.setdefault('sub_messages', {})
                                        sp[stage] = int(progress)
                                        print(f"🎯 普通挖掘ML进度: {stage} -> {progress}%")
                                        if message:
                                            sm[stage] = message
                                        # 不更新主进度，只更新子进度，避免主进度条"缩放"效果
                                    except Exception as e:
                                        print(f"❌ 普通挖掘ML进度回调错误: {e}")
                                
                                # 让tqdm正常工作，显示真实进度
                                ensemble_factors = ml_builder.build_ensemble_factors(
                                    data_result['data'],
                                    progress_callback=ml_progress_callback_normal,
                                    window=252,
                                    n_estimators=100
                                )
                                print(f"集成学习因子构建完成: {len(ensemble_factors.columns)} 个")
                                
                                # 构建PCA因子
                                update_session_progress(session_id, 'factor_building', progress_percent + 10, '正在构建PCA因子...')
                                print(f"开始构建PCA因子...")
                                pca_factors = ml_builder.build_pca_factors(
                                    data_result['data'],
                                    progress_callback=ml_progress_callback_normal,
                                    n_components=10
                                )
                                print(f"PCA因子构建完成: {len(pca_factors.columns)} 个")
                                
                                # 构建特征选择因子
                                update_session_progress(session_id, 'factor_building', progress_percent + 15, '正在构建特征选择因子...')
                                print(f"开始构建特征选择因子...")
                                feature_factors = ml_builder.build_feature_selection_factors(
                                    data_result['data'],
                                    progress_callback=ml_progress_callback_normal,
                                    k_best=20
                                )
                                print(f"特征选择因子构建完成: {len(feature_factors.columns)} 个")
                                
                                # 构建滚动ML因子
                                update_session_progress(session_id, 'factor_building', progress_percent + 18, '正在构建滚动ML因子...')
                                print(f"开始构建滚动ML因子...")
                                rolling_factors = ml_builder.build_rolling_ml_factors(
                                    data_result['data'],
                                    progress_callback=ml_progress_callback_normal,
                                    window=252,
                                    rolling_window=60
                                )
                                print(f"滚动ML因子构建完成: {len(rolling_factors.columns)} 个")
                                
                                # 构建自适应ML因子
                                update_session_progress(session_id, 'factor_building', progress_percent + 20, '正在构建自适应ML因子...')
                                print(f"开始构建自适应ML因子...")
                                adaptive_factors = ml_builder.build_adaptive_ml_factors(
                                    data_result['data'],
                                    progress_callback=ml_progress_callback_normal,
                                    threshold=0.1
                                )
                                print(f"自适应ML因子构建完成: {len(adaptive_factors.columns)} 个")
                                
                                # 合并所有ML因子
                                ml_factors_list = []
                                if not ensemble_factors.empty:
                                    ml_factors_list.append(ensemble_factors)
                                if not pca_factors.empty:
                                    ml_factors_list.append(pca_factors)
                                if not feature_factors.empty:
                                    ml_factors_list.append(feature_factors)
                                if not rolling_factors.empty:
                                    ml_factors_list.append(rolling_factors)
                                if not adaptive_factors.empty:
                                    ml_factors_list.append(adaptive_factors)
                                
                                if ml_factors_list:
                                    ml_factors = pd.concat(ml_factors_list, axis=1)
                                    all_factors = pd.concat([all_factors, ml_factors], axis=1)
                                    print(f"✓ 成功构建ML因子: {len(ml_factors.columns)} 个")
                                    update_session_progress(session_id, 'factor_building', progress_percent + 20, f'机器学习因子构建完成: {len(ml_factors.columns)} 个')
                                else:
                                    print("✗ ML因子构建完成，但结果为空")
                                    update_session_progress(session_id, 'factor_building', progress_percent + 20, '机器学习因子构建完成，但结果为空')
                                    
                            except Exception as ml_error:
                                print(f"✗ ML因子构建失败: {ml_error}")
                                update_session_progress(session_id, 'factor_building', progress_percent + 20, f'机器学习因子构建失败: {str(ml_error)[:50]}...')
                                
                        elif factor_type == 'technical':
                            # 构建技术因子
                            update_session_progress(session_id, 'factor_building', progress_percent, f'正在构建技术因子...')
                            
                            technical_builder = get_technical_factor_builder()
                            
                            try:
                                # 兼容两种接口：优先 calculate_all_factors，其次 build_all_factors
                                if hasattr(technical_builder, 'calculate_all_factors'):
                                    technical_factors = technical_builder.calculate_all_factors(data_result['data'])
                                else:
                                    technical_factors = technical_builder.build_all_factors(data_result['data'])
                                
                                if not technical_factors.empty:
                                    all_factors = pd.concat([all_factors, technical_factors], axis=1)
                                    print(f"✓ 成功构建技术因子: {len(technical_factors.columns)} 个")
                                    update_session_progress(session_id, 'factor_building', progress_percent + 20, f'技术因子构建完成: {len(technical_factors.columns)} 个')
                                else:
                                    print("✗ 技术因子构建完成，但结果为空")
                                    update_session_progress(session_id, 'factor_building', progress_percent + 20, '技术因子构建完成，但结果为空')
                                    
                            except Exception as e:
                                print(f"✗ 技术因子构建失败: {e}")
                                update_session_progress(session_id, 'factor_building', progress_percent + 20, f'技术因子构建失败: {str(e)[:50]}...')
                                
                        elif factor_type == 'statistical':
                            # 构建统计因子
                            update_session_progress(session_id, 'factor_building', progress_percent, f'正在构建统计因子...')
                            
                            statistical_builder = get_statistical_factor_builder()
                            
                            try:
                                if hasattr(statistical_builder, 'calculate_all_factors'):
                                    statistical_df = statistical_builder.calculate_all_factors(data_result['data'])
                                else:
                                    statistical_df = statistical_builder.build_all_factors(data_result['data'])
                                
                                if not statistical_df.empty:
                                    all_factors = pd.concat([all_factors, statistical_df], axis=1)
                                    print(f"✓ 成功构建统计因子: {len(statistical_df.columns)} 个")
                                    update_session_progress(session_id, 'factor_building', progress_percent + 20, f'统计因子构建完成: {len(statistical_df.columns)} 个')
                                else:
                                    print("✗ 统计因子构建完成，但结果为空")
                                    update_session_progress(session_id, 'factor_building', progress_percent + 20, '统计因子构建完成，但结果为空')
                                    
                            except Exception as e:
                                print(f"✗ 统计因子构建失败: {e}")
                                update_session_progress(session_id, 'factor_building', progress_percent + 20, f'统计因子构建失败: {str(e)[:50]}...')
                                
                        elif factor_type == 'advanced':
                            # 构建高级因子
                            update_session_progress(session_id, 'factor_building', progress_percent, f'正在构建高级因子...')
                            
                            advanced_builder = get_advanced_factor_builder()
                            
                            try:
                                if hasattr(advanced_builder, 'calculate_all_factors'):
                                    advanced_factors = advanced_builder.calculate_all_factors(data_result['data'])
                                else:
                                    advanced_factors = advanced_builder.build_all_factors(data_result['data'])
                                
                                if not advanced_factors.empty:
                                    all_factors = pd.concat([all_factors, advanced_factors], axis=1)
                                    print(f"✓ 成功构建高级因子: {len(advanced_factors.columns)} 个")
                                    update_session_progress(session_id, 'factor_building', progress_percent + 20, f'高级因子构建完成: {len(advanced_factors.columns)} 个')
                                else:
                                    print("✗ 高级因子构建完成，但结果为空")
                                    update_session_progress(session_id, 'factor_building', progress_percent + 20, '高级因子构建完成，但结果为空')
                                    
                            except Exception as e:
                                print(f"✗ 高级因子构建失败: {e}")
                                update_session_progress(session_id, 'factor_building', progress_percent + 20, f'高级因子构建失败: {str(e)[:50]}...')
                                
                        else:
                            # 其他因子类型
                            update_session_progress(session_id, 'factor_building', progress_percent, f'正在构建{factor_type_name}...')
                            
                    except Exception as e:
                        print(f"✗ {factor_type_name}构建失败: {e}")
                        update_session_progress(session_id, 'factor_building', progress_percent, f'{factor_type_name}构建失败: {str(e)[:50]}...')
                
                # 如果因子构建成功，使用构建结果；否则使用原有API
                if not all_factors.empty:
                    # 使用真实构建的因子
                    factors_result = {
                        'success': True,
                        'factors': all_factors,
                        'info': {
                            'total_factors': len(all_factors.columns),
                            'factor_names': list(all_factors.columns),
                            'factor_types': factor_types
                        }
                    }
                else:
                    # 使用原有API构建因子
                    factors_result = api.build_factors(
                        data_result['data'], 
                        data['factor_types'], 
                        mining_config
                    )
                
                if not factors_result['success']:
                    raise Exception(f"因子构建失败: {factors_result['error']}")
                
                # 数据清理和索引对齐
                print(f"开始清理因子数据...")
                factors_df = factors_result['factors']
                print(f"原始因子形状: {factors_df.shape}")
                print(f"原始因子索引范围: {factors_df.index.min()} 到 {factors_df.index.max()}")
                print(f"原始因子索引类型: {type(factors_df.index)}")
                
                # 检查并处理重复索引
                if factors_df.index.duplicated().any():
                    print(f"发现重复索引，开始清理...")
                    duplicate_count = factors_df.index.duplicated().sum()
                    print(f"重复索引数量: {duplicate_count}")
                    
                    # 保留最后一个重复值
                    factors_df = factors_df[~factors_df.index.duplicated(keep='last')]
                    print(f"清理后因子形状: {factors_df.shape}")
                
                # 确保索引是datetime类型
                if not isinstance(factors_df.index, pd.DatetimeIndex):
                    print(f"转换索引为datetime类型...")
                    factors_df.index = pd.to_datetime(factors_df.index)
                
                # 与市场数据对齐
                market_data = data_result['data']
                print(f"市场数据形状: {market_data.shape}")
                print(f"市场数据索引范围: {market_data.index.min()} 到 {market_data.index.max()}")
                
                # 找到共同的索引
                common_index = factors_df.index.intersection(market_data.index)
                print(f"共同索引数量: {len(common_index)}")
                
                if len(common_index) < 100:
                    raise Exception(f"因子数据与市场数据对齐后样本太少: {len(common_index)} < 100")
                
                # 对齐数据
                factors_df_aligned = factors_df.loc[common_index]
                market_data_aligned = market_data.loc[common_index]
                
                print(f"对齐后因子形状: {factors_df_aligned.shape}")
                print(f"对齐后市场数据形状: {market_data_aligned.shape}")
                
                # 检查数据质量
                print(f"因子缺失值统计:")
                for col in factors_df_aligned.columns:
                    missing_count = factors_df_aligned[col].isna().sum()
                    missing_ratio = missing_count / len(factors_df_aligned)
                    print(f"  {col}: {missing_count} ({missing_ratio:.2%})")
                
                # 填充缺失值（使用前向填充）
                factors_df_aligned = factors_df_aligned.fillna(method='ffill').fillna(0)
                print(f"缺失值填充完成")
                
                # 更新数据结果
                data_result['data'] = market_data_aligned
                factors_result['factors'] = factors_df_aligned
                
                update_session_progress(session_id, 'factor_building', 100, f'因子构建完成，共{factors_df_aligned.shape[1]}个因子')
                
                # 步骤3: 因子评估
                step_start_time = time.time()
                mining_sessions[session_id]['progress_info']['factor_evaluation']['start_time'] = step_start_time
                mining_sessions[session_id]['progress_info']['factor_evaluation']['current_step_start'] = step_start_time
                
                update_session_progress(session_id, 'factor_evaluation', 0, '开始评估因子...')
                
                # 开始因子评估
                update_session_progress(session_id, 'factor_evaluation', 10, '正在评估因子...')
                
                # 执行真实的因子评估（传入前端可视化的进度回调）
                evaluation_result = api.evaluate_factors(
                    factors_result['factors'], 
                    data_result['data'], 
                    mining_config
                )
                # 基于最小阈值筛选（只保留符合 min_ic/min_ir 的因子）
                try:
                    min_ic = float(mining_config.get('min_ic', 0)) if isinstance(mining_config, dict) else 0.0
                    min_ir = float(mining_config.get('min_ir', 0)) if isinstance(mining_config, dict) else 0.0
                    eval_map = evaluation_result.get('evaluation', {}) or {}
                    def pass_thresholds(res: dict) -> bool:
                        if not isinstance(res, dict):
                            return False
                        ic_candidates = [res.get('ic_pearson'), res.get('ic_spearman')]
                        ic_values = [abs(v) for v in ic_candidates if isinstance(v, (int, float))]
                        ic_ok = (max(ic_values) if ic_values else 0.0) >= min_ic
                        ir_value = res.get('ic_ir')
                        if not isinstance(ir_value, (int, float)):
                            ir_value = res.get('sharpe_ratio') if isinstance(res.get('sharpe_ratio'), (int, float)) else 0.0
                        ir_ok = ir_value >= min_ir
                        return ic_ok and ir_ok
                    selected_names = [name for name, res in eval_map.items() if pass_thresholds(res)]
                    if selected_names:
                        # 过滤因子与评估映射
                        factors_result['factors'] = factors_result['factors'][selected_names]
                        if 'info' in factors_result:
                            factors_result['info']['factor_names'] = selected_names
                            factors_result['info']['total_factors'] = len(selected_names)
                        evaluation_result['evaluation'] = {k: eval_map[k] for k in selected_names}
                        print(f"筛选后保留因子数量: {len(selected_names)} (min_ic={min_ic}, min_ir={min_ir})")
                    else:
                        print(f"筛选结果为空，保留全部因子 (min_ic={min_ic}, min_ir={min_ir})")
                except Exception as e:
                    print(f"筛选因子失败: {e}")
                
                if not evaluation_result['success']:
                    raise Exception(f"因子评估失败: {evaluation_result['error']}")
                
                update_session_progress(session_id, 'factor_evaluation', 100, '因子评估完成')
                
                # 将评估结果存储到V3系统
                try:
                    update_session_progress(session_id, 'factor_evaluation', 95, '正在保存评估结果到V3系统...')
                    api.save_evaluation_results_to_v3(
                        factors_result['factors'],
                        evaluation_result['evaluation'],
                        mining_config
                    )
                    update_session_progress(session_id, 'factor_evaluation', 100, '评估结果已保存到V3系统')
                except Exception as e:
                    print(f"保存评估结果到V3系统失败: {e}")
                    # 不影响主流程，继续执行
                
                # 步骤4: 因子优化
                step_start_time = time.time()
                mining_sessions[session_id]['progress_info']['factor_optimization']['start_time'] = step_start_time
                mining_sessions[session_id]['progress_info']['factor_optimization']['current_step_start'] = step_start_time
                
                update_session_progress(session_id, 'factor_optimization', 0, '开始优化因子...')
                
                # 开始因子优化
                update_session_progress(session_id, 'factor_optimization', 10, '正在优化因子...')
                
                # 执行真实的因子优化
                optimization_result = api.optimize_factor_combination(
                    factors_result['factors'], 
                    data_result['data'],
                    mining_config
                )
                
                if not optimization_result['success']:
                    raise Exception(f"因子优化失败: {optimization_result['error']}")
                
                update_session_progress(session_id, 'factor_optimization', 100, '因子优化完成')
                
                # 步骤5: 结果保存
                step_start_time = time.time()
                mining_sessions[session_id]['progress_info']['result_saving']['start_time'] = step_start_time
                mining_sessions[session_id]['progress_info']['result_saving']['current_step_start'] = step_start_time
                
                update_session_progress(session_id, 'result_saving', 0, '开始保存结果...')
                
                # 保存结果
                results = {
                    'success': True,
                    'session_id': session_id,
                    'data_info': data_result.get('info', {}),
                    'factors_info': factors_result.get('info', {}),
                    'evaluation': evaluation_result.get('evaluation', {}),
                    'optimization': optimization_result,
                    'output_path': '',
                    'report': f"挖掘完成，共生成 {factors_result.get('info', {}).get('total_factors', 0)} 个因子"
                }
                
                # 保存因子定义到factorlib/definitions文件夹
                try:
                    print(f"开始保存因子定义到factorlib/definitions...")
                    from factor_miner.core.factor_builder import FactorBuilder
                    factor_builder = FactorBuilder()
                    
                    # 构建因子数据字典，格式与原来的_save_factors_to_storage方法一致
                    built_factors = {}
                    for factor_type in data.get('factor_types', []):
                        if factor_type == 'ml' and 'ml_factors' in locals():
                            built_factors['ml'] = {col: ml_factors[col] for col in ml_factors.columns}
                        elif factor_type == 'technical' and 'technical_factors' in locals():
                            built_factors['technical'] = {col: technical_factors[col] for col in technical_factors.columns}
                        elif factor_type == 'statistical' and 'statistical_df' in locals():
                            built_factors['statistical'] = {col: statistical_df[col] for col in statistical_df.columns}
                        elif factor_type == 'advanced' and 'advanced_factors' in locals():
                            built_factors['advanced'] = {col: advanced_factors[col] for col in advanced_factors.columns}
                    
                    # 调用保存方法
                    if built_factors:
                        factor_builder._save_factors_to_storage(built_factors, data_result['data'])
                        print(f"因子定义保存成功到factorlib/definitions")
                    else:
                        print(f"没有找到可保存的因子数据")
                except Exception as e:
                    print(f"保存因子定义失败: {e}")
                    import traceback
                    traceback.print_exc()
                
                # 保存到文件
                try:
                    print(f"开始保存挖掘结果到文件...")
                    output_path = save_mining_results(session_id, results, data)
                    results['output_path'] = str(output_path)
                    print(f"挖掘结果保存成功: {output_path}")
                except Exception as e:
                    print(f"保存结果失败: {e}")  # 使用print而不是logger
                    import traceback
                    traceback.print_exc()
                
                update_session_progress(session_id, 'result_saving', 100, '结果保存完成')
                
                # 更新会话状态
                mining_sessions[session_id]['status'] = 'completed'
                mining_sessions[session_id]['results'] = results
                mining_sessions[session_id]['end_time'] = datetime.now().isoformat()
                
                print(f"挖掘会话 {session_id} 完成")  # 使用print而不是logger
                
            except Exception as e:
                print(f"挖掘会话 {session_id} 失败: {e}")  # 使用print而不是logger
                
                # 更新失败状态的进度
                if 'factor_building' in mining_sessions[session_id]['progress']:
                    update_session_progress(session_id, 'factor_building', 0, f'因子构建失败: {str(e)}')
                elif 'factor_evaluation' in mining_sessions[session_id]['progress']:
                    update_session_progress(session_id, 'factor_evaluation', 0, f'因子评估失败: {str(e)}')
                elif 'factor_optimization' in mining_sessions[session_id]['progress']:
                    update_session_progress(session_id, 'factor_optimization', 0, f'因子优化失败: {str(e)}')
                
                mining_sessions[session_id]['status'] = 'error'
                mining_sessions[session_id]['error'] = str(e)
                mining_sessions[session_id]['end_time'] = datetime.now().isoformat()
        
        # 启动后台线程
        thread = threading.Thread(target=run_mining_background)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'message': '因子挖掘已启动，请使用session_id查询进度',
            'estimated_time': total_estimated_time,
            'system_info': system_info
        })
        
    except Exception as e:
        print(f"启动挖掘失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'启动挖掘失败: {str(e)}'
        }), 500

@bp.route('/ml_mining', methods=['POST'])
def start_ml_mining():
    """启动专门的ML因子挖掘"""
    try:
        data = request.get_json()
        
        # 验证必要参数
        required_fields = ['symbols', 'timeframes', 'start_date', 'end_date']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'缺少必要参数: {field}'
                }), 400
        
        # 强制设置为ML因子类型
        data['factor_types'] = ['ml']
        
        # 创建挖掘会话
        session_id = str(uuid.uuid4())
        
        # 获取系统信息
        system_info = get_system_info()
        
        # 初始化进度信息 - ML挖掘需要更多时间
        progress_info = {}
        total_estimated_time = 0
        
        for step_name in ['data_loading', 'factor_building', 'factor_evaluation', 'factor_optimization', 'result_saving']:
            estimated_time, progress_steps = estimate_step_time(step_name, data)
            progress_info[step_name] = {
                'estimated_time': estimated_time,
                'progress_steps': progress_steps,
                'current_progress': 0,
                'start_time': None,
                'current_step_start': None
            }
            total_estimated_time += estimated_time
        
        mining_sessions[session_id] = {
            'status': 'pending',
            'start_time': datetime.now().isoformat(),
            'config': data,
            'progress': {
                'data_loading': 0,
                'factor_building': 0,
                'factor_evaluation': 0,
                'factor_optimization': 0,
                'result_saving': 0
            },
            'progress_info': progress_info,
            'total_estimated_time': total_estimated_time,
            'current_step': 'data_loading',
            'messages': [],
            'system_info': system_info,
            'mining_type': 'ml'  # 标记为ML挖掘
        }
        
        # 构建ML挖掘配置
        ml_config = {
            'factor_types': ['ml'],
            'ml_params': {
                'window': data.get('window', 252),  # 滚动窗口
                'n_components': data.get('n_components', 10),  # PCA组件数
                'k_best': data.get('k_best', 20),  # 特征选择数量
                'ensemble_models': data.get('ensemble_models', ['random_forest', 'gradient_boosting', 'ridge', 'lasso']),
                'rolling_window': data.get('rolling_window', 252),
                'adaptive_threshold': data.get('adaptive_threshold', 0.8)
            },
            'optimization': {
                'method': data.get('optimization_method', 'greedy'),
                'max_factors': data.get('max_factors', 20),  # ML因子通常更多
                'min_ic': data.get('min_ic', 0.015),  # ML因子IC阈值稍低
                'min_ir': data.get('min_ir', 0.08)
            },
            'evaluation': {
                'min_sample_size': data.get('min_sample_size', 50),  # ML需要更多样本
                'metrics': ['ic_pearson', 'ic_spearman', 'sharpe_ratio', 'win_rate', 'factor_decay', 'stability']
            }
        }
        
        # 在后台启动ML挖掘
        def run_ml_mining_background():
            try:
                api = get_mining_api()
                ml_builder = get_ml_factor_builder()
                
                # 更新状态为运行中
                mining_sessions[session_id]['status'] = 'running'
                mining_sessions[session_id]['current_step'] = 'data_loading'
                
                # 步骤1: 数据加载
                step_start_time = time.time()
                mining_sessions[session_id]['progress_info']['data_loading']['start_time'] = step_start_time
                mining_sessions[session_id]['progress_info']['data_loading']['current_step_start'] = step_start_time
                
                update_session_progress(session_id, 'data_loading', 0, '开始加载市场数据...')
                
                # 获取数据大小信息
                data_info = get_data_info(data['symbols'][0], data['timeframes'][0], data['start_date'], data['end_date'])
                
                data_result = api.load_data(
                    data['symbols'][0], 
                    data['timeframes'][0], 
                    data['start_date'], 
                    data['end_date']
                )
                
                if not data_result['success']:
                    raise Exception(f"数据加载失败: {data_result['error']}")
                
                # 更新数据信息
                if 'data_info' not in mining_sessions[session_id]:
                    mining_sessions[session_id]['data_info'] = {}
                mining_sessions[session_id]['data_info'].update(data_info)
                
                update_session_progress(session_id, 'data_loading', 100, '数据加载完成')
                
                # 步骤2: ML因子构建
                step_start_time = time.time()
                mining_sessions[session_id]['progress_info']['factor_building']['start_time'] = step_start_time
                mining_sessions[session_id]['progress_info']['factor_building']['current_step_start'] = step_start_time
                
                update_session_progress(session_id, 'factor_building', 0, '开始构建ML因子...')
                
                # 构建ML因子，实时回传子进度（供前端显示子进度条）
                def ml_progress_callback(stage: str, progress: int, message: str = ""):
                    try:
                        pi = mining_sessions[session_id].setdefault('progress_info', {})
                        sp = pi.setdefault('sub_progress', {})
                        sm = pi.setdefault('sub_messages', {})
                        sp[stage] = int(progress)
                        print(f"🎯 ML进度回调: {stage} -> {progress}%")
                        if message:
                            sm[stage] = message
                        # 同步推进因子构建主进度（映射到10%~90%区间）
                        mapped = 10 + int(max(0, min(100, progress)) * 0.8)
                        update_session_progress(session_id, 'factor_building', mapped, message or 'ML子进度更新')
                    except Exception as e:
                        print(f"❌ ML进度回调错误: {e}")
                        pass

                ml_factors = ml_builder.build_all_ml_factors(
                    data_result['data'],
                    window=ml_config['ml_params']['window'],
                    n_components=ml_config['ml_params']['n_components'],
                    k_best=ml_config['ml_params']['k_best'],
                    progress_callback=ml_progress_callback
                )
                
                if ml_factors.empty:
                    raise Exception("ML因子构建失败，结果为空")
                
                # 数据清理和索引对齐
                print(f"开始清理ML因子数据...")
                print(f"原始ML因子形状: {ml_factors.shape}")
                print(f"原始ML因子索引范围: {ml_factors.index.min()} 到 {ml_factors.index.max()}")
                print(f"原始ML因子索引类型: {type(ml_factors.index)}")
                
                # 检查并处理重复索引
                if ml_factors.index.duplicated().any():
                    print(f"发现重复索引，开始清理...")
                    duplicate_count = ml_factors.index.duplicated().sum()
                    print(f"重复索引数量: {duplicate_count}")
                    
                    # 保留最后一个重复值
                    ml_factors = ml_factors[~ml_factors.index.duplicated(keep='last')]
                    print(f"清理后ML因子形状: {ml_factors.shape}")
                
                # 确保索引是datetime类型
                if not isinstance(ml_factors.index, pd.DatetimeIndex):
                    print(f"转换索引为datetime类型...")
                    ml_factors.index = pd.to_datetime(ml_factors.index)
                
                # 与市场数据对齐
                market_data = data_result['data']
                print(f"市场数据形状: {market_data.shape}")
                print(f"市场数据索引范围: {market_data.index.min()} 到 {market_data.index.max()}")
                
                # 找到共同的索引
                common_index = ml_factors.index.intersection(market_data.index)
                print(f"共同索引数量: {len(common_index)}")
                
                if len(common_index) < 100:
                    raise Exception(f"因子数据与市场数据对齐后样本太少: {len(common_index)} < 100")
                
                # 对齐数据
                ml_factors_aligned = ml_factors.loc[common_index]
                market_data_aligned = market_data.loc[common_index]
                
                print(f"对齐后ML因子形状: {ml_factors_aligned.shape}")
                print(f"对齐后市场数据形状: {market_data_aligned.shape}")
                
                # 检查数据质量
                print(f"ML因子缺失值统计:")
                for col in ml_factors_aligned.columns:
                    missing_count = ml_factors_aligned[col].isna().sum()
                    missing_ratio = missing_count / len(ml_factors_aligned)
                    print(f"  {col}: {missing_count} ({missing_ratio:.2%})")
                
                # 填充缺失值（使用前向填充）
                ml_factors_aligned = ml_factors_aligned.fillna(method='ffill').fillna(0)
                print(f"缺失值填充完成")
                
                # 更新数据结果
                data_result['data'] = market_data_aligned
                
                factors_result = {
                    'success': True,
                    'factors': ml_factors_aligned,
                    'info': {
                        'total_factors': len(ml_factors_aligned.columns),
                        'factor_names': list(ml_factors_aligned.columns),
                        'factor_types': ['ml'],
                        'ml_params': ml_config['ml_params'],
                        'data_alignment': {
                            'original_shape': ml_factors.shape,
                            'aligned_shape': ml_factors_aligned.shape,
                            'common_index_count': len(common_index),
                            'missing_values_filled': True
                        }
                    }
                }
                
                update_session_progress(session_id, 'factor_building', 100, f'ML因子构建完成，共{len(ml_factors.columns)}个因子')
                
                # 步骤3: 因子评估
                step_start_time = time.time()
                mining_sessions[session_id]['progress_info']['factor_evaluation']['start_time'] = step_start_time
                mining_sessions[session_id]['progress_info']['factor_evaluation']['current_step_start'] = step_start_time
                
                update_session_progress(session_id, 'factor_evaluation', 0, '开始评估ML因子...')
                
                # 执行ML因子评估
                evaluation_result = api.evaluate_factors(
                    factors_result['factors'], 
                    data_result['data'], 
                    ml_config
                )
                
                if not evaluation_result['success']:
                    raise Exception(f"ML因子评估失败: {evaluation_result['error']}")
                
                update_session_progress(session_id, 'factor_evaluation', 100, 'ML因子评估完成')
                
                # 步骤4: 因子优化
                step_start_time = time.time()
                mining_sessions[session_id]['progress_info']['factor_optimization']['start_time'] = step_start_time
                mining_sessions[session_id]['progress_info']['factor_optimization']['current_step_start'] = step_start_time
                
                update_session_progress(session_id, 'factor_optimization', 0, '开始优化ML因子组合...')
                
                # 执行ML因子优化
                optimization_result = api.optimize_factor_combination(
                    factors_result['factors'], 
                    data_result['data'],
                    ml_config
                )
                
                if not optimization_result['success']:
                    raise Exception(f"ML因子优化失败: {optimization_result['error']}")
                
                update_session_progress(session_id, 'factor_optimization', 100, 'ML因子优化完成')
                
                # 步骤5: 结果保存
                step_start_time = time.time()
                mining_sessions[session_id]['progress_info']['result_saving']['start_time'] = step_start_time
                mining_sessions[session_id]['progress_info']['result_saving']['current_step_start'] = step_start_time
                
                update_session_progress(session_id, 'result_saving', 0, '开始保存ML挖掘结果...')
                
                # 保存结果
                results = {
                    'success': True,
                    'session_id': session_id,
                    'mining_type': 'ml',
                    'data_info': data_result.get('info', {}),
                    'factors_info': factors_result.get('info', {}),
                    'evaluation': evaluation_result.get('evaluation', {}),
                    'optimization': optimization_result,
                    'ml_config': ml_config,
                    'output_path': '',
                    'report': f"ML因子挖掘完成，共生成 {factors_result.get('info', {}).get('total_factors', 0)} 个ML因子"
                }
                
                # 保存本次挖掘因子到临时文件，供后续用户选择保存（兼容无pyarrow环境）
                try:
                    temp_dir = Path("factorlib") / "temp"
                    temp_dir.mkdir(parents=True, exist_ok=True)
                    df_tmp = factors_result['factors'].reset_index()
                    factors_file = temp_dir / f"factors_{session_id}.feather"
                    try:
                        df_tmp.to_feather(factors_file)
                        results['factors_file'] = str(factors_file)
                        results['factors_format'] = 'feather'
                        print(f"临时因子文件已保存(feather): {factors_file}")
                    except Exception as e_feather:
                        print(f"保存feather失败，改用pickle: {e_feather}")
                        factors_file = temp_dir / f"factors_{session_id}.pkl"
                        df_tmp.to_pickle(factors_file)
                        results['factors_file'] = str(factors_file)
                        results['factors_format'] = 'pickle'
                        print(f"临时因子文件已保存(pickle): {factors_file}")
                except Exception as e:
                    print(f"保存临时因子文件失败: {e}")
                    import traceback
                    traceback.print_exc()

                # 对比本次挖掘因子与库内因子，供用户决策是否保存
                try:
                    from factor_miner.core.factor_diff import compare_mined_factors_with_library
                    diff_report = compare_mined_factors_with_library(factors_result['factors'])
                    results['diff_report'] = diff_report
                    print(f"因子对比完成：新增 {diff_report['summary']['new']}，差异 {diff_report['summary']['different']}")
                except Exception as e:
                    print(f"因子对比失败: {e}")
                    import traceback
                    traceback.print_exc()
                
                # 保存到文件
                try:
                    print(f"开始保存ML挖掘结果到文件...")
                    output_path = save_mining_results(session_id, results, data)
                    results['output_path'] = str(output_path)
                    print(f"ML挖掘结果保存成功: {output_path}")
                except Exception as e:
                    print(f"保存ML结果失败: {e}")
                    import traceback
                    traceback.print_exc()
                
                update_session_progress(session_id, 'result_saving', 100, 'ML挖掘结果保存完成')
                # 标记子阶段完成（供前端子进度使用）
                mining_sessions[session_id].setdefault('progress_info', {}).setdefault('sub_progress', {})['ml'] = 100
                
                # 更新会话状态
                mining_sessions[session_id]['status'] = 'completed'
                mining_sessions[session_id]['results'] = results
                mining_sessions[session_id]['end_time'] = datetime.now().isoformat()
                
                print(f"ML挖掘会话 {session_id} 完成")
                
            except Exception as e:
                print(f"ML挖掘会话 {session_id} 失败: {e}")
                
                # 更新失败状态的进度
                if 'factor_building' in mining_sessions[session_id]['progress']:
                    update_session_progress(session_id, 'factor_building', 0, f'ML因子构建失败: {str(e)}')
                elif 'factor_evaluation' in mining_sessions[session_id]['progress']:
                    update_session_progress(session_id, 'factor_evaluation', 0, f'ML因子评估失败: {str(e)}')
                elif 'factor_optimization' in mining_sessions[session_id]['progress']:
                    update_session_progress(session_id, 'factor_optimization', 0, f'ML因子优化失败: {str(e)}')
                
                mining_sessions[session_id]['status'] = 'error'
                mining_sessions[session_id]['error'] = str(e)
                mining_sessions[session_id]['end_time'] = datetime.now().isoformat()
        
        # 启动后台线程
        thread = threading.Thread(target=run_ml_mining_background)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'message': 'ML因子挖掘已启动，请使用session_id查询进度',
            'estimated_time': total_estimated_time,
            'system_info': system_info,
            'mining_type': 'ml'
        })
        
    except Exception as e:
        print(f"启动ML挖掘失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'启动ML挖掘失败: {str(e)}'
        }), 500

def get_data_info(symbol, timeframe, start_date, end_date):
    """获取数据信息用于进度估算"""
    try:
        # 构建数据文件路径
        data_dir = Path("data/binance/futures")
        if not data_dir.exists():
            return {'data_size_mb': 100, 'record_count': 1000}  # 默认值
        
        # 查找对应的数据文件
        data_file = data_dir / f"{symbol}_USDT_USDT-{timeframe}-futures.feather"
        if data_file.exists():
            # 获取文件大小
            file_size = data_file.stat().st_size
            data_size_mb = file_size / (1024 * 1024)
            
            # 估算记录数（基于文件大小）
            record_count = int(data_size_mb * 100)  # 粗略估算
            
            return {
                'data_size_mb': round(data_size_mb, 2),
                'record_count': record_count,
                'file_path': str(data_file)
            }
        else:
            return {'data_size_mb': 100, 'record_count': 1000}  # 默认值
            
    except Exception as e:
        print(f"获取数据信息失败: {e}")
        return {'data_size_mb': 100, 'record_count': 1000}  # 默认值

@bp.route('/status/<session_id>', methods=['GET'])
def get_mining_status(session_id):
    """获取挖掘状态和进度"""
    if session_id not in mining_sessions:
        return jsonify({
            'success': False,
            'error': '挖掘会话不存在'
        }), 404
    
    session = mining_sessions[session_id]
    
    # 计算时间信息
    time_info = calculate_time_info(session)
    
    return jsonify({
        'success': True,
        'session_id': session_id,
        'status': session['status'],
        'progress': session['progress'],
        'current_step': session['current_step'],
        'messages': session.get('messages', []),
        'start_time': session['start_time'],
        'end_time': session.get('end_time'),
        'error': session.get('error'),
        'time_info': time_info,
        'progress_info': session.get('progress_info', {}),
        'system_info': session.get('system_info', {})
    })

def calculate_time_info(session):
    """计算时间信息"""
    try:
        current_time = time.time()
        start_time = datetime.fromisoformat(session['start_time']).timestamp()
        elapsed_time = current_time - start_time
        
        # 获取当前步骤信息
        current_step = session.get('current_step', 'data_loading')
        progress_info = session.get('progress_info', {})
        
        if current_step not in progress_info:
            return {
                'elapsed_time': round(elapsed_time, 1),
                'estimated_remaining': 0,
                'estimated_total': session.get('total_estimated_time', 0),
                'current_step_progress': 0,
                'current_step_elapsed': 0,
                'current_step_remaining': 0
            }
        
        current_step_info = progress_info[current_step]
        current_step_start = current_step_info.get('current_step_start')
        
        if current_step_start:
            current_step_elapsed = current_time - current_step_start
            current_step_estimated = current_step_info.get('estimated_time', 10)
            
            # 计算当前步骤的剩余时间
            if current_step_elapsed < current_step_estimated:
                current_step_remaining = current_step_estimated - current_step_elapsed
            else:
                current_step_remaining = 0
            
            # 计算总体剩余时间
            total_estimated = session.get('total_estimated_time', 0)
            if total_estimated > elapsed_time:
                estimated_remaining = total_estimated - elapsed_time
            else:
                estimated_remaining = 0
            
            return {
                'elapsed_time': round(elapsed_time, 1),
                'estimated_remaining': round(estimated_remaining, 1),
                'estimated_total': total_estimated,
                'current_step_progress': session['progress'].get(current_step, 0),
                'current_step_elapsed': round(current_step_elapsed, 1),
                'current_step_remaining': round(current_step_remaining, 1),
                'current_step_estimated': current_step_estimated
            }
        else:
            return {
                'elapsed_time': round(elapsed_time, 1),
                'estimated_remaining': session.get('total_estimated_time', 0),
                'estimated_total': session.get('total_estimated_time', 0),
                'current_step_progress': 0,
                'current_step_elapsed': 0,
                'current_step_remaining': 0
            }
            
    except Exception as e:
        print(f"计算时间信息失败: {e}")
        return {
            'elapsed_time': 0,
            'estimated_remaining': 0,
            'estimated_total': 0,
            'current_step_progress': 0,
            'current_step_elapsed': 0,
            'current_step_remaining': 0
        }

@bp.route('/progress/<session_id>', methods=['GET'])
def get_mining_progress(session_id):
    """获取挖掘进度（SSE流）"""
    if session_id not in mining_sessions:
        return jsonify({
            'success': False,
            'error': '挖掘会话不存在'
        }), 404
    
    def generate_progress():
        session = mining_sessions[session_id]
        last_progress = None
        last_time_info = None
        last_sub_progress = None
        
        while session['status'] in ['pending', 'running']:
            current_progress = session['progress']
            current_time_info = calculate_time_info(session)
            current_sub_progress = session.get('progress_info', {}).get('sub_progress', {})
            
            # 检查进度或时间信息是否有变化
            progress_changed = current_progress != last_progress
            time_changed = current_time_info != last_time_info
            sub_changed = current_sub_progress != last_sub_progress
            
            if progress_changed or time_changed or sub_changed:
                data = {
                    'session_id': session_id,
                    'status': session['status'],
                    'progress': current_progress,
                    'current_step': session['current_step'],
                    'messages': session.get('messages', []),
                    'time_info': current_time_info,
                    'progress_info': session.get('progress_info', {}),
                    'system_info': session.get('system_info', {})
                }
                # 确保sub_progress和sub_messages结构存在，但不强制回填主进度
                try:
                    pi = data.setdefault('progress_info', {})
                    pi.setdefault('sub_progress', {})
                    pi.setdefault('sub_messages', {})
                except Exception:
                    pass
                
                yield f"data: {json.dumps(data)}\n\n"
                last_progress = current_progress.copy()
                last_time_info = current_time_info.copy()
                last_sub_progress = current_sub_progress.copy() if isinstance(current_sub_progress, dict) else current_sub_progress
            
            time.sleep(0.5)  # 更频繁的更新（每0.5秒）
        
        # 发送最终状态
        final_data = {
            'session_id': session_id,
            'status': session['status'],
            'progress': session['progress'],
            'current_step': session['current_step'],
            'messages': session.get('messages', []),
            'results': session.get('results'),
            'error': session.get('error'),
            'time_info': calculate_time_info(session),
            'progress_info': session.get('progress_info', {}),
            'system_info': session.get('system_info', {})
        }
        
        yield f"data: {json.dumps(final_data)}\n\n"
    
    return Response(generate_progress(), mimetype='text/event-stream')

@bp.route('/diff/<session_id>', methods=['GET'])
def get_mining_diff(session_id: str):
    """获取某次挖掘与库内因子的对比报告"""
    try:
        if session_id not in mining_sessions:
            return jsonify({'success': False, 'message': '会话不存在'}), 404
        results = mining_sessions[session_id].get('results') or {}
        diff_report = results.get('diff_report') or {}
        # 如果内存中没有对比报告，基于临时因子文件即时生成一次
        if not diff_report or not diff_report.get('items'):
            factors_file = results.get('factors_file')
            factors_format = results.get('factors_format')
            try:
                if factors_file and Path(factors_file).exists():
                    import pandas as pd
                    if factors_format == 'pickle' or factors_file.endswith('.pkl'):
                        df = pd.read_pickle(factors_file).set_index('index')
                    else:
                        df = pd.read_feather(factors_file).set_index('index')
                    from factor_miner.core.factor_diff import compare_mined_factors_with_library
                    diff_report = compare_mined_factors_with_library(df)
                    # 回写到会话，便于后续读取
                    mining_sessions[session_id]['results']['diff_report'] = diff_report
                else:
                    # 进一步兜底：若没有临时文件，用因子名称列表构造空DataFrame对比
                    factor_names = []
                    info = results.get('factors_info') or {}
                    if 'factor_names' in info:
                        factor_names = info.get('factor_names') or []
                    # 若内存中无，则尝试从历史文件加载
                    if not factor_names:
                        try:
                            results_dir = Path("factorlib") / "mining_history"
                            result_file = results_dir / f"mining_results_{session_id}.json"
                            if result_file.exists():
                                import json as _json
                                with open(result_file, 'r', encoding='utf-8') as f:
                                    stored = _json.load(f)
                                fi = (stored or {}).get('factors_info') or {}
                                factor_names = fi.get('factor_names') or []
                        except Exception:
                            factor_names = []
                    if factor_names:
                        import pandas as pd
                        df = pd.DataFrame(index=pd.RangeIndex(0), data={name: [] for name in factor_names})
                        from factor_miner.core.factor_diff import compare_mined_factors_with_library
                        diff_report = compare_mined_factors_with_library(df)
                        mining_sessions[session_id]['results']['diff_report'] = diff_report
            except Exception as _:
                pass
        return jsonify({'success': True, 'session_id': session_id, 'diff_report': diff_report})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@bp.route('/save_selected_factors', methods=['POST'])
def save_selected_factors():
    """根据对比结果选择性保存挖掘因子定义（不自动覆盖）"""
    try:
        payload = request.get_json() or {}
        session_id = payload.get('session_id')
        selected = payload.get('factor_ids') or []
        if not session_id or not selected:
            return jsonify({'success': False, 'message': '缺少 session_id 或 factor_ids'}), 400

        if session_id not in mining_sessions:
            return jsonify({'success': False, 'message': '会话不存在'}), 404

        session = mining_sessions[session_id]
        results = session.get('results') or {}
        factors_file = results.get('factors_file')
        if not factors_file or not Path(factors_file).exists():
            return jsonify({'success': False, 'message': '临时因子文件不存在，请重新挖掘'}), 400

        # 加载临时因子数据
        df = pd.read_feather(factors_file).set_index('index')

        # 过滤出选择的列
        missing = [c for c in selected if c not in df.columns]
        if missing:
            return jsonify({'success': False, 'message': f'所选因子不存在: {missing}'}), 400

        subset = df[selected]

        # 保存定义（调用核心构建器的保存逻辑，优先ml_model）
        try:
            from factor_miner.core.factor_builder import FactorBuilder
            builder = FactorBuilder()
            built = {'ml': {col: subset[col] for col in subset.columns}}
            builder._save_factors_to_storage(built, subset)
        except Exception as e:
            return jsonify({'success': False, 'message': f'保存定义失败: {e}'}), 500

        # 仅保存选中因子的评估结果（如果本次会话有评估数据）
        try:
            evaluation_map = (results.get('evaluation') or {})
            if evaluation_map:
                from factor_miner.core.evaluation_io import save_evaluation_results as core_save_eval
                for fid in selected:
                    if fid in evaluation_map:
                        core_save_eval(fid, evaluation_map[fid], {
                            'mining_session': True,
                            'session_id': session_id
                        })
        except Exception as e:
            return jsonify({'success': False, 'message': f'保存评估结果失败: {e}'}), 500

        return jsonify({'success': True, 'saved_count': len(selected), 'saved_factor_ids': selected})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@bp.route('/cancel/<session_id>', methods=['POST'])
def cancel_mining(session_id):
    """取消挖掘会话"""
    if session_id not in mining_sessions:
        return jsonify({
            'success': False,
            'error': '挖掘会话不存在'
        }), 404
    
    session = mining_sessions[session_id]
    if session['status'] in ['completed', 'error']:
        return jsonify({
            'success': False,
            'error': '挖掘会话已完成，无法取消'
        }), 400
    
    session['status'] = 'cancelled'
    session['end_time'] = datetime.now().isoformat()
    
    return jsonify({
        'success': True,
        'message': '挖掘会话已取消'
    })

@bp.route('/history', methods=['GET'])
def get_mining_history():
    """获取挖掘历史"""
    try:
        print("=== 开始获取挖掘历史 ===")
        
        # 从内存中获取活跃会话
        active_sessions = []
        for session_id, session in mining_sessions.items():
            if session['status'] in ['pending', 'running']:
                active_sessions.append({
                    'session_id': session_id,
                    'timestamp': session['start_time'],
                    'status': session['status'],
                    'config': session['config'],
                    'progress': session['progress'],
                    'current_step': session['current_step']
                })
        
        print(f"内存中的活跃会话: {len(active_sessions)}")
        
        # 从文件系统加载已完成的会话
        print("开始加载已完成的会话...")
        completed_sessions = load_completed_mining_sessions()
        print(f"从文件加载的已完成会话: {len(completed_sessions)}")
        
        # 合并所有会话
        all_sessions = active_sessions + completed_sessions
        print(f"总会话数: {len(all_sessions)}")
        
        # 按时间排序（最新的在前）
        all_sessions.sort(key=lambda x: x['timestamp'], reverse=True)
        
        # 打印第一个会话的配置信息作为示例
        if all_sessions:
            first_session = all_sessions[0]
            print(f"第一个会话配置示例:")
            print(f"  session_id: {first_session.get('session_id')}")
            print(f"  config: {first_session.get('config')}")
            print(f"  config类型: {type(first_session.get('config'))}")
            print(f"  config键: {list(first_session.get('config', {}).keys()) if first_session.get('config') else 'None'}")
            print(f"  symbols: {first_session.get('config', {}).get('symbols', [])}")
            print(f"  symbols类型: {type(first_session.get('config', {}).get('symbols', []))}")
            print(f"  timeframes: {first_session.get('config', {}).get('timeframes', [])}")
            print(f"  timeframes类型: {type(first_session.get('config', {}).get('timeframes', []))}")
            print(f"  factor_types: {first_session.get('config', {}).get('factor_types', [])}")
            print(f"  factor_types类型: {type(first_session.get('config', {}).get('factor_types', []))}")
            
            # 检查配置是否被修改
            if 'results' in first_session:
                print(f"  results中的data_info: {first_session['results'].get('data_info', {})}")
                print(f"  results中的factors_info: {first_session['results'].get('factors_info', {})}")
        
        return jsonify({
            'success': True,
            'sessions': all_sessions,
            'total': len(all_sessions),
            'active': len(active_sessions),
            'completed': len(completed_sessions)
        })
        
    except Exception as e:
        print(f"获取挖掘历史失败: {e}")
        return jsonify({
            'success': False,
            'error': f'获取挖掘历史失败: {str(e)}'
        }), 500

@bp.route('/result/<session_id>', methods=['GET'])
def get_mining_result(session_id):
    """获取挖掘结果"""
    try:
        # 检查内存中的会话
        if session_id in mining_sessions:
            session = mining_sessions[session_id]
            if session['status'] == 'completed' and 'results' in session:
                return jsonify(session['results'])
        
        # 从文件加载结果
        try:
            # 构建文件路径
            results_dir = Path("factorlib") / "mining_history"
            result_file = results_dir / f"mining_results_{session_id}.json"
            
            if not result_file.exists():
                return jsonify({
                    'success': False,
                    'error': f'挖掘结果文件不存在: {result_file}'
                }), 404
            
            # 直接读取文件
            with open(result_file, 'r', encoding='utf-8') as f:
                result = json.load(f)
            
            print(f"成功加载挖掘结果: {result_file}")
            return jsonify(result)
            
        except Exception as e:
            print(f"从文件加载挖掘结果失败: {e}")
            return jsonify({
                'success': False,
                'error': f'加载挖掘结果失败: {str(e)}'
            }), 500
        
    except Exception as e:
        print(f"获取挖掘结果失败: {e}")
        return jsonify({
            'success': False,
            'error': f'获取挖掘结果失败: {str(e)}'
        }), 500

@bp.route('/config', methods=['GET'])
def get_mining_config():
    """获取挖掘配置选项"""
    return jsonify({
        'factor_types': [
            {'value': 'technical', 'label': '技术因子', 'description': '基于价格和成交量的技术指标'},
            {'value': 'statistical', 'label': '统计因子', 'description': '基于统计学的因子'},
            {'value': 'advanced', 'label': '高级因子', 'description': '复杂的趋势和动量因子'},
            {'value': 'ml', 'label': '机器学习因子', 'description': '使用ML算法生成的因子'},
            {'value': 'crypto', 'label': '加密因子', 'description': '加密货币特有因子'},
            {'value': 'pattern', 'label': '形态因子', 'description': '价格形态识别因子'},
            {'value': 'composite', 'label': '复合因子', 'description': '多因子组合'},
            {'value': 'sentiment', 'label': '情感因子', 'description': '市场情绪因子'}
        ],
        'optimization_methods': [
            {'value': 'greedy', 'label': '贪心算法', 'description': '快速选择最优因子'},
            {'value': 'genetic', 'label': '遗传算法', 'description': '全局优化，耗时较长'},
            {'value': 'correlation', 'label': '相关性过滤', 'description': '去除高相关性因子'}
        ],
        'default_settings': {
            'max_factors': 15,
            'min_ic': 0.02,
            'min_ir': 0.1,
            'min_sample_size': 30
        }
    })

def update_session_progress(session_id, step, progress, message):
    """更新会话进度"""
    if session_id in mining_sessions:
        session = mining_sessions[session_id]
        session['progress'][step] = progress
        session['current_step'] = step
        
        if message:
            session['messages'].append({
                'timestamp': datetime.now().isoformat(),
                'step': step,
                'message': message,
                'progress': progress
            })
        
        # 限制消息数量
        if len(session['messages']) > 100:
            session['messages'] = session['messages'][-100:]
        
        # 初始化子进度结构，但不强制同步主进度到子进度
        try:
            if step == 'factor_building':
                pi = session.setdefault('progress_info', {})
                sp = pi.setdefault('sub_progress', {})
                # 仅在子进度未初始化时设置为0，不强制同步主进度
                sp.setdefault('ml', 0)
        except Exception:
            pass
        
        # 添加调试日志
        print(f"会话 {session_id} 步骤 {step} 进度: {progress}% - {message}")

def load_completed_mining_sessions():
    """加载已完成的挖掘会话"""
    try:
        print("  load_completed_mining_sessions: 开始加载")
        completed_sessions = []
        results_dir = Path("factorlib") / "mining_history"
        
        if not results_dir.exists():
            print(f"  load_completed_mining_sessions: 结果目录不存在: {results_dir}")
            return completed_sessions
        
        print(f"  load_completed_mining_sessions: 结果目录存在: {results_dir}")
        
        # 查找所有可能的挖掘结果文件
        result_files = []
        
        # 1. 查找 mining_results_*.json 文件
        factor_mining_dir = results_dir / "factor_mining"
        if factor_mining_dir.exists():
            result_files.extend(factor_mining_dir.glob("mining_results_*.json"))
        
        # 2. 查找 factor_results.json 文件
        factor_results_file = results_dir / "factor_results.json"
        if factor_results_file.exists():
            result_files.append(factor_results_file)
        
        # 3. 查找其他可能的挖掘结果文件
        result_files.extend(results_dir.glob("*mining*.json"))
        result_files.extend(results_dir.glob("*factor*.json"))
        
        # 去重
        result_files = list(set(result_files))
        
        for result_file in result_files:
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    result_data = json.load(f)
                
                # 处理不同的文件格式
                if 'factor_results' in result_data:
                    # factor_results.json 格式
                    for factor in result_data['factor_results']:
                        session_id = factor.get('session_id', f"session_{factor.get('factor_id', 'unknown')}")
                        timestamp = factor.get('created_at', datetime.now().isoformat())
                        
                        completed_sessions.append({
                            'session_id': session_id,
                            'timestamp': timestamp,
                            'status': 'completed',
                            'config': {
                                'symbols': factor.get('data_requirements', {}).get('symbols', []),
                                'timeframes': factor.get('data_requirements', {}).get('timeframes', []),
                                'factor_types': [factor.get('category', 'unknown')]
                            },
                            'results': {
                                'total_factors': 1,
                                'selected_factors': [factor.get('name', 'Unknown Factor')],
                                'optimization_score': factor.get('performance', {}).get('ic', 0)
                            },
                            'factors_count': 1,
                            'start_time': timestamp,
                            'end_time': timestamp
                        })
                
                elif 'session_id' in result_data:
                    # mining_results_*.json 格式
                    session_id = result_data.get('session_id', '')
                    if session_id:
                        # 尝试从文件修改时间获取时间戳
                        try:
                            file_stat = result_file.stat()
                            timestamp = datetime.fromtimestamp(file_stat.st_mtime).isoformat()
                        except:
                            # 如果获取文件时间失败，使用当前时间
                            timestamp = datetime.now().isoformat()
                        
                        # 构建标准化的配置信息
                        data_info = result_data.get('data_info', {})
                        factors_info = result_data.get('factors_info', {})
                        
                        config = {
                            'symbols': [data_info.get('symbol', '')] if data_info.get('symbol') else [],
                            'timeframes': [data_info.get('timeframe', '')] if data_info.get('timeframe') else [],
                            'factor_types': factors_info.get('factor_types', [])
                        }
                        
                        # 添加调试日志
                        print(f"构建配置信息: session_id={session_id}")
                        print(f"  data_info: {data_info}")
                        print(f"  factors_info: {factors_info}")
                        print(f"  构建的config: {config}")
                        
                        completed_sessions.append({
                            'session_id': session_id,
                            'timestamp': timestamp,
                            'status': 'completed',
                            'config': config,
                            'results': result_data,
                            'factors_count': factors_info.get('total_factors', 0),
                            'start_time': timestamp,
                            'end_time': timestamp
                        })
                
                elif 'mining_sessions' in result_data:
                    # mining_sessions.json 格式
                    for session in result_data.get('mining_sessions', []):
                        completed_sessions.append({
                            'session_id': session.get('session_id', ''),
                            'timestamp': session.get('timestamp', datetime.now().isoformat()),
                            'status': session.get('status', 'completed'),
                            'config': session.get('config', {}),
                            'results': session.get('results', {}),
                            'factors_count': session.get('results', {}).get('total_factors', 0),
                            'start_time': session.get('timestamp', datetime.now().isoformat()),
                            'end_time': session.get('timestamp', datetime.now().isoformat())
                        })
                
            except Exception as e:
                print(f"加载挖掘结果文件失败 {result_file}: {e}")
                continue
        
        # 去重（基于session_id）
        seen_ids = set()
        unique_sessions = []
        for session in completed_sessions:
            if session['session_id'] not in seen_ids:
                seen_ids.add(session['session_id'])
                unique_sessions.append(session)
        
        print(f"  load_completed_mining_sessions: 去重后会话数: {len(unique_sessions)}")
        print(f"  load_completed_mining_sessions: 加载完成")
        
        return unique_sessions
        
    except Exception as e:
        print(f"加载挖掘会话失败: {e}")
        return []

def save_mining_results(session_id, results, config):
    """保存挖掘结果"""
    try:
        print(f"保存挖掘结果: session_id={session_id}")
        print(f"结果结构: {list(results.keys())}")
        print(f"配置结构: {list(config.keys())}")
        
        # 创建结果目录
        results_dir = Path("factorlib") / "mining_history"
        results_dir.mkdir(parents=True, exist_ok=True)
        print(f"结果目录: {results_dir}")
        
        # 保存结果文件
        output_path = results_dir / f"mining_results_{session_id}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        print(f"结果文件保存: {output_path}")
        
        # 保存因子数据为CSV
        if 'evaluation' in results and results['evaluation']:
            csv_path = results_dir / f"factors_{session_id}.csv"
            factors_df = pd.DataFrame(results['evaluation'])
            factors_df.to_csv(csv_path, index=False)
            print(f"因子CSV保存: {csv_path}")
        else:
            print(f"没有评估结果，跳过CSV保存")
        
        # 同时保存到会话历史文件
        try:
            print(f"开始保存会话历史...")
            save_session_to_history(session_id, results, config)
            print(f"会话历史保存成功")
        except Exception as e:
            print(f"保存会话历史失败: {e}")
            import traceback
            traceback.print_exc()
        
        return output_path
        
    except Exception as e:
        print(f"保存挖掘结果失败: {e}")
        import traceback
        traceback.print_exc()
        raise

def save_session_to_history(session_id, results, config):
    """保存会话到历史文件"""
    try:
        print(f"保存会话历史: session_id={session_id}")
        
        # 创建历史目录
        history_dir = Path("factorlib") / "mining_history"
        history_dir.mkdir(parents=True, exist_ok=True)
        print(f"历史目录: {history_dir}")
        
        # 历史文件路径
        history_file = history_dir / "mining_sessions.json"
        print(f"历史文件: {history_file}")
        
        # 读取现有历史
        if history_file.exists():
            with open(history_file, 'r', encoding='utf-8') as f:
                history_data = json.load(f)
            print(f"读取现有历史: {len(history_data.get('mining_sessions', []))} 个会话")
        else:
            history_data = {"mining_sessions": [], "metadata": {"total_sessions": 0, "last_updated": "", "version": "1.0"}}
            print(f"创建新的历史数据结构")
        
        # 创建新的会话记录
        session_record = {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "status": "completed",
            "config": {
                "symbols": config.get('symbols', []),
                "timeframes": config.get('timeframes', []),
                "factor_types": config.get('factor_types', []),
                "max_factors": config.get('max_factors', 15),
                "optimization_method": config.get('optimization_method', 'greedy'),
                "start_date": config.get('start_date', ''),
                "end_date": config.get('end_date', '')
            },
            "results": {
                "total_factors": results.get('factors_info', {}).get('total_factors', 0),
                "selected_factors": results.get('optimization', {}).get('selected_factors', []),
                "optimization_score": results.get('optimization', {}).get('score', 0),
                "execution_time": 0,  # 可以添加实际执行时间
                "output_path": str(results.get('output_path', ''))
            }
        }
        
        print(f"会话记录: {session_record}")
        
        # 添加到历史列表
        history_data["mining_sessions"].append(session_record)
        
        # 更新元数据
        history_data["metadata"]["total_sessions"] = len(history_data["mining_sessions"])
        history_data["metadata"]["last_updated"] = datetime.now().isoformat()
        
        # 保存历史文件
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"会话 {session_id} 已保存到历史文件")
        
    except Exception as e:
        print(f"保存会话历史失败: {e}")
        import traceback
        traceback.print_exc()
        raise
