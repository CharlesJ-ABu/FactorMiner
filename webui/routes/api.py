"""
API路由
"""

from flask import Blueprint, request, jsonify, current_app

bp = Blueprint('api', __name__)

@bp.route('/factor-info', methods=['GET'])
def get_factor_info():
    """获取因子信息"""
    try:
        api = current_app.config['FACTOR_API']
        result = api.get_factor_info()
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@bp.route('/load-data', methods=['POST'])
def load_data():
    """加载数据"""
    try:
        data = request.get_json()
        symbol = data.get('symbol', 'BTC_USDT')
        timeframe = data.get('timeframe', '1h')
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        
        api = current_app.config['FACTOR_API']
        result = api.load_data(symbol, timeframe, start_date, end_date)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@bp.route('/build-factors', methods=['POST'])
def build_factors():
    """构建因子"""
    try:
        data = request.get_json()
        market_data = data.get('market_data')
        factor_types = data.get('factor_types', ['technical', 'statistical'])
        
        # 这里需要处理数据格式转换
        # 暂时返回模拟结果
        return jsonify({
            'success': True,
            'info': {
                'total_factors': 50,
                'factor_names': ['factor_1', 'factor_2', 'factor_3'],
                'data_points': 1000
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@bp.route('/run-analysis', methods=['POST'])
def run_analysis():
    """运行完整分析"""
    try:
        data = request.get_json()
        symbol = data.get('symbol', 'BTC_USDT')
        timeframe = data.get('timeframe', '1h')
        factor_types = data.get('factor_types', ['technical', 'statistical'])
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        
        api = current_app.config['FACTOR_API']
        result = api.run_complete_analysis(
            symbol=symbol,
            timeframe=timeframe,
            factor_types=factor_types,
            start_date=start_date,
            end_date=end_date
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}) 