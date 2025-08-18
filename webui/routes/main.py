"""
主路由
"""

from flask import Blueprint, render_template, current_app

bp = Blueprint('main', __name__)

@bp.route('/')
def index():
    """主页"""
    config = current_app.config['FRONTEND_CONFIG']
    return render_template('index.html', config=config)

@bp.route('/about')
def about():
    """关于页面"""
    config = current_app.config['FRONTEND_CONFIG']
    return render_template('about.html', config=config)





@bp.route('/data-download')
def data_download():
    """数据下载页面"""
    config = current_app.config['FRONTEND_CONFIG']
    return render_template('data_management.html', config=config)

@bp.route('/data-viewer')
def data_viewer():
    """数据查看页面"""
    config = current_app.config['FRONTEND_CONFIG']
    return render_template('data_viewer.html', config=config)

@bp.route('/factor-library')
def factor_library():
    """因子库页面"""
    config = current_app.config['FRONTEND_CONFIG']
    return render_template('factor_library.html', config=config)



@bp.route('/factor-evaluation')
def factor_evaluation():
    """因子评估页面"""
    config = current_app.config['FRONTEND_CONFIG']
    return render_template('factor_evaluation.html', config=config)

@bp.route('/factor-mining')
def factor_mining():
    """因子挖掘页面"""
    config = current_app.config['FRONTEND_CONFIG']
    return render_template('factor_mining.html', config=config) 