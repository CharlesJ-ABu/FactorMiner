"""
FactorMiner WebUI Flask应用
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from pathlib import Path
import sys

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from factor_miner import FactorAPI
from config import settings, WEBUI_CONFIG, FRONTEND_CONFIG

def create_app():
    """创建Flask应用"""
    app = Flask(__name__, static_folder='static', static_url_path='/static')
    
    # 配置
    app.config['SECRET_KEY'] = WEBUI_CONFIG['secret_key']
    app.config['MAX_CONTENT_LENGTH'] = WEBUI_CONFIG['max_content_length']
    app.config['DATA_DIR'] = WEBUI_CONFIG['data_dir']
    
    # 启用CORS
    CORS(app)
    
    # 注册路由
    from .routes import main, factors, api, data_api, mining_api
    app.register_blueprint(main.bp)
    app.register_blueprint(factors.bp, url_prefix='/api/factors')  # 仅保留API路由
    app.register_blueprint(api.bp, url_prefix='/api')
    app.register_blueprint(data_api.bp, url_prefix='/api/data')
    app.register_blueprint(mining_api.bp, url_prefix='/api/mining')
    
    # 全局变量
    app.config['FACTOR_API'] = FactorAPI()
    app.config['FRONTEND_CONFIG'] = FRONTEND_CONFIG
    
    return app

app = create_app()

if __name__ == '__main__':
    app.run(
        host=WEBUI_CONFIG['host'],
        port=WEBUI_CONFIG['port'],
        debug=WEBUI_CONFIG['debug']
    ) 