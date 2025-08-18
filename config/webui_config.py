"""
WebUI 配置文件
"""

import os
from pathlib import Path

# WebUI配置
WEBUI_CONFIG = {
    'host': '0.0.0.0',
    'port': 8080,
    'debug': True,
    'secret_key': os.environ.get('SECRET_KEY', 'your-secret-key-here'),
    'upload_folder': Path(__file__).parent.parent / "webui" / "uploads",
    'data_dir': Path(__file__).parent.parent / "data",
    'max_content_length': 16 * 1024 * 1024  # 16MB
}

# 创建上传目录
WEBUI_CONFIG['upload_folder'].mkdir(exist_ok=True)

# 前端配置
FRONTEND_CONFIG = {
    'title': 'FactorMiner - 量化因子挖掘平台',
    'version': '1.0.0',
    'description': '专业的量化因子挖掘、评估和优化平台',
    'author': 'FactorMiner Team'
}

# API配置
API_CONFIG = {
    'version': 'v1',
    'prefix': '/api/v1',
    'rate_limit': '100 per minute',
    'cors_origins': ['http://localhost:3000', 'http://127.0.0.1:3000']
} 