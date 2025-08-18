#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FactorMiner WebUI 启动脚本
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from webui.app import app
from config import WEBUI_CONFIG


def main():
    """启动WebUI"""
    print("=== FactorMiner WebUI 启动 ===")
    print(f"访问地址: http://{WEBUI_CONFIG['host']}:{WEBUI_CONFIG['port']}")
    print(f"调试模式: {'开启' if WEBUI_CONFIG['debug'] else '关闭'}")
    print("按 Ctrl+C 停止服务")
    print()
    
    try:
        app.run(
            host=WEBUI_CONFIG['host'],
            port=WEBUI_CONFIG['port'],
            debug=WEBUI_CONFIG['debug']
        )
    except KeyboardInterrupt:
        print("\n服务已停止")
    except Exception as e:
        print(f"启动失败: {e}")


if __name__ == "__main__":
    main() 