"""
用户配置管理模块
管理用户的交易所API密钥、数据源配置等
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import ccxt
from datetime import datetime


class UserConfigManager:
    """用户配置管理器"""
    
    def __init__(self, config_dir: str = None):
        if config_dir is None:
            config_dir = Path(__file__).parent / "user_config"
        
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.config_file = self.config_dir / "config.json"
        self.load_config()
    
    def load_config(self):
        """加载配置"""
        if self.config_file.exists():
            with open(self.config_file, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        else:
            self.config = {
                'exchanges': [],
                'data_sources': [],
                'preferences': {}
            }
            self.save_config()
    
    def save_config(self):
        """保存配置"""
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
    
    def add_exchange_config(self, name: str, exchange_id: str, api_key: str = None, 
                           secret: str = None, password: str = None, sandbox: bool = False) -> bool:
        """添加交易所配置"""
        try:
            exchange_config = {
                'id': len(self.config['exchanges']) + 1,
                'name': name,
                'exchange_id': exchange_id,
                'api_key': api_key,
                'secret': secret,
                'password': password,
                'sandbox': sandbox,
                'created_at': datetime.now().isoformat()
            }
            
            self.config['exchanges'].append(exchange_config)
            self.save_config()
            return True
        except Exception as e:
            print(f"添加交易所配置失败: {e}")
            return False
    
    def get_exchange_configs(self) -> List[Dict]:
        """获取所有交易所配置"""
        return self.config['exchanges']
    
    def get_exchange_config(self, config_id: int) -> Optional[Dict]:
        """获取指定交易所配置"""
        for config in self.config['exchanges']:
            if config['id'] == config_id:
                return config
        return None
    
    def delete_exchange_config(self, config_id: int) -> bool:
        """删除交易所配置"""
        try:
            self.config['exchanges'] = [
                config for config in self.config['exchanges'] 
                if config['id'] != config_id
            ]
            self.save_config()
            return True
        except Exception as e:
            print(f"删除交易所配置失败: {e}")
            return False
    
    def test_exchange_connection(self, config_id: int) -> Dict:
        """测试交易所连接"""
        try:
            config = self.get_exchange_config(config_id)
            if not config:
                return {'success': False, 'error': '配置不存在'}
            
            exchange_class = getattr(ccxt, config['exchange_id'])
            exchange = exchange_class({
                'apiKey': config['api_key'],
                'secret': config['secret'],
                'password': config['password'],
                'sandbox': config['sandbox'],
                'enableRateLimit': True
            })
            
            markets = exchange.load_markets()
            
            return {
                'success': True,
                'exchange_name': exchange.name,
                'markets_count': len(markets),
                'markets': list(markets.keys())[:10]
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_available_exchanges(self) -> List[str]:
        """获取可用的交易所列表"""
        return ccxt.exchanges
    
    def set_preference(self, key: str, value: str) -> bool:
        """设置用户偏好"""
        try:
            self.config['preferences'][key] = value
            self.save_config()
            return True
        except Exception as e:
            print(f"设置用户偏好失败: {e}")
            return False
    
    def get_preference(self, key: str, default: str = None) -> str:
        """获取用户偏好"""
        return self.config['preferences'].get(key, default)


# 全局配置管理器实例
config_manager = UserConfigManager() 