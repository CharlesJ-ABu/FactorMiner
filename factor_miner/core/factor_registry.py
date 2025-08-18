"""
因子注册和存储系统
统一管理所有类型的因子
"""

import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Union, Callable, Any
from pathlib import Path
from abc import ABC, abstractmethod
import hashlib
import inspect


class FactorDefinition:
    """因子定义类"""
    
    def __init__(self, 
                 factor_id: str,
                 name: str,
                 description: str,
                 category: str,
                 subcategory: str,
                 compute_func: Callable,
                 parameters: Dict[str, Any] = None,
                 dependencies: List[str] = None,
                 output_type: str = 'series',
                 metadata: Dict[str, Any] = None):
        """
        初始化因子定义
        
        Args:
            factor_id: 因子唯一标识符
            name: 因子名称
            description: 因子描述
            category: 主要分类 (technical, statistical, ml, advanced)
            subcategory: 子分类 (momentum, volatility, trend, etc.)
            compute_func: 计算函数
            parameters: 计算参数
            dependencies: 依赖的因子列表
            output_type: 输出类型 (series, dataframe)
            metadata: 元数据
        """
        self.factor_id = factor_id
        self.name = name
        self.description = description
        self.category = category
        self.subcategory = subcategory
        self.compute_func = compute_func
        self.parameters = parameters or {}
        self.dependencies = dependencies or []
        self.output_type = output_type
        self.metadata = metadata or {}
        
        # 自动生成的属性
        self.created_at = datetime.now()
        self.function_source = self._get_function_source()
        self.checksum = self._compute_checksum()
    
    def _get_function_source(self) -> str:
        """获取函数源代码"""
        try:
            return inspect.getsource(self.compute_func)
        except:
            return str(self.compute_func)
    
    def _compute_checksum(self) -> str:
        """计算因子定义的校验和"""
        content = f"{self.factor_id}_{self.function_source}_{str(self.parameters)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'factor_id': self.factor_id,
            'name': self.name,
            'description': self.description,
            'category': self.category,
            'subcategory': self.subcategory,
            'parameters': self.parameters,
            'dependencies': self.dependencies,
            'output_type': self.output_type,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
            'function_source': self.function_source,
            'checksum': self.checksum
        }


class FactorStorage:
    """因子算法定义存储管理器 - 专注于算法定义，不缓存计算结果"""
    
    def __init__(self, storage_dir: str = None):
        """
        初始化存储管理器
        
        Args:
            storage_dir: 存储目录
        """
        if storage_dir is None:
            storage_dir = Path(__file__).parent.parent.parent / "factorlib" / "storage"
        
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # 重新设计目录结构 - 不再永久存储计算结果
        self.definitions_dir = self.storage_dir / "definitions"  # 因子算法定义
        self.evaluations_dir = self.storage_dir / "evaluations"  # 评估结果
        self.temp_dir = self.storage_dir / "temp"                # 临时缓存
        
        for dir_path in [self.definitions_dir, self.evaluations_dir, self.temp_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def save_factor_definition(self, factor_def: FactorDefinition) -> bool:
        """
        保存因子定义（仅保存元数据，不保存函数）
        
        Args:
            factor_def: 因子定义
            
        Returns:
            是否成功
        """
        try:
            # 只保存定义文件，不保存函数（函数在代码中定义）
            def_file = self.definitions_dir / f"{factor_def.factor_id}.json"
            
            # 创建副本用于保存，排除compute_func
            save_data = factor_def.to_dict()
            save_data.pop('function_source', None)  # 移除函数源码
            
            with open(def_file, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            print(f"保存因子定义失败: {e}")
            return False
    
    def load_factor_definition(self, factor_id: str) -> Optional[FactorDefinition]:
        """
        加载因子定义（仅元数据）
        
        Args:
            factor_id: 因子ID
            
        Returns:
            因子定义或None
        """
        try:
            # 加载定义文件
            def_file = self.definitions_dir / f"{factor_id}.json"
            if not def_file.exists():
                return None
            
            with open(def_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 重构因子定义（不包含函数）
            factor_def = FactorDefinition(
                factor_id=data['factor_id'],
                name=data['name'],
                description=data['description'],
                category=data['category'],
                subcategory=data['subcategory'],
                compute_func=None,  # 函数需要从注册表中获取
                parameters=data.get('parameters', {}),
                dependencies=data.get('dependencies', []),
                output_type=data.get('output_type', 'series'),
                metadata=data.get('metadata', {})
            )
            
            return factor_def
            
        except Exception as e:
            print(f"加载因子定义失败: {e}")
            return None
    
    def save_factor_values(self, 
                          factor_id: str,
                          values: Union[pd.Series, pd.DataFrame],
                          symbol: str = 'universal',
                          timeframe: str = 'universal',
                          start_date: str = None,
                          end_date: str = None) -> bool:
        """
        保存因子计算值
        
        Args:
            factor_id: 因子ID
            values: 因子值
            symbol: 交易对
            timeframe: 时间框架
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            是否成功
        """
        try:
            # 构建文件名
            filename = f"{factor_id}_{symbol}_{timeframe}.feather"
            file_path = self.values_dir / filename
            
            # 准备数据
            if isinstance(values, pd.Series):
                df = pd.DataFrame({factor_id: values})
            else:
                df = values.copy()
            
            # 确保索引是datetime类型
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            
            # 过滤日期范围
            if start_date and end_date:
                df = df[(df.index >= start_date) & (df.index <= end_date)]
            
            # 保存为feather格式（高效存储）
            df.reset_index().to_feather(file_path)
            
            # 保存元数据
            metadata = {
                'factor_id': factor_id,
                'symbol': symbol,
                'timeframe': timeframe,
                'start_date': str(df.index.min()) if len(df) > 0 else None,
                'end_date': str(df.index.max()) if len(df) > 0 else None,
                'records_count': len(df),
                'columns': list(df.columns),
                'saved_at': datetime.now().isoformat()
            }
            
            metadata_file = self.metadata_dir / f"{factor_id}_{symbol}_{timeframe}_meta.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            print(f"保存因子值失败: {e}")
            return False
    
    def load_factor_values(self,
                          factor_id: str,
                          symbol: str = 'universal',
                          timeframe: str = 'universal',
                          start_date: str = None,
                          end_date: str = None) -> Optional[Union[pd.Series, pd.DataFrame]]:
        """
        加载因子计算值
        
        Args:
            factor_id: 因子ID
            symbol: 交易对
            timeframe: 时间框架
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            因子值或None
        """
        try:
            filename = f"{factor_id}_{symbol}_{timeframe}.feather"
            file_path = self.values_dir / filename
            
            if not file_path.exists():
                return None
            
            # 读取数据
            df = pd.read_feather(file_path)
            
            # 设置索引
            if 'index' in df.columns:
                df.set_index('index', inplace=True)
            elif len(df.columns) > 1 and 'timestamp' in df.columns:
                df.set_index('timestamp', inplace=True)
            
            # 确保索引是datetime类型
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            
            # 过滤日期范围
            if start_date and end_date:
                df = df[(df.index >= start_date) & (df.index <= end_date)]
            
            # 如果只有一列且列名是factor_id，返回Series
            if len(df.columns) == 1 and df.columns[0] == factor_id:
                return df[factor_id]
            
            return df
            
        except Exception as e:
            print(f"加载因子值失败: {e}")
            return None
    
    def list_factor_definitions(self) -> List[str]:
        """列出所有因子定义ID"""
        definitions = []
        for file_path in self.definitions_dir.glob("*.json"):
            if not file_path.name.endswith("_meta.json"):
                definitions.append(file_path.stem)
        return sorted(definitions)
    
    def list_factor_values(self, factor_id: str = None) -> List[Dict]:
        """
        列出因子值文件
        
        Args:
            factor_id: 指定因子ID，如果为None则列出所有
            
        Returns:
            因子值文件信息列表
        """
        value_files = []
        
        pattern = f"{factor_id}_*.feather" if factor_id else "*.feather"
        
        for file_path in self.values_dir.glob(pattern):
            # 解析文件名
            parts = file_path.stem.split('_')
            if len(parts) >= 3:
                file_factor_id = parts[0]
                symbol = parts[1]
                timeframe = parts[2]
                
                # 尝试加载元数据
                metadata_file = self.metadata_dir / f"{file_path.stem}_meta.json"
                metadata = {}
                if metadata_file.exists():
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                
                value_files.append({
                    'factor_id': file_factor_id,
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'file_path': str(file_path),
                    'file_size': file_path.stat().st_size,
                    'modified_time': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                    'metadata': metadata
                })
        
        return sorted(value_files, key=lambda x: x['factor_id'])
    
    def get_cache_key(self, factor_id: str, symbol: str, timeframe: str, params: Dict) -> str:
        """生成缓存键"""
        content = f"{factor_id}_{symbol}_{timeframe}_{str(sorted(params.items()))}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def save_cache(self, cache_key: str, data: Any, ttl_hours: int = 24) -> bool:
        """保存缓存"""
        try:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            cache_data = {
                'data': data,
                'created_at': datetime.now(),
                'ttl_hours': ttl_hours
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            return True
            
        except Exception as e:
            print(f"保存缓存失败: {e}")
            return False
    
    def load_cache(self, cache_key: str) -> Any:
        """加载缓存"""
        try:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            
            if not cache_file.exists():
                return None
            
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            # 检查TTL
            created_at = cache_data['created_at']
            ttl_hours = cache_data['ttl_hours']
            
            if (datetime.now() - created_at).total_seconds() > ttl_hours * 3600:
                # 缓存过期，删除文件
                cache_file.unlink()
                return None
            
            return cache_data['data']
            
        except Exception as e:
            print(f"加载缓存失败: {e}")
            return None


class FactorRegistry:
    """因子注册中心"""
    
    def __init__(self, storage_dir: str = None):
        """
        初始化注册中心
        
        Args:
            storage_dir: 存储目录
        """
        if storage_dir is None:
            storage_dir = Path(__file__).parent.parent.parent / "factorlib" / "storage"
        self.storage = FactorStorage(storage_dir)
        self.registered_factors: Dict[str, FactorDefinition] = {}
        self._load_existing_factors()
    
    def _load_existing_factors(self):
        """加载已存在的因子定义"""
        for factor_id in self.storage.list_factor_definitions():
            factor_def = self.storage.load_factor_definition(factor_id)
            if factor_def:
                self.registered_factors[factor_id] = factor_def
    
    def register_factor(self, factor_def: FactorDefinition) -> bool:
        """
        注册因子
        
        Args:
            factor_def: 因子定义
            
        Returns:
            是否成功
        """
        # 检查是否已存在
        if factor_def.factor_id in self.registered_factors:
            existing = self.registered_factors[factor_def.factor_id]
            if existing.checksum == factor_def.checksum:
                # print(f"因子 {factor_def.factor_id} 已存在且未发生变化")
                return True
            else:
                print(f"因子 {factor_def.factor_id} 已更新")
        
        # 直接在内存中注册，不持久化保存（避免pickle问题）
        self.registered_factors[factor_def.factor_id] = factor_def
        # print(f"成功注册因子: {factor_def.factor_id}")
        return True
    
    def get_factor(self, factor_id: str) -> Optional[FactorDefinition]:
        """获取因子定义"""
        return self.registered_factors.get(factor_id)
    
    def list_factors(self, 
                    category: str = None, 
                    subcategory: str = None) -> List[FactorDefinition]:
        """
        列出因子
        
        Args:
            category: 过滤分类
            subcategory: 过滤子分类
            
        Returns:
            因子定义列表
        """
        factors = list(self.registered_factors.values())
        
        if category:
            factors = [f for f in factors if f.category == category]
        
        if subcategory:
            factors = [f for f in factors if f.subcategory == subcategory]
        
        return sorted(factors, key=lambda x: x.factor_id)
    
    def compute_factor(self,
                      factor_id: str,
                      data: pd.DataFrame,
                      symbol: str = 'universal',
                      timeframe: str = 'universal',
                      use_cache: bool = True,
                      save_result: bool = True,
                      **kwargs) -> Optional[Union[pd.Series, pd.DataFrame]]:
        """
        计算因子值
        
        Args:
            factor_id: 因子ID
            data: 市场数据
            symbol: 交易对
            timeframe: 时间框架
            use_cache: 是否使用缓存
            save_result: 是否保存结果
            **kwargs: 额外参数
            
        Returns:
            因子值
        """
        factor_def = self.get_factor(factor_id)
        if not factor_def:
            print(f"未找到因子定义: {factor_id}")
            return None
        
        # 合并参数
        params = {**factor_def.parameters, **kwargs}
        
        # 检查缓存
        if use_cache:
            cache_key = self.storage.get_cache_key(factor_id, symbol, timeframe, params)
            cached_result = self.storage.load_cache(cache_key)
            if cached_result is not None:
                print(f"使用缓存结果: {factor_id}")
                return cached_result
        
        try:
            # 计算因子
            print(f"计算因子: {factor_id}")
            result = factor_def.compute_func(data, **params)
            
            # 保存缓存
            if use_cache:
                self.storage.save_cache(cache_key, result)
            
            # 保存结果
            if save_result:
                start_date = str(data.index.min()) if len(data) > 0 else None
                end_date = str(data.index.max()) if len(data) > 0 else None
                self.storage.save_factor_values(
                    factor_id, result, symbol, timeframe, start_date, end_date
                )
            
            return result
            
        except Exception as e:
            print(f"计算因子失败 {factor_id}: {e}")
            return None
    
    def load_factor_values(self, 
                          factor_id: str,
                          symbol: str = 'universal',
                          timeframe: str = 'universal',
                          start_date: str = None,
                          end_date: str = None) -> Optional[Union[pd.Series, pd.DataFrame]]:
        """加载已保存的因子值"""
        return self.storage.load_factor_values(factor_id, symbol, timeframe, start_date, end_date)
    
    def get_factor_info(self) -> Dict:
        """获取因子注册信息统计"""
        categories = {}
        for factor_def in self.registered_factors.values():
            if factor_def.category not in categories:
                categories[factor_def.category] = {}
            
            if factor_def.subcategory not in categories[factor_def.category]:
                categories[factor_def.category][factor_def.subcategory] = 0
            
            categories[factor_def.category][factor_def.subcategory] += 1
        
        return {
            'total_factors': len(self.registered_factors),
            'categories': categories,
            'storage_path': str(self.storage.storage_dir),
            'last_updated': datetime.now().isoformat()
        }


# 全局因子注册中心实例
factor_registry = FactorRegistry()


# 装饰器用于注册因子
def register_factor(factor_id: str,
                   name: str,
                   description: str,
                   category: str,
                   subcategory: str,
                   parameters: Dict[str, Any] = None,
                   dependencies: List[str] = None,
                   output_type: str = 'series',
                   metadata: Dict[str, Any] = None):
    """
    因子注册装饰器
    
    Usage:
        @register_factor(
            factor_id='rsi_14',
            name='RSI 14期',
            description='14期相对强弱指数',
            category='technical',
            subcategory='momentum'
        )
        def calculate_rsi_14(data, period=14):
            # 计算逻辑
            pass
    """
    def decorator(func):
        factor_def = FactorDefinition(
            factor_id=factor_id,
            name=name,
            description=description,
            category=category,
            subcategory=subcategory,
            compute_func=func,
            parameters=parameters,
            dependencies=dependencies,
            output_type=output_type,
            metadata=metadata
        )
        
        factor_registry.register_factor(factor_def)
        return func
    
    return decorator
