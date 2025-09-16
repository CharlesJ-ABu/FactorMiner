"""
统一因子构建器 V4.0
纯调度器，所有算法从 user_algo 目录动态加载
"""

import pandas as pd
from typing import Dict, List, Optional, Callable
from pathlib import Path
import importlib.util
import warnings
warnings.filterwarnings('ignore')

from .factor_storage import TransparentFactorStorage
from .factor_engine import FactorEngine


class FactorBuilder:
    """
    统一因子构建器 V4.0
    纯调度器，所有算法从 user_algo 目录动态加载
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化因子构建器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.storage = TransparentFactorStorage()
        self.engine = FactorEngine()
        
        # 用户算法目录
        self.user_algo_dir = Path(__file__).parent.parent.parent / "user_algo"
        
        # 算法缓存
        self.algorithm_cache = {}
        
    def build_all_factors(self, data: pd.DataFrame, 
                         factor_types: Optional[List[str]] = None,
                         selected_algorithms: Optional[List[str]] = None,
                         save_to_storage: bool = True,
                         progress_callback: Optional[callable] = None,
                         **kwargs) -> pd.DataFrame:
        """
        构建所有类型的因子
        
        Args:
            data: 市场数据
            factor_types: 因子类型列表 (technical, statistical, ml, etc.)
            selected_algorithms: 选中的具体算法列表
            save_to_storage: 是否保存到存储系统
            progress_callback: 进度回调函数
            **kwargs: 其他参数
            
        Returns:
            因子DataFrame
        """
        if factor_types is None:
            factor_types = self._get_available_categories()
        
        if selected_algorithms is None:
            selected_algorithms = self._get_algorithms_by_categories(factor_types)
        
        all_factors = {}
        built_factors = {}
        
        print(f"🚀 开始构建因子，算法: {', '.join(selected_algorithms)}")
        
        for algo_id in selected_algorithms:
            try:
                print(f"📊 执行算法: {algo_id}")
                factors = self._execute_algorithm(algo_id, data, **kwargs)
                
                if isinstance(factors, dict):
                    all_factors.update(factors)
                    built_factors[algo_id] = factors
                    print(f"✅ 算法 {algo_id} 执行成功，生成 {len(factors)} 个因子")
                else:
                    print(f"⚠️ 算法 {algo_id} 返回格式错误")
                    
            except Exception as e:
                print(f"❌ 算法 {algo_id} 执行失败: {e}")
                continue
        
        # 转换为DataFrame
        factors_df = pd.DataFrame(all_factors, index=data.index)
        print(f"🎉 因子构建完成，总共 {len(factors_df.columns)} 个因子")
        
        # 保存到存储系统
        if save_to_storage:
            self._save_factors_to_storage(built_factors, data, **kwargs)
        
        return factors_df
    
    def _execute_algorithm(self, algo_id: str, data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """执行指定算法"""
        # 检查缓存
        if algo_id in self.algorithm_cache:
            module = self.algorithm_cache[algo_id]
        else:
            # 动态加载算法
            module = self._load_algorithm(algo_id)
            if module is None:
                raise ImportError(f"无法加载算法: {algo_id}")
            self.algorithm_cache[algo_id] = module
        
        # 获取算法参数
        algo_params = kwargs.get(f'{algo_id}_params', {})
        
        # 执行算法
        return module.calculate_factors(data, **algo_params)
    
    def _load_algorithm(self, algo_id: str):
        """动态加载算法模块"""
        # 支持多种路径格式
        possible_paths = [
            self.user_algo_dir / f"{algo_id}.py",
        ]
        
        for file_path in possible_paths:
            if file_path.exists():
                try:
                    spec = importlib.util.spec_from_file_location(algo_id, file_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    return module
                except Exception as e:
                    print(f"⚠️ 加载算法 {file_path} 失败: {e}")
                    continue
        
        return None
    
    def scan_all_algorithms(self) -> List[Dict]:
        """扫描所有可用算法"""
        algorithms = []
        
        if not self.user_algo_dir.exists():
            return algorithms
        
        # 扫描 user_algo 目录
        for file_path in self.user_algo_dir.glob("*.py"):
            if file_path.name == "__init__.py":
                continue
                
            try:
                # 动态导入算法文件
                spec = importlib.util.spec_from_file_location(
                    file_path.stem, file_path
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # 检查是否有 calculate_factors 函数
                if hasattr(module, 'calculate_factors'):
                    algo_info = {
                        'id': file_path.stem,
                        'name': file_path.stem,
                        'description': '算法',
                        'file_path': str(file_path),
                        'category': 'custom'
                    }
                    
                    # 如果有算法信息，使用它
                    if hasattr(module, 'ALGORITHM_INFO'):
                        algo_info.update(module.ALGORITHM_INFO)
                    
                    algorithms.append(algo_info)
                    
            except Exception as e:
                print(f"⚠️ 加载算法 {file_path.name} 失败: {e}")
                continue
        
        return algorithms
    
    def _get_available_categories(self) -> List[str]:
        """获取可用算法分类"""
        algorithms = self.scan_all_algorithms()
        categories = set(algo.get('category', 'unknown') for algo in algorithms)
        return sorted(list(categories))
    
    def _get_algorithms_by_categories(self, categories: List[str]) -> List[str]:
        """根据分类获取算法列表"""
        algorithms = self.scan_all_algorithms()
        return [algo['id'] for algo in algorithms if algo.get('category') in categories]
    
    def get_algorithm_info(self, algorithm_id: str) -> Optional[Dict]:
        """获取算法信息"""
        algorithms = self.scan_all_algorithms()
        for algo in algorithms:
            if algo['id'] == algorithm_id:
                return algo
        return None
    
    def list_algorithms(self) -> List[Dict]:
        """列出所有算法"""
        return self.scan_all_algorithms()
    
    def _save_factors_to_storage(self, built_factors: Dict, data: pd.DataFrame, **kwargs):
        """将构建的因子保存到存储系统"""
        print("💾 开始保存因子到存储系统...")
        
        for algo_id, factors in built_factors.items():
            for factor_name, factor_series in factors.items():
                try:
                    # 创建因子定义
                    factor_id = f"{algo_id}_{factor_name}"
                    
                    # 保存为公式类型
                    success = self.storage.save_formula_factor(
                        factor_id=factor_id,
                        name=factor_name,
                        formula=f"# {algo_id} 算法生成的 {factor_name} 因子",
                        description=f"由 {algo_id} 算法生成的 {factor_name} 因子",
                        category=algo_id.split('_')[0] if '_' in algo_id else 'custom',
                        parameters={}
                    )
                    
                    if success:
                        print(f"✅ 因子 {factor_id} 保存成功")
                    else:
                        print(f"❌ 因子 {factor_id} 保存失败")
                        
                except Exception as e:
                    print(f"❌ 保存因子 {factor_name} 时出错: {e}")
                    continue
        
        print("💾 因子保存完成")
    
    def get_available_factors(self) -> Dict[str, List[str]]:
        """获取所有可用因子的信息（兼容性方法）"""
        algorithms = self.scan_all_algorithms()
        categories = {}
        
        for algo in algorithms:
            category = algo.get('category', 'other')
            if category not in categories:
                categories[category] = []
            categories[category].append(algo['id'])
        
        return categories
    
    def build_custom_factor(self, data: pd.DataFrame, factor_name: str, 
                          factor_func: Callable, **kwargs) -> pd.Series:
        """
        构建自定义因子（兼容性方法）
        
        Args:
            data: 市场数据
            factor_name: 因子名称
            factor_func: 因子计算函数
            **kwargs: 其他参数
            
        Returns:
            自定义因子Series
        """
        try:
            factor = factor_func(data, **kwargs)
            if isinstance(factor, pd.Series):
                factor.name = factor_name
                return factor
            else:
                raise ValueError("因子函数必须返回pandas.Series")
        except Exception as e:
            print(f"构建自定义因子 {factor_name} 失败: {e}")
            return pd.Series(dtype=float)
