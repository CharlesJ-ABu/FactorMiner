"""
统一因子构建器 V4.0
纯调度器，所有算法从 user_algo 目录动态加载

版本历史：
- V1.0: 基础因子构建功能
- V2.0: 添加多种因子类型支持
- V3.0: 整合所有因子构建功能，与透明因子存储系统完全兼容
- V4.0: 重构为纯调度器，支持用户自定义算法
  * 将具体算法实现提取到 user_algo 目录
  * 支持动态加载和热插拔算法
  * 保持向后兼容性
  * 代码从1000+行精简到200行

主要变更：
- 移除所有 _build_*_factors 方法
- 添加动态算法加载机制
- 支持用户自定义算法接口
- 保持原有API接口不变
"""

import pandas as pd
from typing import Dict, List, Optional, Callable
from pathlib import Path
import importlib.util
import warnings

from .factor_storage import TransparentFactorStorage
from .factor_engine import FactorEngine

warnings.filterwarnings('ignore')


class FactorBuilder:
    """
    统一因子构建器 V4.0
    纯调度器，所有算法从 user_algo 目录动态加载
    
    演进历程：
    - V1.0: 基础技术指标因子构建
    - V2.0: 添加统计、高级、ML等因子类型
    - V3.0: 整合所有因子类型，支持透明存储
    - V4.0: 重构为纯调度器，支持用户自定义算法
    
    核心功能：
    - 动态扫描和加载 user_algo 目录中的算法
    - 支持算法缓存和热插拔
    - 统一的算法执行接口
    - 自动因子存储和管理
    
    使用方式：
    ```python
    builder = FactorBuilder()
    factors = builder.build_all_factors(data, selected_algorithms=['ml_momentum'])
    ```
    """
    
    def __init__(self, storage: TransparentFactorStorage = None):
        """
        初始化因子构建器
        
        Args:
            storage: 因子存储实例，如果为None则创建新实例
        """
        self.storage = storage or TransparentFactorStorage()
        self.engine = FactorEngine(self.storage)
        self._algorithm_cache = {}
        self._user_algo_dir = Path(__file__).parent.parent.parent / "user_algo"
        
        print("🚀 因子构建器 V4.0 已初始化")
        print(f"📁 用户算法目录: {self._user_algo_dir}")
    
    def build_all_factors(self, data: pd.DataFrame, selected_algorithms: List[str], 
                         save_to_storage: bool = True, **kwargs) -> Dict:
        """
        构建所有因子（主要接口）
        
        Args:
            data: 市场数据
            selected_algorithms: 选中的算法列表
            save_to_storage: 是否保存到存储系统
            **kwargs: 其他参数
            
        Returns:
            Dict: 包含成功状态、因子数据、统计信息等
        """
        print(f"🚀 开始构建因子，算法: {', '.join(selected_algorithms)}")
        
        all_factors = {}
        built_factors = {}
        
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
        if save_to_storage and built_factors:
            self._save_factors_to_storage(built_factors, data, **kwargs)
        
        return {
            'success': True,
            'factors': all_factors,
            'factors_df': factors_df,
            'total_factors': len(factors_df.columns),
            'algorithms_used': selected_algorithms
        }
    
    def _execute_algorithm(self, algo_id: str, data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """
        执行单个算法
        
        Args:
            algo_id: 算法ID
            data: 市场数据
            **kwargs: 算法参数
            
        Returns:
            Dict[str, pd.Series]: 因子名称到因子值的映射
        """
        # 加载算法模块
        algorithm_module = self._load_algorithm_module(algo_id)
        if not algorithm_module:
            raise ValueError(f"无法加载算法: {algo_id}")
        
        # 调用算法的 calculate_factors 函数
        if hasattr(algorithm_module, 'calculate_factors'):
            return algorithm_module.calculate_factors(data, **kwargs)
        else:
            raise ValueError(f"算法 {algo_id} 缺少 calculate_factors 函数")
    
    def _load_algorithm_module(self, algo_id: str):
        """动态加载算法模块"""
        if algo_id in self._algorithm_cache:
            return self._algorithm_cache[algo_id]
        
        try:
            # 查找算法文件
            algo_file = self._user_algo_dir / f"{algo_id}.py"
            if not algo_file.exists():
                print(f"❌ 算法文件不存在: {algo_file}")
                return None
            
            # 动态导入
            spec = importlib.util.spec_from_file_location(algo_id, algo_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # 缓存模块
            self._algorithm_cache[algo_id] = module
            print(f"✅ 算法模块已加载: {algo_id}")
            return module
            
        except Exception as e:
            print(f"❌ 加载算法模块失败 {algo_id}: {e}")
            return None
    
    def scan_all_algorithms(self) -> List[Dict]:
        """扫描所有可用算法"""
        algorithms = []
        
        if not self._user_algo_dir.exists():
            print(f"❌ 用户算法目录不存在: {self._user_algo_dir}")
            return algorithms
        
        # 扫描所有Python文件
        for algo_file in self._user_algo_dir.glob("*.py"):
            if algo_file.name.startswith("__"):
                continue
                
            algo_id = algo_file.stem
            try:
                # 加载模块获取信息
                module = self._load_algorithm_module(algo_id)
                if module and hasattr(module, 'ALGORITHM_INFO'):
                    info = module.ALGORITHM_INFO
                    algorithms.append({
                        'id': algo_id,
                        'name': info.get('name', algo_id),
                        'description': info.get('description', ''),
                        'category': info.get('category', 'other'),
                        'version': info.get('version', '1.0.0'),
                        'file_path': str(algo_file)
                    })
                else:
                    # 默认信息
                    algorithms.append({
                        'id': algo_id,
                        'name': algo_id,
                        'description': '用户自定义算法',
                        'category': 'custom',
                        'version': '1.0.0',
                        'file_path': str(algo_file)
                    })
                    
            except Exception as e:
                print(f"❌ 扫描算法失败 {algo_id}: {e}")
                continue
        
        print(f"📋 扫描完成，找到 {len(algorithms)} 个算法")
        return algorithms
    
    def get_algorithm_info(self, algo_id: str) -> Optional[Dict]:
        """获取算法信息"""
        algorithms = self.scan_all_algorithms()
        for algo in algorithms:
            if algo['id'] == algo_id:
                return algo
        return None
    
    def _save_factors_to_storage(self, built_factors: Dict, data: pd.DataFrame, **kwargs):
        """将构建的因子保存到存储系统（简化版：只保存定义和模型）"""
        print("💾 开始保存因子到存储系统...")
        
        for algo_id, factors in built_factors.items():
            try:
                # 获取算法信息
                algo_info = self.get_algorithm_info(algo_id)
                if not algo_info:
                    print(f"❌ 无法获取算法 {algo_id} 的信息")
                    continue
                
                print(f"📖 处理算法: {algo_id}")
                
                # 为每个因子保存定义
                for factor_name, factor_series in factors.items():
                    try:
                        # 创建因子定义
                        factor_id = f"{algo_id}_{factor_name}"
                        
                        # 保存因子定义（包含算法信息）
                        success = self.storage.save_minactor_factor(
                            factor_id=factor_id,
                            name=factor_name,
                            algorithm_name=algo_id,
                            description=f"由 {algo_id} 算法生成的 {factor_name} 因子",
                            category=algo_id.split('_')[0] if '_' in algo_id else 'ml',
                            model_file=f"{algo_id}_{factor_name}.pkl",  # 假设模型文件
                            performance_metrics={}  # 稍后填充
                        )
                        
                        if success:
                            print(f"✅ 因子定义 {factor_id} 保存成功")
                        else:
                            print(f"❌ 因子定义 {factor_id} 保存失败")
                        
                    except Exception as e:
                        print(f"❌ 保存因子 {factor_name} 时出错: {e}")
                        continue
                    
            except Exception as e:
                print(f"❌ 处理算法 {algo_id} 时出错: {e}")
                continue
        
        print("💾 因子定义保存完成")
    
    def list_algorithms(self) -> List[Dict]:
        """列出所有算法"""
        return self.scan_all_algorithms()
    
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