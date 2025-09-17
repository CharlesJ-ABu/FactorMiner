"""
透明因子保存系统 v3.0
完全透明的因子计算逻辑存储

支持的因子类型：
- function: 函数类型因子，保存在 functions/ 目录
- ml_model: 机器学习模型因子，保存在 models/ 目录

目录结构：
- definitions/  - 因子定义 (JSON格式)
- functions/    - 因子函数代码 (Python文件)
- models/       - 机器学习模型文件 (.pkl文件，仅用于需要量化模型的因子)
- temp/         - 临时缓存
"""

import json
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import importlib.util
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FactorDefinition:
    """完整的因子定义 - 包含所有计算信息"""
    factor_id: str              # 唯一标识符
    name: str                   # 因子名称  
    description: str            # 因子描述
    category: str               # 因子类别
    subcategory: str = ""       # 子类别
    
    # 计算信息 - 核心扩展
    computation_type: str = "function"  # function, ml_model
    computation_data: Dict = None      # 具体的计算数据
    
    parameters: Dict = None     # 默认参数
    dependencies: List = None   # 依赖的其他因子/数据
    output_type: str = "series" # 输出类型
    metadata: Dict = None       # 其他元数据
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}
        if self.dependencies is None:
            self.dependencies = []
        if self.metadata is None:
            self.metadata = {}
        if self.computation_data is None:
            self.computation_data = {}
            
        # 自动生成校验和
        self.metadata['checksum'] = self._calculate_checksum()
        self.metadata['created_at'] = datetime.now().isoformat()
    
    def _calculate_checksum(self) -> str:
        """计算因子定义的校验和"""
        content = f"{self.factor_id}_{self.name}_{str(self.computation_data)}"
        return hashlib.md5(content.encode()).hexdigest()[:8]
    
    def to_dict(self) -> Dict:
        """转换为字典，用于JSON序列化"""
        return asdict(self)


class TransparentFactorStorage:
    """
    完全透明的因子存储管理器
    
    透明因子保存机制只支持两种因子类型：
    1. function - 函数类型因子：将因子计算逻辑保存为Python函数文件
    2. ml_model - 机器学习模型因子：将训练好的模型保存为.pkl文件
    
    目录结构：
    - definitions/  - 因子定义 (JSON格式)
    - functions/    - 因子函数代码 (Python文件)  
    - models/       - 机器学习模型文件 (.pkl文件，仅用于需要量化模型的因子)
    - temp/         - 临时缓存
    """
    
    def __init__(self, storage_dir: str = None):
        if storage_dir is None:
            # V4 重新组织：使用新的文件夹结构
            storage_dir = Path(__file__).parent.parent.parent / "factorlib"
        
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # 新的目录结构
        # technicals/ - 传统技术指标
        self.technicals_definitions_dir = self.storage_dir / "technicals" / "definitions"
        self.technicals_functions_dir = self.storage_dir / "technicals" / "functions"
        self.technicals_evaluations_dir = self.storage_dir / "technicals" / "evaluations"
        
        # minactors/ - 挖掘因子
        self.minactors_definitions_dir = self.storage_dir / "minactors" / "definitions"
        self.minactors_models_dir = self.storage_dir / "minactors" / "models"
        self.minactors_evaluations_dir = self.storage_dir / "minactors" / "evaluations"
        self.minactors_mining_history_dir = self.storage_dir / "minactors" / "mining_history"
        
        # 其他目录
        self.temp_dir = self.storage_dir / "temp"
        
        # 创建所有目录
        for dir_path in [
            self.technicals_definitions_dir, self.technicals_functions_dir, self.technicals_evaluations_dir,
            self.minactors_definitions_dir, self.minactors_models_dir, self.minactors_evaluations_dir, 
            self.minactors_mining_history_dir, self.temp_dir
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def save_function_factor(self, factor_id: str, name: str, 
                            function_code: str, entry_point: str = "calculate",
                            description: str = "", category: str = "custom",
                            parameters: Dict = None, imports: List[str] = None) -> bool:
        """
        保存函数类因子（已废弃，请使用 save_technical_factor）
        """
        return self.save_technical_factor(
            factor_id=factor_id,
            name=name,
            function_code=function_code,
            entry_point=entry_point,
            description=description,
            category=category,
            parameters=parameters,
            imports=imports
        )
    
    def compute_factor(self, factor_id: str, data: pd.DataFrame, **kwargs) -> Optional[pd.Series]:
        """动态计算因子"""
        factor_def = self.load_factor_definition(factor_id)
        if not factor_def:
            raise ValueError(f"因子不存在: {factor_id}")
        
        # 合并参数
        params = factor_def.parameters.copy()
        params.update(kwargs)
        
        try:
            if factor_def.computation_type == "function":
                return self._compute_function_factor(factor_def, data, params)
            elif factor_def.computation_type == "ml_model":
                return self._compute_ml_model_factor(factor_def, data, params)
            else:
                raise ValueError(f"不支持的计算类型: {factor_def.computation_type}。透明因子保存机制只支持 function 和 ml_model 类型")
                
        except Exception as e:
            logger.error(f"计算因子失败 {factor_id}: {e}")
            return None
    
    def _compute_function_factor(self, factor_def: FactorDefinition,
                                data: pd.DataFrame, params: Dict) -> pd.Series:
        """计算函数类因子"""
        comp_data = factor_def.computation_data
        func_file = self.storage_dir / comp_data["function_file"]
        entry_point = comp_data["entry_point"]
        
        # 动态导入模块
        spec = importlib.util.spec_from_file_location(
            f"factor_{factor_def.factor_id}", func_file
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # 获取入口函数
        if hasattr(module, entry_point):
            func = getattr(module, entry_point)
            return func(data, **params)
        else:
            raise ValueError(f"函数中未找到入口点: {entry_point}")
    

    def _compute_ml_model_factor(self, factor_def: FactorDefinition,
                                 data: pd.DataFrame, params: Dict) -> pd.Series:
        """基于已训练的.pkl模型进行推理的因子计算"""
        import pickle
        from .feature_pipeline import build_ml_features
        import pandas as pd  # 确保本地作用域有pd

        comp_data = factor_def.computation_data
        artifact_relpath = comp_data.get("artifact_path")
        if not artifact_relpath:
            raise ValueError("ml_model 定义缺少 artifact_path")

        artifact_file = self.storage_dir / artifact_relpath
        if not artifact_file.exists():
            # 兼容：若提供的相对路径不在factorlib下，尝试models目录
            candidate = self.models_dir / Path(artifact_relpath).name
            if candidate.exists():
                artifact_file = candidate
            else:
                raise FileNotFoundError(f"找不到模型文件: {artifact_file}")

        with open(artifact_file, "rb") as f:
            artifact = pickle.load(f)

        model = artifact.get("model")
        feature_columns = artifact.get("feature_columns") or []
        scaler = artifact.get("scaler")

        # 构建与训练一致的特征
        features = build_ml_features(data)

        # 对齐所需列
        missing = [c for c in feature_columns if c not in features.columns]
        if missing:
            # 对缺失列补NaN，保持列齐全
            for c in missing:
                features[c] = np.nan
        X = features[feature_columns]

        # 清洗与标准化
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(method='ffill').fillna(method='bfill')
        if scaler is not None:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X.values

        # 预测
        y_pred = model.predict(X_scaled)
        return pd.Series(y_pred, index=data.index)

    def save_ml_model_factor(self, factor_id: str, name: str,
                             artifact_filename: str,
                             description: str = "",
                             category: str = "ml",
                             parameters: Dict = None,
                             feature_set: str = "basic_v1") -> bool:
        """
        保存基于已训练模型的因子定义（ml_model），artifact 文件应放置在 factorlib/models/ 下
        """
        try:
            # 仅保存定义，不复制artifact
            artifact_rel = str(Path("models") / Path(artifact_filename).name)
            factor_def = FactorDefinition(
                factor_id=factor_id,
                name=name,
                description=description,
                category=category,
                computation_type="ml_model",
                computation_data={
                    "artifact_path": artifact_rel,
                    "feature_set": feature_set
                },
                parameters=parameters or {}
            )
            return self._save_factor_definition(factor_def)
        except Exception as e:
            logger.error(f"保存ML模型因子失败: {e}")
            return False
    
    def save_ml_factor(self, factor_id: str, name: str, algorithm_name: str,
                      description: str = "", category: str = "ml",
                      model_file: str = "", performance_metrics: Dict = None) -> bool:
        """
        保存ML因子定义（简化版）
        """
        try:
            factor_def = FactorDefinition(
                factor_id=factor_id,
                name=name,
                description=description,
                category=category,
                computation_type="ml_model",
                computation_data={
                    "algorithm_name": algorithm_name,
                    "model_file": model_file,
                    "performance_metrics": performance_metrics or {}
                },
                parameters={}
            )
            return self._save_factor_definition(factor_def)
        except Exception as e:
            logger.error(f"保存ML因子失败: {e}")
            return False
    
    def load_factor_definition(self, factor_id: str) -> Optional[FactorDefinition]:
        """加载因子定义"""
        try:
            # 先在minactors中查找
            def_file = self.minactors_definitions_dir / f"{factor_id}.json"
            if not def_file.exists():
                # 再在technicals中查找
                def_file = self.technicals_definitions_dir / f"{factor_id}.json"
                if not def_file.exists():
                    return None
            
            with open(def_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return FactorDefinition(**data)
            
        except Exception as e:
            logger.error(f"加载因子定义失败: {e}")
            return None
    
    def list_factors(self) -> List[str]:
        """列出所有因子ID"""
        try:
            # 合并两个目录的因子
            technicals_files = self.technicals_definitions_dir.glob("*.json")
            minactors_files = self.minactors_definitions_dir.glob("*.json")
            
            factor_ids = []
            factor_ids.extend([f.stem for f in technicals_files])
            factor_ids.extend([f.stem for f in minactors_files])
            
            return factor_ids
        except Exception as e:
            logger.error(f"列出因子失败: {e}")
            return []
    
    def get_factors_by_category(self, category: str) -> List[str]:
        """按分类获取因子"""
        factors = []
        for factor_id in self.list_factors():
            factor_def = self.load_factor_definition(factor_id)
            if factor_def and factor_def.category == category:
                factors.append(factor_id)
        return factors
    
    def delete_factor(self, factor_id: str) -> bool:
        """删除因子"""
        try:
            # 删除定义文件（先尝试minactors，再尝试technicals）
            def_file = self.minactors_definitions_dir / f"{factor_id}.json"
            if not def_file.exists():
                def_file = self.technicals_definitions_dir / f"{factor_id}.json"
            if def_file.exists():
                def_file.unlink()
            
            # 删除相关文件
            function_file = self.technicals_functions_dir / f"{factor_id}.py"
            if function_file.exists():
                function_file.unlink()
            
            return True
            
        except Exception as e:
            logger.error(f"删除因子失败: {e}")
            return False
    
    # 删除重复的save_evaluation_record方法，直接复用因子评估网页的存储方法
    
    def _save_factor_definition(self, factor_def: FactorDefinition) -> bool:
        """保存因子定义到JSON"""
        try:
            # 根据因子类型选择目录
            if factor_def.computation_type == "ml_model" or factor_def.category == "ml":
                def_file = self.minactors_definitions_dir / f"{factor_def.factor_id}.json"
            else:
                def_file = self.technicals_definitions_dir / f"{factor_def.factor_id}.json"
            
            with open(def_file, 'w', encoding='utf-8') as f:
                json.dump(factor_def.to_dict(), f, ensure_ascii=False, indent=2)
            
            logger.info(f"因子定义已保存: {factor_def.factor_id}")
            return True
            
        except Exception as e:
            logger.error(f"保存因子定义失败: {e}")
            return False

    # ==================== 新的简洁API接口 ====================
    
    def save_technical_factor(self, factor_id: str, name: str, function_code: str, 
                            description: str = "", category: str = "technical",
                            entry_point: str = "calculate", imports: List[str] = None) -> bool:
        """
        保存技术指标因子到 technicals/ 目录
        
        Args:
            factor_id: 因子唯一标识
            name: 因子名称
            function_code: Python函数代码
            description: 因子描述
            category: 因子分类
            entry_point: 入口函数名
            imports: 导入语句列表
            
        Returns:
            bool: 保存是否成功
        """
        try:
            # 保存函数代码
            func_file = self.technicals_functions_dir / f"{factor_id}.py"
            with open(func_file, 'w', encoding='utf-8') as f:
                if imports:
                    for imp in imports:
                        f.write(f"{imp}\n")
                    f.write("\n")
                f.write(function_code)
            
            # 保存因子定义
            factor_def = FactorDefinition(
                factor_id=factor_id,
                name=name,
                description=description,
                category=category,
                computation_type="function",
                computation_data={
                    "function_file": str(func_file.relative_to(self.storage_dir)),
                    "function_code": function_code,
                    "entry_point": entry_point,
                    "imports": imports or []
                },
                parameters={}
            )
            
            def_file = self.technicals_definitions_dir / f"{factor_id}.json"
            with open(def_file, 'w', encoding='utf-8') as f:
                json.dump(factor_def.to_dict(), f, ensure_ascii=False, indent=2)
            
            logger.info(f"技术指标因子已保存: {factor_id}")
            return True
            
        except Exception as e:
            logger.error(f"保存技术指标因子失败: {e}")
            return False
    
    def save_minactor_factor(self, factor_id: str, name: str, algorithm_name: str,
                           model_file: str = "", description: str = "", 
                           category: str = "ml", performance_metrics: Dict = None) -> bool:
        """
        保存挖掘因子到 minactors/ 目录
        
        Args:
            factor_id: 因子唯一标识
            name: 因子名称
            algorithm_name: 算法名称
            model_file: 模型文件名
            description: 因子描述
            category: 因子分类
            performance_metrics: 性能指标
            
        Returns:
            bool: 保存是否成功
        """
        try:
            # 保存因子定义
            factor_def = FactorDefinition(
                factor_id=factor_id,
                name=name,
                description=description,
                category=category,
                computation_type="ml_model",
                computation_data={
                    "algorithm_name": algorithm_name,
                    "model_file": model_file,
                    "performance_metrics": performance_metrics or {}
                },
                parameters={}
            )
            
            def_file = self.minactors_definitions_dir / f"{factor_id}.json"
            with open(def_file, 'w', encoding='utf-8') as f:
                json.dump(factor_def.to_dict(), f, ensure_ascii=False, indent=2)
            
            logger.info(f"挖掘因子已保存: {factor_id}")
            return True
            
        except Exception as e:
            logger.error(f"保存挖掘因子失败: {e}")
            return False
    
    def save_model(self, factor_id: str, model_data: bytes, model_type: str = "pkl") -> bool:
        """
        保存模型文件到 minactors/models/ 目录
        
        Args:
            factor_id: 因子标识
            model_data: 模型数据（字节）
            model_type: 模型类型（pkl, joblib等）
            
        Returns:
            bool: 保存是否成功
        """
        try:
            model_file = self.minactors_models_dir / f"{factor_id}.{model_type}"
            with open(model_file, 'wb') as f:
                f.write(model_data)
            
            logger.info(f"模型文件已保存: {factor_id}.{model_type}")
            return True
            
        except Exception as e:
            logger.error(f"保存模型文件失败: {e}")
            return False
    
    def load_model(self, factor_id: str, model_type: str = "pkl") -> Optional[bytes]:
        """
        加载模型文件从 minactors/models/ 目录
        
        Args:
            factor_id: 因子标识
            model_type: 模型类型（pkl, joblib等）
            
        Returns:
            bytes: 模型数据，如果文件不存在则返回None
        """
        try:
            model_file = self.minactors_models_dir / f"{factor_id}.{model_type}"
            if not model_file.exists():
                return None
            
            with open(model_file, 'rb') as f:
                model_data = f.read()
            
            logger.info(f"模型文件已加载: {factor_id}.{model_type}")
            return model_data
            
        except Exception as e:
            logger.error(f"加载模型文件失败: {e}")
            return None
    
    def save_evaluation(self, factor_id: str, evaluation_data: Dict, 
                       source: str = "minactors") -> bool:
        """
        保存评估结果
        
        Args:
            factor_id: 因子标识
            evaluation_data: 评估数据
            source: 来源（technicals 或 minactors）
            
        Returns:
            bool: 保存是否成功
        """
        try:
            if source == "technicals":
                eval_dir = self.technicals_evaluations_dir
            else:
                eval_dir = self.minactors_evaluations_dir
            
            eval_file = eval_dir / f"{factor_id}.json"
            
            # 加载现有评估数据
            existing_data = {}
            if eval_file.exists():
                with open(eval_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            
            # 添加新评估
            evaluations = existing_data.get('evaluations', [])
            evaluations.append({
                'evaluated_at': datetime.now().isoformat(),
                'results': evaluation_data
            })
            existing_data['evaluations'] = evaluations
            
            # 保存
            with open(eval_file, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"评估结果已保存: {factor_id}")
            return True
            
        except Exception as e:
            logger.error(f"保存评估结果失败: {e}")
            return False
    
    def save_mining_history(self, session_id: str, session_data: Dict) -> bool:
        """
        保存挖掘历史到 minactors/mining_history/ 目录
        
        Args:
            session_id: 会话ID
            session_data: 会话数据
            
        Returns:
            bool: 保存是否成功
        """
        try:
            # 保存到mining_sessions.json
            sessions_file = self.minactors_mining_history_dir / "mining_sessions.json"
            
            # 加载现有会话
            sessions = {}
            if sessions_file.exists():
                with open(sessions_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        sessions = json.loads(content)
            
            # 添加新会话
            sessions[session_id] = session_data
            
            # 保存
            with open(sessions_file, 'w', encoding='utf-8') as f:
                json.dump(sessions, f, ensure_ascii=False, indent=2)
            
            # 保存详细结果
            result_file = self.minactors_mining_history_dir / f"mining_results_{session_id}.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"挖掘历史已保存: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"保存挖掘历史失败: {e}")
            return False


# 全局实例
_global_storage = None

def get_global_storage() -> TransparentFactorStorage:
    """获取全局存储实例"""
    global _global_storage
    if _global_storage is None:
        _global_storage = TransparentFactorStorage()
    return _global_storage
