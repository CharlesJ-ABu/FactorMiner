"""
JSON驱动的因子存储系统 v3.0
完全透明的因子计算逻辑存储
"""

import json
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import importlib.util
import sys
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
    computation_type: str = "formula"  # formula, function, model, pipeline
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
    """完全透明的因子存储管理器"""
    
    def __init__(self, storage_dir: str = None):
        if storage_dir is None:
            # V3 扁平化：直接使用 factorlib 根目录
            storage_dir = Path(__file__).parent.parent.parent / "factorlib"
        
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # 目录结构（扁平化，位于 factorlib/ 下）
        self.definitions_dir = self.storage_dir / "definitions"   # 因子定义
        self.formulas_dir = self.storage_dir / "formulas"        # 公式
        self.functions_dir = self.storage_dir / "functions"      # 函数代码
        self.pipelines_dir = self.storage_dir / "pipelines"      # ML流水线
        self.temp_dir = self.storage_dir / "temp"                # 临时缓存
        self.models_dir = self.storage_dir / "models"            # 训练好的ML模型artifact (.pkl)
        
        # 创建目录
        for dir_path in [self.definitions_dir, self.formulas_dir, 
                        self.functions_dir, self.pipelines_dir, self.temp_dir, self.models_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def save_formula_factor(self, factor_id: str, name: str, formula: str,
                           description: str = "", category: str = "custom",
                           parameters: Dict = None) -> bool:
        """
        保存公式类因子
        公式直接以文本形式存储，支持pandas表达式
        """
        try:
            # 保存公式文件
            formula_file = self.formulas_dir / f"{factor_id}.txt"
            with open(formula_file, 'w', encoding='utf-8') as f:
                f.write(formula)
            
            factor_def = FactorDefinition(
                factor_id=factor_id,
                name=name,
                description=description,
                category=category,
                computation_type="formula",
                computation_data={
                    "formula_file": str(formula_file.relative_to(self.storage_dir)),
                    "formula": formula,
                    "language": "pandas"
                },
                parameters=parameters or {}
            )
            
            return self._save_factor_definition(factor_def)
            
        except Exception as e:
            logger.error(f"保存公式因子失败: {e}")
            return False
    
    def save_function_factor(self, factor_id: str, name: str, 
                            function_code: str, entry_point: str = "calculate",
                            description: str = "", category: str = "custom",
                            parameters: Dict = None, imports: List[str] = None) -> bool:
        """
        保存函数类因子
        函数代码以Python文件形式存储
        """
        try:
            # 保存函数代码
            func_file = self.functions_dir / f"{factor_id}.py"
            with open(func_file, 'w', encoding='utf-8') as f:
                # 写入导入语句
                if imports:
                    for imp in imports:
                        f.write(f"{imp}\n")
                    f.write("\n")
                # 写入函数代码
                f.write(function_code)
            
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
                parameters=parameters or {}
            )
            
            return self._save_factor_definition(factor_def)
            
        except Exception as e:
            logger.error(f"保存函数因子失败: {e}")
            return False
    
    def save_pipeline_factor(self, factor_id: str, name: str,
                            pipeline_steps: List[Dict], 
                            description: str = "", category: str = "ml",
                            parameters: Dict = None) -> bool:
        """
        保存ML流水线因子
        每个步骤都是一个独立的操作，支持特征工程、模型等
        
        Args:
            pipeline_steps: [
                {
                    "type": "feature_engineering",
                    "code": "features = pd.DataFrame(index=data.index)...",
                    "outputs": ["price_momentum", "volume_ratio"]
                },
                {
                    "type": "model",
                    "algorithm": "LinearRegression",
                    "parameters": {"fit_intercept": true},
                    "features": ["price_momentum", "volume_ratio"],
                    "target": "next_return"
                },
                {
                    "type": "postprocess",
                    "code": "signals = predictions.copy()..."
                }
            ]
        """
        try:
            # 保存流水线定义
            pipeline_file = self.pipelines_dir / f"{factor_id}.json"
            with open(pipeline_file, 'w', encoding='utf-8') as f:
                json.dump(pipeline_steps, f, indent=2, ensure_ascii=False)
            
            factor_def = FactorDefinition(
                factor_id=factor_id,
                name=name,
                description=description,
                category=category,
                computation_type="pipeline",
                computation_data={
                    "pipeline_file": str(pipeline_file.relative_to(self.storage_dir)),
                    "pipeline_steps": pipeline_steps
                },
                parameters=parameters or {}
            )
            
            return self._save_factor_definition(factor_def)
            
        except Exception as e:
            logger.error(f"保存流水线因子失败: {e}")
            return False
    
    def compute_factor(self, factor_id: str, data: pd.DataFrame, **kwargs) -> Optional[pd.Series]:
        """动态计算因子"""
        factor_def = self.load_factor_definition(factor_id)
        if not factor_def:
            raise ValueError(f"因子不存在: {factor_id}")
        
        # 合并参数
        params = factor_def.parameters.copy()
        params.update(kwargs)
        
        try:
            if factor_def.computation_type == "formula":
                return self._compute_formula_factor(factor_def, data, params)
            elif factor_def.computation_type == "function":
                return self._compute_function_factor(factor_def, data, params)
            elif factor_def.computation_type == "pipeline":
                return self._compute_pipeline_factor(factor_def, data, params)
            elif factor_def.computation_type == "ml_model":
                return self._compute_ml_model_factor(factor_def, data, params)
            else:
                raise ValueError(f"不支持的计算类型: {factor_def.computation_type}")
                
        except Exception as e:
            logger.error(f"计算因子失败 {factor_id}: {e}")
            return None
    
    def _compute_formula_factor(self, factor_def: FactorDefinition, 
                               data: pd.DataFrame, params: Dict) -> pd.Series:
        """计算公式类因子"""
        formula = factor_def.computation_data["formula"]
        
        # 创建计算环境
        env = {
            'data': data,
            'pd': pd,
            'np': np,
            **params
        }
        
        # 添加常用的列别名
        if 'close' in data.columns:
            env['close'] = data['close']
        if 'open' in data.columns:
            env['open'] = data['open']
        if 'high' in data.columns:
            env['high'] = data['high']
        if 'low' in data.columns:
            env['low'] = data['low']
        if 'volume' in data.columns:
            env['volume'] = data['volume']
        
        # 执行公式
        result = eval(formula, env)
        
        if isinstance(result, pd.Series):
            return result
        elif isinstance(result, (int, float)):
            return pd.Series([result] * len(data), index=data.index)
        else:
            return pd.Series(result, index=data.index)
    
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
    
    def _compute_pipeline_factor(self, factor_def: FactorDefinition,
                                data: pd.DataFrame, params: Dict) -> pd.Series:
        """计算ML流水线因子"""
        pipeline_steps = factor_def.computation_data["pipeline_steps"]
        current_data = data.copy()
        
        for step in pipeline_steps:
            step_type = step["type"]
            
            if step_type == "feature_engineering":
                # 执行特征工程代码
                env = {
                    'data': current_data,
                    'pd': pd,
                    'np': np,
                    **params
                }
                exec(step["code"], env)
                features = env.get('features')
                if features is not None:
                    current_data = features
            
            elif step_type == "model":
                # 准备特征
                features = step["features"]
                X = current_data[features]
                
                # 加载和应用模型
                from sklearn.linear_model import LinearRegression  # 示例
                model = LinearRegression(**step.get("parameters", {}))
                
                if "target" in step:
                    # 训练模式
                    y = current_data[step["target"]]
                    model.fit(X, y)
                    predictions = model.predict(X)
                else:
                    # 预测模式
                    predictions = model.predict(X)
                
                current_data = pd.Series(predictions, index=data.index)
            
            elif step_type == "postprocess":
                # 执行后处理代码
                env = {
                    'predictions': current_data,
                    'data': data,
                    'pd': pd,
                    'np': np,
                    **params
                }
                exec(step["code"], env)
                signals = env.get('signals')
                if signals is not None:
                    current_data = signals
        
        return current_data

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
    
    def load_factor_definition(self, factor_id: str) -> Optional[FactorDefinition]:
        """加载因子定义"""
        try:
            def_file = self.definitions_dir / f"{factor_id}.json"
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
            factor_files = self.definitions_dir.glob("*.json")
            return [f.stem for f in factor_files]
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
            # 删除定义文件
            def_file = self.definitions_dir / f"{factor_id}.json"
            if def_file.exists():
                def_file.unlink()
            
            # 删除相关文件
            formula_file = self.formulas_dir / f"{factor_id}.txt"
            if formula_file.exists():
                formula_file.unlink()
            
            function_file = self.functions_dir / f"{factor_id}.py"
            if function_file.exists():
                function_file.unlink()
            
            pipeline_file = self.pipelines_dir / f"{factor_id}.json"
            if pipeline_file.exists():
                pipeline_file.unlink()
            
            return True
            
        except Exception as e:
            logger.error(f"删除因子失败: {e}")
            return False
    
    # 删除重复的save_evaluation_record方法，直接复用因子评估网页的存储方法
    
    def _save_factor_definition(self, factor_def: FactorDefinition) -> bool:
        """保存因子定义到JSON"""
        try:
            def_file = self.definitions_dir / f"{factor_def.factor_id}.json"
            with open(def_file, 'w', encoding='utf-8') as f:
                json.dump(factor_def.to_dict(), f, ensure_ascii=False, indent=2)
            
            logger.info(f"因子定义已保存: {factor_def.factor_id}")
            return True
            
        except Exception as e:
            logger.error(f"保存因子定义失败: {e}")
            return False


# 全局实例
_global_storage = None

def get_global_storage() -> TransparentFactorStorage:
    """获取全局存储实例"""
    global _global_storage
    if _global_storage is None:
        _global_storage = TransparentFactorStorage()
    return _global_storage
