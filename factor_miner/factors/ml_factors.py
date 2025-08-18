"""
机器学习因子构建模块
使用机器学习方法构建和优化因子
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import pickle
import threading
import time
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import warnings
warnings.filterwarnings('ignore')

# 导入tqdm，并配置为适合后台运行
try:
    from tqdm import tqdm
    # 配置tqdm为适合后台运行的模式
    import os
    if os.environ.get('TQDM_DISABLE') != '1':
        # 检查tqdm版本，使用兼容的配置方式
        try:
            # 新版本tqdm支持set_defaults
            tqdm.set_defaults(
                position=0,  # 固定位置
                leave=True,  # 保留进度条
                ncols=80,    # 固定宽度
                ascii=True,  # 使用ASCII字符，避免特殊字符问题
                dynamic_ncols=False  # 固定列数
            )
        except AttributeError:
            # 旧版本tqdm不支持set_defaults，使用默认配置
            print("使用默认tqdm配置（旧版本）")
except ImportError:
    # 如果没有tqdm，创建一个简单的进度显示
    class SimpleProgress:
        def __init__(self, iterable=None, desc="", total=None, unit="", **kwargs):
            self.iterable = iterable
            self.desc = desc
            self.total = total or (len(iterable) if iterable else 0)
            self.unit = unit
            self.current = 0
            if desc:
                print(f"{desc}: 开始处理...")
        
        def __iter__(self):
            if self.iterable:
                for item in self.iterable:
                    yield item
                    self.current += 1
                    if self.current % max(1, self.total // 10) == 0:  # 每10%显示一次
                        print(f"{self.desc}: {self.current}/{self.total} ({self.current/self.total*100:.1f}%)")
        
        def write(self, text):
            print(text)
        
        def __enter__(self):
            return self
        
        def __exit__(self, *args):
            if self.desc:
                print(f"{self.desc}: 完成 ({self.current}/{self.total})")
    
    tqdm = SimpleProgress


class ProgressSimulator:
    """模拟连续进度更新的类，用于在模型训练期间提供tqdm式的进度效果"""
    
    def __init__(self, progress_callback, stage, start_progress, end_progress, duration, message_template):
        self.progress_callback = progress_callback
        self.stage = stage
        self.start_progress = start_progress
        self.end_progress = end_progress
        self.duration = duration
        self.message_template = message_template
        self.current_progress = start_progress
        self.timer = None
        self.stop_flag = False
        
    def start(self):
        """开始模拟进度"""
        self.stop_flag = False
        # 立即发送第一个进度更新
        if self.progress_callback:
            message = self.message_template.format(progress=self.current_progress)
            print(f"📈 ProgressSimulator开始: {self.stage} -> {self.current_progress}% | {message}")
            self.progress_callback(stage=self.stage, progress=self.current_progress, message=message)
        self._update_progress()
        
    def stop(self):
        """停止模拟进度"""
        self.stop_flag = True
        if self.timer:
            self.timer.cancel()
        # 确保最终进度为结束值
        if self.progress_callback:
            self.progress_callback(stage=self.stage, progress=self.end_progress, message=self.message_template.format(progress=self.end_progress))
    
    def _update_progress(self):
        """内部进度更新方法"""
        if self.stop_flag:
            return
            
        if self.current_progress < self.end_progress:
            # 计算下一个进度点
            increment = max(1, (self.end_progress - self.start_progress) // 10)  # 每次增加较大步长，更明显
            self.current_progress = min(self.current_progress + increment, self.end_progress)
            
            if self.progress_callback:
                message = self.message_template.format(progress=self.current_progress)
                print(f"📈 ProgressSimulator更新: {self.stage} -> {self.current_progress}% | {message}")
                self.progress_callback(stage=self.stage, progress=self.current_progress, message=message)
            
            # 调度下一次更新 - 适中的更新频率，避免过于频繁
            interval = max(0.5, self.duration / max(1, (self.end_progress - self.start_progress)))
            self.timer = threading.Timer(interval, self._update_progress)
            self.timer.start()


class MLFactorBuilder:
    """
    机器学习因子构建器
    使用机器学习方法构建和优化因子
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化机器学习因子构建器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.scaler = StandardScaler()
        self.models = {}
        # 模型持久化目录：factorlib/models
        self.models_dir = (Path(__file__).parent.parent.parent / "factorlib" / "models")
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def _artifact_path(self, factor_id: str) -> Path:
        return self.models_dir / f"{factor_id}.pkl"

    def _save_model_artifact(
        self,
        factor_id: str,
        model,
        feature_columns: List[str],
        scaler: Optional[StandardScaler] = None
    ) -> None:
        try:
            artifact = {
                "model": model,
                "feature_columns": list(feature_columns),
                "scaler": scaler,
            }
            with open(self._artifact_path(factor_id), "wb") as f:
                pickle.dump(artifact, f)
        except Exception as _:
            print(f"❌ 保存模型文件失败 {factor_id}")
        
    def build_ensemble_factors(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        构建集成学习因子
        
        Args:
            data: 市场数据
            **kwargs: 其他参数
            
        Returns:
            集成学习因子DataFrame
        """
        factors = {}
        progress_callback = kwargs.get('progress_callback')
        
        # 准备特征（统一特征管道）
        if progress_callback:
            progress_callback(stage='ml', progress=5, message='集成模型训练: 准备特征数据...')
        try:
            from factor_miner.core.feature_pipeline import build_ml_features
            features = build_ml_features(data)
        except Exception:
            features = self._prepare_features(data)
        target = self._prepare_target(data)
        
        # 移除NaN值 - 确保索引对齐
        features_nan_mask = features.isna().any(axis=1)
        target_nan_mask = target.isna()
        
        # 重新索引以确保对齐
        common_index = features.index.intersection(target.index)
        features_nan_aligned = features_nan_mask.reindex(common_index, fill_value=True)
        target_nan_aligned = target_nan_mask.reindex(common_index, fill_value=True)
        
        valid_idx = ~(features_nan_aligned | target_nan_aligned)
        features_clean = features.reindex(common_index).loc[valid_idx]
        target_clean = target.reindex(common_index).loc[valid_idx]
        
        if len(features_clean) < 100:
            print("数据量不足，无法构建集成因子")
            return pd.DataFrame(factors, index=data.index)
        
        # 标准化特征
        if progress_callback:
            progress_callback(stage='ml', progress=15, message='集成模型训练: 标准化特征...')
        features_scaled = self.scaler.fit_transform(features_clean)
        
        # 构建多个模型
        models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=0.01)
        }
        
        # 训练模型并生成预测
        for model_idx, (name, model) in enumerate(tqdm(models.items(), desc="训练集成模型", unit="模型")):
            base_progress = 20 + (model_idx / len(models)) * 60  # 20-80%
            
            # 立即发送开始训练的进度更新
            if progress_callback:
                progress_callback(stage='ml', progress=int(base_progress), message=f'集成模型训练: 开始训练 {name} 模型...')
                # 立即发送一个稍高的进度，让用户看到变化
                progress_callback(stage='ml', progress=int(base_progress + 1), message=f'集成模型训练: {name} 初始化...')
            
            try:
                # 启动进度模拟器模拟模型训练过程
                simulator = None
                if progress_callback:
                    simulator = ProgressSimulator(
                        progress_callback=progress_callback,
                        stage='ml',
                        start_progress=int(base_progress),
                        end_progress=int(base_progress + 12),
                        duration=2.0,  # 2秒的模拟训练时间
                        message_template=f'集成模型训练: {name} 训练中... ' + '{progress}%'
                    )
                    simulator.start()
                
                model.fit(features_scaled, target_clean)
                
                # 停止进度模拟器
                if simulator:
                    simulator.stop()
                
                if progress_callback:
                    progress_callback(stage='ml', progress=int(base_progress + 14), message=f'集成模型训练: {name} 生成预测...')
                
                predictions = model.predict(features_scaled)
                
                # 创建因子序列
                factor_series = pd.Series(index=data.index, dtype=float)
                factor_series.loc[valid_idx] = predictions
                
                factor_id = f"ensemble_{name}"
                factors[factor_id] = factor_series
                self.models[name] = model

                if progress_callback:
                    progress_callback(stage='ml', progress=int(base_progress + 12), message=f'集成模型训练: {name} 保存模型...')

                # 持久化模型（用于后续通过core加载推理）
                self._save_model_artifact(
                    factor_id=factor_id,
                    model=model,
                    feature_columns=list(features_clean.columns),
                    scaler=self.scaler
                )
                
                tqdm.write(f"✓ 成功训练 {name} 模型")
                if progress_callback:
                    completion_progress = 20 + ((model_idx + 1) / len(models)) * 60
                    progress_callback(stage='ml', progress=int(completion_progress), message=f'集成模型训练: {name} 完成')
            
            except Exception:
                tqdm.write(f"✗ 训练 {name} 模型失败")
                continue
        
        if progress_callback:
            progress_callback(stage='ml', progress=85, message='集成模型训练: 保存模型完成')
        
        return pd.DataFrame(factors, index=data.index)

    def build_pca_factors(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        构建PCA因子
        
        Args:
            data: 市场数据
            **kwargs: 其他参数
            
        Returns:
            PCA因子DataFrame
        """
        factors = {}
        progress_callback = kwargs.get('progress_callback')
        
        # 准备特征
        if progress_callback:
            progress_callback(stage='ml', progress=5, message='PCA 降维: 准备特征数据...')
        print("准备PCA特征数据...")
        features = self._prepare_features(data)
        print(f"PCA特征数据形状: {features.shape}")
        
        if features.empty:
            print("❌ PCA特征数据为空，无法构建PCA因子")
            return pd.DataFrame(factors, index=data.index)
        
        # 移除NaN值
        valid_idx = ~features.isna().any(axis=1)
        features_clean = features.loc[valid_idx]
        
        print(f"PCA有效数据索引数量: {valid_idx.sum()}")
        
        if len(features_clean) < 50:
            print(f"❌ PCA数据量不足50条（只有{len(features_clean)}条），无法构建PCA因子")
            return pd.DataFrame(factors, index=data.index)
        
        # 标准化特征
        if progress_callback:
            progress_callback(stage='ml', progress=20, message='PCA 降维: 标准化特征数据...')
        print("标准化PCA特征数据...")
        features_scaled = self.scaler.fit_transform(features_clean)
        
        # 计算PCA
        n_components = kwargs.get('n_components', min(10, features_clean.shape[1]))
        print(f"PCA组件数: {n_components}")
        
        try:
            # 启动PCA进度模拟器
            simulator = None
            if progress_callback:
                simulator = ProgressSimulator(
                    progress_callback=progress_callback,
                    stage='ml',
                    start_progress=40,
                    end_progress=55,
                    duration=1.5,  # 1.5秒的模拟PCA时间
                    message_template='PCA 降维: 主成分分析进行中... {progress}%'
                )
                simulator.start()
            
            print("执行PCA降维...")
            pca = PCA(n_components=n_components)
            pca_components = pca.fit_transform(features_scaled)
            
            # 停止PCA进度模拟器
            if simulator:
                simulator.stop()
            
            print(f"PCA降维完成，形状: {pca_components.shape}")
            
            # 创建因子
            if progress_callback:
                progress_callback(stage='ml', progress=60, message='PCA 降维: 生成PCA因子...')
            for i in tqdm(range(n_components), desc="创建PCA因子", unit="因子"):
                if progress_callback and i % max(1, n_components // 5) == 0:
                    progress_pct = 60 + int((i / n_components) * 30)
                    progress_callback(stage='ml', progress=progress_pct, message=f'PCA 降维: 创建第 {i+1}/{n_components} 个组件')
                
                factor_series = pd.Series(index=data.index, dtype=float)
                factor_series.loc[valid_idx] = pca_components[:, i]
                factors[f'pca_component_{i+1}'] = factor_series
            
            # 保存解释方差比例
            explained_variance_ratio = pca.explained_variance_ratio_
            tqdm.write(f"PCA解释方差比例: {explained_variance_ratio[:5]}")
            print(f"前5个组件累计解释方差: {explained_variance_ratio[:5].sum():.3f}")
                
        except Exception as e:
            print(f"❌ PCA计算失败: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"✅ PCA因子构建完成，共 {len(factors)} 个因子")
        return pd.DataFrame(factors, index=data.index)
    
    def build_feature_selection_factors(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        构建特征选择因子
        
        Args:
            data: 市场数据
            **kwargs: 其他参数
            
        Returns:
            特征选择因子DataFrame
        """
        factors = {}
        progress_callback = kwargs.get('progress_callback')
        
        # 准备特征和目标
        if progress_callback:
            progress_callback(stage='ml', progress=5, message='特征选择: 准备特征数据...')
        features = self._prepare_features(data)
        target = self._prepare_target(data)
        
        # 移除NaN值 - 确保索引对齐
        features_nan_mask = features.isna().any(axis=1)
        target_nan_mask = target.isna()
        
        # 重新索引以确保对齐
        common_index = features.index.intersection(target.index)
        features_nan_aligned = features_nan_mask.reindex(common_index, fill_value=True)
        target_nan_aligned = target_nan_mask.reindex(common_index, fill_value=True)
        
        valid_idx = ~(features_nan_aligned | target_nan_aligned)
        features_clean = features.reindex(common_index).loc[valid_idx]
        target_clean = target.reindex(common_index).loc[valid_idx]
        
        if len(features_clean) < 50:
            print("数据量不足，无法构建特征选择因子")
            return pd.DataFrame(factors, index=data.index)
        
        # 特征选择方法
        selection_methods = {
            'f_regression': f_regression,
            'mutual_info': mutual_info_regression
        }
        
        k_best = kwargs.get('k_best', min(20, features_clean.shape[1]))
        
        for method_idx, (method_name, method_func) in enumerate(tqdm(selection_methods.items(), desc="特征选择", unit="方法")):
            try:
                if progress_callback:
                    base_progress = 10 + (method_idx / len(selection_methods)) * 70
                    progress_callback(stage='ml', progress=int(base_progress), message=f'特征选择: 执行 {method_name} 算法...')
                
                # 特征选择
                selector = SelectKBest(score_func=method_func, k=k_best)
                selector.fit(features_clean, target_clean)
                
                # 获取选中的特征
                selected_features = features_clean.iloc[:, selector.get_support()]
                
                # 计算选中特征的组合
                if len(selected_features.columns) > 0:
                    # 简单平均组合
                    factor_series = pd.Series(index=data.index, dtype=float)
                    factor_series.loc[valid_idx] = selected_features.mean(axis=1)
                    factors[f'feature_selection_{method_name}_mean'] = factor_series
                    
                    # 加权组合（基于特征重要性）
                    scores = selector.scores_[selector.get_support()]
                    weights = scores / scores.sum()
                    weighted_factor = (selected_features * weights).sum(axis=1)
                    
                    factor_series = pd.Series(index=data.index, dtype=float)
                    factor_series.loc[valid_idx] = weighted_factor
                    factors[f'feature_selection_{method_name}_weighted'] = factor_series
                
                tqdm.write(f"✓ 成功完成 {method_name} 特征选择")
                if progress_callback:
                    completion_progress = 10 + ((method_idx + 1) / len(selection_methods)) * 70
                    progress_callback(stage='ml', progress=int(completion_progress), message=f'特征选择: {method_name} 完成, 选择了 {len(selected_features.columns)} 个特征')
                
            except Exception:
                tqdm.write(f"✗ 特征选择 {method_name} 失败")
                continue
        
        return pd.DataFrame(factors, index=data.index)
    
    def build_rolling_ml_factors(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        构建滚动机器学习因子
        
        Args:
            data: 市场数据
            **kwargs: 其他参数
            
        Returns:
            滚动ML因子DataFrame
        """
        factors = {}
        progress_callback = kwargs.get('progress_callback')
        
        # 准备特征
        if progress_callback:
            progress_callback(stage='ml', progress=5, message='滚动ML: 准备时间序列特征...')
        features = self._prepare_features(data)
        target = self._prepare_target(data)
        
        # 滚动窗口参数
        window = kwargs.get('window', 252)  # 一年
        step = kwargs.get('step', 21)  # 一个月
        
        # 初始化因子序列
        factor_series = pd.Series(index=data.index, dtype=float)
        
        # 滚动训练和预测
        total_steps = len(range(window, len(data), step))
        for step_idx, i in enumerate(tqdm(range(window, len(data), step), desc="滚动ML训练", total=total_steps, unit="窗口")):
            # 更频繁的进度更新
            if progress_callback:
                progress_pct = 10 + int((step_idx / total_steps) * 80)
                progress_callback(stage='ml', progress=progress_pct, message=f'滚动ML: 窗口 {step_idx+1}/{total_steps}, 训练期 {i-window}~{i}')
                
                # 在窗口训练过程中添加中间进度更新
                if step_idx % max(1, total_steps // 40) == 0:  # 每2.5%更新一次
                    progress_callback(stage='ml', progress=progress_pct + 1, message=f'滚动ML: 处理数据窗口 {step_idx+1}/{total_steps}...')
            
            # 训练窗口
            train_features = features.iloc[i-window:i]
            train_target = target.iloc[i-window:i]
            
            # 移除NaN值 - 确保索引对齐
            features_nan_mask = train_features.isna().any(axis=1)
            target_nan_mask = train_target.isna()
            
            # 重新索引以确保对齐
            common_index = train_features.index.intersection(train_target.index)
            features_nan_aligned = features_nan_mask.reindex(common_index, fill_value=True)
            target_nan_aligned = target_nan_mask.reindex(common_index, fill_value=True)
            
            valid_idx = ~(features_nan_aligned | target_nan_aligned)
            train_features_clean = train_features.reindex(common_index).loc[valid_idx]
            train_target_clean = train_target.reindex(common_index).loc[valid_idx]
            
            if len(train_features_clean) < 50:
                continue
            
            # 标准化特征
            features_scaled = self.scaler.fit_transform(train_features_clean)
            
            # 训练模型
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            
            try:
                model.fit(features_scaled, train_target_clean)
                
                # 预测下一个时间点
                if i < len(data):
                    next_features = features.iloc[i:i+1]
                    if not next_features.isna().any().any():
                        next_features_scaled = self.scaler.transform(next_features)
                        prediction = model.predict(next_features_scaled)[0]
                        factor_series.iloc[i] = prediction
                
            except Exception:
                continue
        
        factors['rolling_ml_factor'] = factor_series
        
        return pd.DataFrame(factors, index=data.index)
    
    def build_adaptive_ml_factors(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        构建自适应机器学习因子
        
        Args:
            data: 市场数据
            **kwargs: 其他参数
            
        Returns:
            自适应ML因子DataFrame
        """
        factors = {}
        progress_callback = kwargs.get('progress_callback')
        
        # 准备特征
        if progress_callback:
            progress_callback(stage='ml', progress=5, message='自适应ML训练: 准备特征与环境检测...')
        features = self._prepare_features(data)
        target = self._prepare_target(data)
        
        # 获取正确的列名
        close_col = 'close' if 'close' in data.columns else 'S_DQ_CLOSE'
        
        # 市场状态检测
        returns = data[close_col].pct_change()
        volatility = returns.rolling(window=20).std()
        
        # 根据波动率调整模型参数
        high_vol_threshold = volatility.quantile(0.8)
        low_vol_threshold = volatility.quantile(0.2)
        
        # 初始化因子序列
        factor_series = pd.Series(index=data.index, dtype=float)
        
        window = kwargs.get('window', 252)
        
        total_steps = len(range(window, len(data)))
        for step_idx, i in enumerate(tqdm(range(window, len(data)), desc="自适应ML训练", total=total_steps, unit="时间点")):
            if progress_callback and step_idx % max(1, total_steps // 50) == 0:  # 每2%报告一次
                progress_pct = 10 + int((step_idx / total_steps) * 80)
                current_vol = volatility.iloc[i]
                vol_level = "高波动" if current_vol > high_vol_threshold else "低波动" if current_vol < low_vol_threshold else "中等波动"
                progress_callback(stage='ml', progress=progress_pct, message=f'自适应ML训练: {step_idx+1}/{total_steps}, {vol_level}环境')
            
            # 额外的细粒度更新
            if progress_callback and step_idx % max(1, total_steps // 100) == 0:  # 每1%一次微更新
                progress_pct = 10 + int((step_idx / total_steps) * 80)
                progress_callback(stage='ml', progress=progress_pct, message=f'自适应ML训练: 分析时间点 {step_idx+1}/{total_steps}...')
            
            # 获取当前波动率
            current_vol = volatility.iloc[i]
            
            # 根据波动率选择模型参数
            if current_vol > high_vol_threshold:
                # 高波动率环境：使用更保守的模型
                model = Ridge(alpha=10.0)
            elif current_vol < low_vol_threshold:
                # 低波动率环境：使用更激进的模型
                model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
            else:
                # 中等波动率环境：使用平衡的模型
                model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
            
            # 训练数据
            train_features = features.iloc[i-window:i]
            train_target = target.iloc[i-window:i]
            
            # 移除NaN值 - 确保索引对齐
            features_nan_mask = train_features.isna().any(axis=1)
            target_nan_mask = train_target.isna()
            
            # 重新索引以确保对齐
            common_index = train_features.index.intersection(train_target.index)
            features_nan_aligned = features_nan_mask.reindex(common_index, fill_value=True)
            target_nan_aligned = target_nan_mask.reindex(common_index, fill_value=True)
            
            valid_idx = ~(features_nan_aligned | target_nan_aligned)
            train_features_clean = train_features.reindex(common_index).loc[valid_idx]
            train_target_clean = train_target.reindex(common_index).loc[valid_idx]
            
            if len(train_features_clean) < 50:
                continue
            
            # 标准化特征
            features_scaled = self.scaler.fit_transform(train_features_clean)
            
            try:
                model.fit(features_scaled, train_target_clean)
                
                # 预测下一个时间点
                if i < len(data):
                    next_features = features.iloc[i:i+1]
                    if not next_features.isna().any().any():
                        next_features_scaled = self.scaler.transform(next_features)
                        prediction = model.predict(next_features_scaled)[0]
                        factor_series.iloc[i] = prediction
                
            except Exception:
                continue
        
        factors['adaptive_ml_factor'] = factor_series
        
        return pd.DataFrame(factors, index=data.index)
    
    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        准备特征数据
        
        Args:
            data: 市场数据
            
        Returns:
            特征DataFrame
        """
        features = pd.DataFrame(index=data.index)
        
        # 获取正确的列名
        close_col = 'close' if 'close' in data.columns else 'S_DQ_CLOSE'
        volume_col = 'volume' if 'volume' in data.columns else 'S_DQ_VOLUME'
        high_col = 'high' if 'high' in data.columns else 'S_DQ_HIGH'
        low_col = 'low' if 'low' in data.columns else 'S_DQ_LOW'
        open_col = 'open' if 'open' in data.columns else 'S_DQ_OPEN'
        
        # 价格特征
        features['returns'] = data[close_col].pct_change()
        features['log_returns'] = np.log(data[close_col] / data[close_col].shift(1))
        features['high_low_ratio'] = data[high_col] / (data[low_col] + 1e-8)
        features['close_open_ratio'] = data[close_col] / (data[open_col] + 1e-8)
        
        # 移动平均
        for window in [5, 10, 20, 50]:
            features[f'ma_{window}'] = data[close_col].rolling(window=window).mean()
            features[f'ma_ratio_{window}'] = data[close_col] / (features[f'ma_{window}'] + 1e-8)
        
        # 波动率特征
        for window in [5, 10, 20]:
            features[f'volatility_{window}'] = features['returns'].rolling(window=window).std()
        
        # 成交量特征
        features['volume_ratio'] = data[volume_col] / (data[volume_col].rolling(window=20).mean() + 1e-8)
        features['volume_ma_5'] = data[volume_col].rolling(window=5).mean()
        features['volume_ma_20'] = data[volume_col].rolling(window=20).mean()
        
        # 动量特征
        for period in [1, 5, 10, 20]:
            features[f'momentum_{period}'] = data[close_col] / data[close_col].shift(period) - 1
        
        # 技术指标
        # RSI
        for window in [14, 21]:
            delta = data[close_col].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / (loss + 1e-8)
            features[f'rsi_{window}'] = 100 - (100 / (1 + rs))
        
        # 价格位置
        for window in [20, 50]:
            min_price = data[close_col].rolling(window=window).min()
            max_price = data[close_col].rolling(window=window).max()
            features[f'price_position_{window}'] = (data[close_col] - min_price) / (max_price - min_price + 1e-8)
        
        return features
    
    def _prepare_target(self, data: pd.DataFrame) -> pd.Series:
        """
        准备目标变量
        
        Args:
            data: 市场数据
            
        Returns:
            目标变量Series
        """
        # 获取正确的列名
        close_col = 'close' if 'close' in data.columns else 'S_DQ_CLOSE'
        
        # ⚠️ 注意：这里使用未来1期收益率作为ML模型的训练目标
        # 这是ML模型训练的正常做法，不是因子计算中的未来函数问题
        # 在实盘使用时，模型已经训练完成，只使用历史特征进行预测
        future_returns = data[close_col].shift(-1) / data[close_col] - 1
        return future_returns
    
    def build_all_ml_factors(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        构建所有机器学习因子
        
        Args:
            data: 市场数据
            **kwargs: 其他参数
            
        Returns:
            所有ML因子DataFrame
        """
        print("构建机器学习因子...")
        
        all_factors = pd.DataFrame(index=data.index)
        
        # 构建各类ML因子
        factor_types = [
            'ensemble_factors',
            'pca_factors',
            'feature_selection_factors',
            'rolling_ml_factors',
            'adaptive_ml_factors'
        ]
        
        progress_callback = kwargs.get('progress_callback')
        
        # 立即发送初始进度更新
        if progress_callback:
            progress_callback(stage='ml', progress=1, message='机器学习因子: 初始化完成，开始构建...')
        
        for idx, factor_type in enumerate(tqdm(factor_types, desc="构建ML因子", unit="类型")):
            try:
                base_progress = int((idx / len(factor_types)) * 90)  # 每个阶段占90%中的一部分
                
                if factor_type == 'ensemble_factors':
                    if progress_callback:
                        progress_callback(stage='ml', progress=base_progress, message='集成模型训练: 开始训练集成模型...')
                        progress_callback(stage='ml', progress=base_progress + 1, message='集成模型训练: 准备特征与训练集...')
                    factors = self.build_ensemble_factors(data, progress_callback=progress_callback, **kwargs)
                elif factor_type == 'pca_factors':
                    if progress_callback:
                        progress_callback(stage='ml', progress=base_progress + 2, message='PCA 降维: 开始主成分分析...')
                    factors = self.build_pca_factors(data, progress_callback=progress_callback, **kwargs)
                elif factor_type == 'feature_selection_factors':
                    if progress_callback:
                        progress_callback(stage='ml', progress=base_progress + 2, message='特征选择: 分析特征重要性...')
                    factors = self.build_feature_selection_factors(data, progress_callback=progress_callback, **kwargs)
                elif factor_type == 'rolling_ml_factors':
                    if progress_callback:
                        progress_callback(stage='ml', progress=base_progress + 2, message='滚动ML: 开始时间窗口训练...')
                    factors = self.build_rolling_ml_factors(data, progress_callback=progress_callback, **kwargs)
                elif factor_type == 'adaptive_ml_factors':
                    if progress_callback:
                        progress_callback(stage='ml', progress=base_progress + 2, message='自适应ML训练: 分析市场环境...')
                    factors = self.build_adaptive_ml_factors(data, progress_callback=progress_callback, **kwargs)
                
                all_factors = pd.concat([all_factors, factors], axis=1)
                tqdm.write(f"✓ 成功构建 {factor_type}: {len(factors.columns)} 个因子")
                if progress_callback:
                    completion_progress = int(((idx+1) / len(factor_types)) * 90)
                    progress_callback(stage='ml', progress=completion_progress, message=f'{factor_type} 完成: {len(factors.columns)} 个因子')
                
            except Exception as e:
                tqdm.write(f"✗ 构建 {factor_type} 失败: {e}")
                continue
        
        # 处理异常值
        all_factors = all_factors.replace([np.inf, -np.inf], np.nan)
        all_factors = all_factors.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        print(f"总共构建了 {len(all_factors.columns)} 个机器学习因子")
        if progress_callback:
            progress_callback(stage='ml', progress=100, message='ML因子构建完成')
        
        return all_factors 