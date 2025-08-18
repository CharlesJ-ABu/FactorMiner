#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子优化器模块
提供因子参数优化、因子组合优化和自动因子选择功能
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import warnings
warnings.filterwarnings('ignore')

class FactorOptimizer:
    """因子优化器主类"""
    
    def __init__(self, data=None, returns=None):
        self.data = data
        self.returns = returns
        self.scaler = StandardScaler()
        self.best_params = {}
        self.optimization_history = []
        
    def set_data(self, data, returns):
        """设置数据和收益率"""
        self.data = data
        self.returns = returns
        
    def optimize_factor_parameters(self, factor_func, param_grid, metric='ic'):
        """
        优化因子参数
        
        Args:
            factor_func: 因子计算函数
            param_grid: 参数网格
            metric: 优化指标 ('ic', 'ir', 'effectiveness_score')
            
        Returns:
            best_params: 最佳参数
            best_score: 最佳得分
        """
        print(f"开始优化因子参数，参数网格大小: {len(param_grid)}")
        
        best_score = -np.inf
        best_params = None
        
        for i, params in enumerate(param_grid):
            try:
                # 计算因子
                factor = factor_func(self.data, **params)
                
                # 计算指标
                score = self._calculate_metric(factor, self.returns, metric)
                
                # 记录结果
                self.optimization_history.append({
                    'params': params,
                    'score': score,
                    'iteration': i
                })
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    
                if (i + 1) % 10 == 0:
                    print(f"进度: {i+1}/{len(param_grid)}, 当前最佳得分: {best_score:.4f}")
                    
            except Exception as e:
                print(f"参数 {params} 计算失败: {e}")
                continue
        
        self.best_params = best_params
        print(f"参数优化完成，最佳得分: {best_score:.4f}")
        print(f"最佳参数: {best_params}")
        
        return best_params, best_score
    
    def optimize_factor_combination(self, factors_df, max_factors=10, method='greedy'):
        """
        优化因子组合
        
        Args:
            factors_df: 因子DataFrame
            max_factors: 最大因子数量
            method: 优化方法 ('greedy', 'genetic', 'lasso')
            
        Returns:
            best_combination: 最佳因子组合
            best_score: 最佳得分
        """
        print(f"开始优化因子组合，总因子数: {len(factors_df.columns)}")
        
        if method == 'greedy':
            return self._greedy_factor_selection(factors_df, max_factors)
        elif method == 'genetic':
            return self._genetic_factor_selection(factors_df, max_factors)
        elif method == 'lasso':
            return self._lasso_factor_selection(factors_df, max_factors)
        else:
            raise ValueError(f"不支持的优化方法: {method}")
    
    def _greedy_factor_selection(self, factors_df, max_factors):
        """贪婪因子选择"""
        print("使用贪婪算法选择因子...")
        
        available_factors = list(factors_df.columns)
        selected_factors = []
        best_score = -np.inf
        
        for i in range(min(max_factors, len(available_factors))):
            current_best_factor = None
            current_best_score = -np.inf
            
            for factor in available_factors:
                # 临时添加因子
                test_factors = selected_factors + [factor]
                test_df = factors_df[test_factors]
                
                # 计算组合得分
                score = self._calculate_combination_score(test_df, self.returns)
                
                if score > current_best_score:
                    current_best_score = score
                    current_best_factor = factor
            
            if current_best_factor:
                selected_factors.append(current_best_factor)
                available_factors.remove(current_best_factor)
                best_score = current_best_score
                
                print(f"第 {i+1} 轮: 选择因子 {current_best_factor}, 得分: {best_score:.4f}")
        
        return selected_factors, best_score
    
    def _genetic_factor_selection(self, factors_df, max_factors, population_size=50, generations=100):
        """遗传算法因子选择"""
        print("使用遗传算法选择因子...")
        
        n_factors = len(factors_df.columns)
        factor_names = list(factors_df.columns)
        
        # 初始化种群
        population = []
        for _ in range(population_size):
            # 随机选择因子
            selected = np.random.choice([0, 1], size=n_factors, p=[0.7, 0.3])
            population.append(selected)
        
        best_individual = None
        best_score = -np.inf
        
        for generation in range(generations):
            # 评估适应度
            fitness_scores = []
            for individual in population:
                if individual.sum() > 0 and individual.sum() <= max_factors:
                    selected_factors = [factor_names[i] for i in range(n_factors) if individual[i]]
                    score = self._calculate_combination_score(factors_df[selected_factors], self.returns)
                    fitness_scores.append(score)
                else:
                    fitness_scores.append(-np.inf)
            
            # 找到最佳个体
            best_idx = np.argmax(fitness_scores)
            if fitness_scores[best_idx] > best_score:
                best_score = fitness_scores[best_idx]
                best_individual = population[best_idx].copy()
            
            # 选择
            selected_indices = self._tournament_selection(fitness_scores, population_size)
            new_population = [population[i] for i in selected_indices]
            
            # 交叉和变异
            for i in range(0, population_size, 2):
                if i + 1 < population_size:
                    # 交叉
                    if np.random.random() < 0.8:
                        crossover_point = np.random.randint(1, n_factors)
                        new_population[i], new_population[i+1] = self._crossover(
                            new_population[i], new_population[i+1], crossover_point
                        )
                    
                    # 变异
                    if np.random.random() < 0.1:
                        new_population[i] = self._mutate(new_population[i])
                    if np.random.random() < 0.1:
                        new_population[i+1] = self._mutate(new_population[i+1])
            
            population = new_population
            
            if generation % 20 == 0:
                print(f"第 {generation} 代: 最佳得分: {best_score:.4f}")
        
        # 返回最佳组合
        selected_factors = [factor_names[i] for i in range(n_factors) if best_individual[i]]
        return selected_factors, best_score
    
    def _lasso_factor_selection(self, factors_df, max_factors):
        """Lasso回归因子选择"""
        print("使用Lasso回归选择因子...")
        
        # 数据预处理
        X = factors_df.fillna(0)
        y = self.returns.fillna(0)
        
        # 标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # Lasso回归
        lasso = Lasso(alpha=0.01, max_iter=1000)
        lasso.fit(X_scaled, y)
        
        # 获取非零系数
        coefficients = lasso.coef_
        selected_indices = np.where(np.abs(coefficients) > 1e-6)[0]
        
        # 按系数绝对值排序，选择前max_factors个
        factor_importance = [(i, abs(coefficients[i])) for i in selected_indices]
        factor_importance.sort(key=lambda x: x[1], reverse=True)
        
        selected_factors = [factors_df.columns[i] for i, _ in factor_importance[:max_factors]]
        
        # 计算组合得分
        score = self._calculate_combination_score(factors_df[selected_factors], self.returns)
        
        return selected_factors, score
    
    def _tournament_selection(self, fitness_scores, population_size, tournament_size=3):
        """锦标赛选择"""
        selected_indices = []
        for _ in range(population_size):
            tournament_indices = np.random.choice(len(fitness_scores), tournament_size)
            tournament_scores = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_scores)]
            selected_indices.append(winner_idx)
        return selected_indices
    
    def _crossover(self, parent1, parent2, crossover_point):
        """交叉操作"""
        child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
        return child1, child2
    
    def _mutate(self, individual, mutation_rate=0.1):
        """变异操作"""
        for i in range(len(individual)):
            if np.random.random() < mutation_rate:
                individual[i] = 1 - individual[i]  # 翻转
        return individual
    
    def _calculate_metric(self, factor, returns, metric):
        """计算单个指标"""
        # 对齐数据
        factor_aligned = factor.dropna()
        # 使用索引交集更稳健
        common_index = factor_aligned.index.intersection(returns.index)
        if len(common_index) < 10:
            return -np.inf
        factor_aligned = factor_aligned.loc[common_index]
        returns_aligned = returns.loc[common_index]
        
        if metric == 'ic':
            return factor_aligned.corr(returns_aligned)
        elif metric == 'ir':
            ic = factor_aligned.corr(returns_aligned)
            ic_std = factor_aligned.rolling(20).corr(returns_aligned).std()
            return ic / ic_std if ic_std > 0 else 0
        elif metric == 'effectiveness_score':
            # 计算有效性评分
            ic = abs(factor_aligned.corr(returns_aligned))
            win_rate = self._calculate_win_rate(factor_aligned, returns_aligned)
            return (ic + win_rate) / 2
        else:
            raise ValueError(f"不支持的指标: {metric}")
    
    def _calculate_combination_score(self, factors_df, returns):
        """计算因子组合得分"""
        if factors_df.empty:
            return -np.inf
        
        # 对齐数据（使用索引交集）
        factors_aligned = factors_df.dropna()
        common_index = factors_aligned.index.intersection(returns.index)
        if len(common_index) < 10:
            return -np.inf
        factors_aligned = factors_aligned.loc[common_index]
        returns_aligned = returns.loc[common_index]
        
        # 计算组合因子（简单平均）
        combined_factor = factors_aligned.mean(axis=1)
        
        # 计算IC
        ic = combined_factor.corr(returns_aligned)
        
        # 计算胜率
        win_rate = self._calculate_win_rate(combined_factor, returns_aligned)
        
        # 综合得分
        score = (abs(ic) + win_rate) / 2
        
        return score
    
    def _calculate_win_rate(self, factor, returns):
        """计算胜率"""
        # 计算因子方向
        factor_direction = np.sign(factor)
        
        # 计算收益率方向
        returns_direction = np.sign(returns)
        
        # 计算正确预测的比例
        correct_predictions = (factor_direction == returns_direction).mean()
        
        return correct_predictions
    
    def create_ensemble_factor(self, factors_df, method='equal_weight'):
        """
        创建集成因子
        
        Args:
            factors_df: 因子DataFrame
            method: 集成方法 ('equal_weight', 'ic_weight', 'ml_weight')
            
        Returns:
            ensemble_factor: 集成因子
        """
        print(f"创建集成因子，方法: {method}")
        
        if method == 'equal_weight':
            return factors_df.mean(axis=1)
        elif method == 'ic_weight':
            return self._create_ic_weighted_factor(factors_df)
        elif method == 'ml_weight':
            return self._create_ml_weighted_factor(factors_df)
        else:
            raise ValueError(f"不支持的集成方法: {method}")
    
    def _create_ic_weighted_factor(self, factors_df):
        """创建IC加权因子"""
        # 计算每个因子的IC
        ic_scores = {}
        for col in factors_df.columns:
            ic = self._calculate_metric(factors_df[col], self.returns, 'ic')
            ic_scores[col] = abs(ic)
        
        # 归一化权重
        total_ic = sum(ic_scores.values())
        if total_ic == 0:
            return factors_df.mean(axis=1)
        
        weights = {col: ic / total_ic for col, ic in ic_scores.items()}
        
        # 加权平均
        weighted_factor = pd.Series(0, index=factors_df.index)
        for col, weight in weights.items():
            weighted_factor += weight * factors_df[col]
        
        return weighted_factor
    
    def _create_ml_weighted_factor(self, factors_df):
        """创建ML加权因子"""
        # 数据预处理
        X = factors_df.fillna(0)
        y = self.returns.fillna(0)
        
        # 使用随机森林学习权重
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        # 获取特征重要性作为权重
        feature_importance = rf.feature_importances_
        
        # 加权平均
        weighted_factor = pd.Series(0, index=factors_df.index)
        for i, col in enumerate(factors_df.columns):
            weighted_factor += feature_importance[i] * factors_df[col]
        
        return weighted_factor
    
    def optimize_hyperparameters(self, factors_df, model_type='ensemble'):
        """
        优化超参数
        
        Args:
            factors_df: 因子DataFrame
            model_type: 模型类型 ('ensemble', 'linear', 'ml')
            
        Returns:
            best_params: 最佳参数
            best_score: 最佳得分
        """
        print(f"优化超参数，模型类型: {model_type}")
        
        if model_type == 'ensemble':
            return self._optimize_ensemble_params(factors_df)
        elif model_type == 'linear':
            return self._optimize_linear_params(factors_df)
        elif model_type == 'ml':
            return self._optimize_ml_params(factors_df)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}") 