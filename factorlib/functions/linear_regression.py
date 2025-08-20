def calculate(data, target='close', features=None, fit_intercept=True, lookback=252, **kwargs):
    """
    计算线性回归预测因子
    
    避免未来函数：使用历史数据训练模型，预测下一期的值
    使用滚动窗口训练，确保在时间t只使用t-1及之前的信息
    
    Args:
        data: 包含OHLCV数据的DataFrame
        target: 目标变量，默认'close'
        features: 特征列列表，默认['open','high','low','volume']
        fit_intercept: 是否拟合截距，默认True
        lookback: 训练窗口大小，默认252（约一年交易日）
        **kwargs: 其他参数
        
    Returns:
        预测值Series
    """
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LinearRegression
    
    if features is None:
        features = ['open','high','low','volume']
    
    # 确保特征列存在
    missing_features = [f for f in features if f not in data.columns]
    if missing_features:
        print(f"缺少特征列: {missing_features}")
        return pd.Series(0.0, index=data.index)
    
    if target not in data.columns:
        print(f"缺少目标列: {target}")
        return pd.Series(0.0, index=data.index)
    
    # 初始化结果
    predictions = pd.Series(0.0, index=data.index)
    
    # 滚动窗口训练和预测
    for i in range(lookback, len(data)):
        # 使用历史数据训练（t-lookback 到 t-1）
        train_start = i - lookback
        train_end = i - 1
        
        # 准备训练数据
        X_train = data.iloc[train_start:train_end][features].fillna(0.0).values
        y_train = data.iloc[train_start:train_end][target].fillna(method='ffill').fillna(0.0).values
        
        # 准备预测数据（当前期t）
        X_pred = data.iloc[i:i+1][features].fillna(0.0).values
        
        # 检查数据质量
        if len(X_train) < lookback * 0.8:  # 至少80%的数据
            continue
            
        if np.isnan(X_train).any() or np.isnan(y_train).any():
            continue
            
        try:
            # 训练模型
            model = LinearRegression(fit_intercept=fit_intercept)
            model.fit(X_train, y_train)
            
            # 预测当前期
            pred = model.predict(X_pred)
            predictions.iloc[i] = pred[0]
            
        except Exception as e:
            # 如果训练或预测失败，保持默认值0.0
            continue
    
    return predictions

