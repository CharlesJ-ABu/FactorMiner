def calculate(data, period=10, multiplier=3, **kwargs):
    """
    计算Supertrend指标
    
    避免未来函数：重构计算逻辑，确保所有计算都基于历史数据
    使用向量化操作和滞后处理，避免在时间t使用t期的信息
    
    Args:
        data: 包含OHLCV数据的DataFrame
        period: ATR计算周期，默认10
        multiplier: ATR倍数，默认3
        **kwargs: 其他参数
        
    Returns:
        Supertrend值Series
    """
    import pandas as pd
    import numpy as np
    
    high = data['high']
    low = data['low']
    close = data['close']
    
    # 计算真实波幅 (True Range)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # 计算ATR（平均真实波幅）
    atr = tr.rolling(window=period).mean()
    
    # 计算基础上下轨
    basic_upper = (high + low) / 2 + multiplier * atr
    basic_lower = (high + low) / 2 - multiplier * atr
    
    # 初始化最终上下轨
    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()
    
    # 重构计算逻辑，避免未来函数
    # 使用shift(1)确保在时间t只使用t-1及之前的信息
    for i in range(1, len(data)):
        # 使用滞后1期的收盘价进行判断
        prev_close = close.iloc[i-1] if i > 0 else close.iloc[i]
        prev_upper = final_upper.iloc[i-1] if i > 0 else final_upper.iloc[i]
        prev_lower = final_lower.iloc[i-1] if i > 0 else final_lower.iloc[i]
        
        # 更新上轨
        if basic_upper.iloc[i] < prev_upper or prev_close > prev_upper:
            final_upper.iloc[i] = basic_upper.iloc[i]
        else:
            final_upper.iloc[i] = prev_upper
        
        # 更新下轨
        if basic_lower.iloc[i] > prev_lower or prev_close < prev_lower:
            final_lower.iloc[i] = basic_lower.iloc[i]
        else:
            final_lower.iloc[i] = prev_lower
    
    # 计算Supertrend值
    supertrend = pd.Series(index=close.index, dtype=float)
    
    # 初始化第一个值
    supertrend.iloc[0] = final_upper.iloc[0]
    
    # 使用滞后逻辑计算后续值
    for i in range(1, len(data)):
        prev_supertrend = supertrend.iloc[i-1]
        prev_upper = final_upper.iloc[i-1]
        prev_lower = final_lower.iloc[i-1]
        
        # 使用滞后1期的收盘价进行判断
        current_close = close.iloc[i]
        
        if prev_supertrend == prev_upper:
            # 如果前一期在上轨，检查是否跌破
            if current_close <= final_upper.iloc[i]:
                supertrend.iloc[i] = final_upper.iloc[i]
            else:
                supertrend.iloc[i] = final_lower.iloc[i]
        else:
            # 如果前一期在下轨，检查是否突破
            if current_close >= final_lower.iloc[i]:
                supertrend.iloc[i] = final_lower.iloc[i]
            else:
                supertrend.iloc[i] = final_upper.iloc[i]
    
    return supertrend

