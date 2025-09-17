def calculate(data, swing_lookback=5, **kwargs):
    import pandas as pd
    high = data['high']; low = data['low']
    
    # 🚨 修复未来函数问题：
    # 原来的错误：使用 high.shift(-1) 和 low.shift(-1) 获取未来数据
    # 修复后：只使用历史数据，延迟确认swing点
    
    # 延迟确认swing high：需要等待下一个bar确认
    swing_high = high[(high.shift(1) < high) & (high.shift(2) < high.shift(1))].rolling(swing_lookback).max()
    
    # 延迟确认swing low：需要等待下一个bar确认  
    swing_low = low[(low.shift(1) > low) & (low.shift(2) > low.shift(1))].rolling(swing_lookback).min()
    
    # 简化：结构方向 = 最近swing高/低的相对位置
    dir_series = (swing_high.fillna(method='ffill') > swing_high.fillna(method='ffill').shift(1)).astype(int) \
                 - (swing_low.fillna(method='ffill') < swing_low.fillna(method='ffill').shift(1)).astype(int)
    
    return dir_series.fillna(0.0)

