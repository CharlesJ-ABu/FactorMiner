def calculate(data, lookback=20, **kwargs):
    import pandas as pd
    close = data['close']
    recent_high = close.rolling(window=lookback).max()
    recent_low = close.rolling(window=lookback).min()
    # 强度：突破相对区间位置
    strength = (close - recent_low) / (recent_high - recent_low).replace(0, pd.NA)
    return strength.fillna(0.0)

