import pandas as pd
import numpy as np

def calculate(data, period=10, fast_sc=2, slow_sc=30, **kwargs):
    close = data['close']
    change = (close - close.shift(period)).abs()
    volatility = close.diff().abs().rolling(window=period).sum()
    er = change / volatility
    fast = 2/(fast_sc+1)
    slow = 2/(slow_sc+1)
    sc = (er*(fast - slow)+slow)**2
    kama = pd.Series(index=close.index, dtype=float)
    if len(close) > period: kama.iloc[period] = close.iloc[period]
    for i in range(period+1, len(close)):
        kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i]*(close.iloc[i]-kama.iloc[i-1])
    return kama

