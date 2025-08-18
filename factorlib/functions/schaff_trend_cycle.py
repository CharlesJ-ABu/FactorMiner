def calculate(data, fast=23, slow=50, k_period=10, d_period=3, **kwargs):
    import pandas as pd
    close = data['close']
    ema_fast = close.ewm(span=int(fast)).mean()
    ema_slow = close.ewm(span=int(slow)).mean()
    macd = ema_fast - ema_slow
    # Stochastic of MACD
    lowest = macd.rolling(int(k_period)).min()
    highest = macd.rolling(int(k_period)).max()
    stoch = (macd - lowest) / (highest - lowest).replace(0, pd.NA) * 100
    stc = stoch.rolling(int(d_period)).mean()
    return stc.fillna(0.0)


