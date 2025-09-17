def calculate(data, s1=7, s2=14, s3=28, **kwargs):
    import pandas as pd
    high, low, close = data['high'], data['low'], data['close']
    bp = close - low.combine(close.shift(1), min)
    tr = (high.combine(close.shift(1), max) - low.combine(close.shift(1), min))
    avg1 = bp.rolling(s1).sum() / tr.rolling(s1).sum()
    avg2 = bp.rolling(s2).sum() / tr.rolling(s2).sum()
    avg3 = bp.rolling(s3).sum() / tr.rolling(s3).sum()
    uo = 100 * (4*avg1 + 2*avg2 + avg3) / 7
    return uo.fillna(0.0)

