def calculate(data, column='close', period=100, **kwargs):
    import pandas as pd
    s = data[column]
    return s.rolling(window=period).apply(lambda x: (x.rank(pct=True).iloc[-1] if len(x) > 0 else 0.0), raw=False).fillna(0.0)

