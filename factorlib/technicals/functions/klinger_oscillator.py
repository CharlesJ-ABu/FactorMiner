def calculate(data, fast=34, slow=55, **kwargs):
    import pandas as pd
    high, low, close, vol = data['high'], data['low'], data['close'], data['volume']
    dm = close - close.shift(1)
    trend = ((high + low + close) / 3) - ((high.shift(1) + low.shift(1) + close.shift(1)) / 3)
    vf = vol * trend
    ko = vf.ewm(span=int(fast)).mean() - vf.ewm(span=int(slow)).mean()
    return ko.fillna(0.0)


