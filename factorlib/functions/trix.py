def calculate(data, period=15, **kwargs):
    import pandas as pd
    close = data['close']
    ema1 = close.ewm(span=int(period)).mean()
    ema2 = ema1.ewm(span=int(period)).mean()
    ema3 = ema2.ewm(span=int(period)).mean()
    trix = ema3.pct_change() * 100
    return trix.fillna(0.0)


