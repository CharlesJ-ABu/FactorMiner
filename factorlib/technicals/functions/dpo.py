def calculate(data, period=20, **kwargs):
    import pandas as pd
    shifted = int(period/2) + 1
    sma = data['close'].rolling(window=int(period)).mean()
    dpo = data['close'].shift(shifted) - sma
    return dpo.fillna(0.0)


