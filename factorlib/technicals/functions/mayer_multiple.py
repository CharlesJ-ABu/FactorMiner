def calculate(data, period=200, **kwargs):
    import pandas as pd
    mm = data['close'] / data['close'].rolling(window=period).mean().replace(0, pd.NA)
    return mm.fillna(0.0)

