def calculate(data, period=10, **kwargs):
    import pandas as pd
    close = data['close']
    std = close.rolling(window=period).std()
    rvi = (close - close.rolling(window=period).mean()) / (std.replace(0, pd.NA))
    return rvi.fillna(0.0)

