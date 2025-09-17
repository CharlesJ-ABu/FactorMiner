def calculate(data, period=20, std_dev=2, **kwargs):
    import pandas as pd
    sma = data['close'].rolling(window=period).mean()
    std = data['close'].rolling(window=period).std()
    upper = sma + std*std_dev
    lower = sma - std*std_dev
    bbp = (data['close'] - lower) / (upper - lower).replace(0, pd.NA)
    return bbp.fillna(0.0)

