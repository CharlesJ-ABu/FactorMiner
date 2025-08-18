def calculate(data, period=20, **kwargs):
    import pandas as pd
    ret = data['close'].pct_change()
    vol = ret.rolling(window=period).std()
    var = ret / vol.replace(0, pd.NA)
    return var.fillna(0.0)

