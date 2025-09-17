def calculate(data, period=20, **kwargs):
    import pandas as pd
    tp = (data['high'] + data['low'] + data['close']) / 3
    ma = tp.rolling(window=period).mean()
    md = (tp - ma).abs().rolling(window=period).mean()
    cci = (tp - ma) / (0.015 * md.replace(0, pd.NA))
    return cci.fillna(0.0)

