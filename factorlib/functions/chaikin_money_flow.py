def calculate(data, period=20, **kwargs):
    import pandas as pd
    mfm = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low']).replace(0, pd.NA)
    mfv = mfm * data['volume']
    cmf = mfv.rolling(window=period).sum() / data['volume'].rolling(window=period).sum()
    return cmf.fillna(0.0)

