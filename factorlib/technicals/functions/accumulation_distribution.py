def calculate(data, **kwargs):
    import pandas as pd
    high = data['high']
    low = data['low']
    close = data['close']
    volume = data['volume']
    mf_multiplier = ((close - low) - (high - close)) / (high - low).replace(0, pd.NA)
    mf_volume = mf_multiplier * volume
    ad = mf_volume.cumsum()
    return ad.fillna(0.0)

