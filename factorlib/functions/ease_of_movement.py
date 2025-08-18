def calculate(data, volume_period=14, **kwargs):
    import pandas as pd
    distance_moved = ((data['high'] + data['low'])/2 - (data['high'].shift(1) + data['low'].shift(1))/2)
    box_ratio = (data['volume'] / (data['high'] - data['low']).replace(0, pd.NA))
    emv = distance_moved / box_ratio.replace(0, pd.NA)
    return emv.rolling(window=volume_period).mean().fillna(0.0)

