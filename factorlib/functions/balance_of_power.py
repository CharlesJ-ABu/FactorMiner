def calculate(data, **kwargs):
    import pandas as pd
    bop = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, pd.NA)
    return bop.fillna(0.0)


