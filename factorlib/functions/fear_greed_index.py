def calculate(data, column='fear_greed', period=3, **kwargs):
    import pandas as pd
    if column in data.columns:
        return data[column].rolling(window=period).mean().fillna(0.0)
    return pd.Series(0.0, index=data.index)

