def calculate(data, column='sentiment_score', period=7, normalize=True, **kwargs):
    import pandas as pd
    if column in data.columns:
        s = data[column].rolling(window=period).mean()
        if normalize:
            mu = s.rolling(window=period*4).mean()
            sd = s.rolling(window=period*4).std()
            s = (s - mu) / sd.replace(0, pd.NA)
        return s.fillna(0.0)
    return pd.Series(0.0, index=data.index)

