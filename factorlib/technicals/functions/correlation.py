def calculate(data, x='close', y='volume', period=20, method='pearson', **kwargs):
    import pandas as pd
    s1 = data[x]
    s2 = data[y]
    if method == 'spearman':
        s1 = s1.rank()
        s2 = s2.rank()
    cov = (s1 - s1.rolling(period).mean()) * (s2 - s2.rolling(period).mean())
    cov = cov.rolling(period).mean()
    std1 = s1.rolling(period).std()
    std2 = s2.rolling(period).std()
    corr = cov / (std1 * std2).replace(0, pd.NA)
    return corr.fillna(0.0)

