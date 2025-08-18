def calculate(data, target='close', features=None, fit_intercept=True, **kwargs):
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LinearRegression
    if features is None:
        features = ['open','high','low','volume']
    X = data[features].fillna(0.0).values
    y = data[target].fillna(method='ffill').fillna(0.0).values
    model = LinearRegression(fit_intercept=fit_intercept)
    try:
        model.fit(X, y)
        pred = model.predict(X)
    except Exception:
        pred = np.zeros(len(y))
    return pd.Series(pred, index=data.index)

