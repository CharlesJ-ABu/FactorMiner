def calculate(data, period=12, method='pct', **kwargs):
    import pandas as pd
    close = data['close']
    if method == 'log':
        return (close / close.shift(period)).apply(lambda x: pd.NA if x <= 0 else (pd.np.log(x))).fillna(0.0)
    else:
        return close.pct_change(periods=period).fillna(0.0)

