def calculate(data, roc_long=14, roc_short=11, wma_period=10, **kwargs):
    import pandas as pd
    import numpy as np
    close = data['close']
    roc1 = close.pct_change(int(roc_long)) * 100
    roc2 = close.pct_change(int(roc_short)) * 100
    coppock_raw = roc1 + roc2
    n = int(wma_period)
    if n <= 1:
        return coppock_raw.fillna(0.0)
    weights = np.arange(1, n + 1)
    wma = coppock_raw.rolling(window=n).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    return wma.fillna(0.0)


