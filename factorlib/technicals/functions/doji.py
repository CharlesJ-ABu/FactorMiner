def calculate(data, threshold=0.001, **kwargs):
    import numpy as np
    open_ = data['open']
    close = data['close']
    high = data['high']
    low = data['low']
    body = (close - open_).abs()
    range_ = (high - low).replace(0, np.nan)
    ratio = body / range_
    doji = (ratio <= float(threshold)).astype(float)
    return doji.reindex(data.index).fillna(0.0)

