def calculate(data, fast=5, slow=34, **kwargs):
    import pandas as pd
    median = (data['high'] + data['low']) / 2
    ao = median.rolling(window=int(fast)).mean() - median.rolling(window=int(slow)).mean()
    return ao.fillna(0.0)


