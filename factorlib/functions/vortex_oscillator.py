def calculate(data, period=14, mode='osc', **kwargs):
    import pandas as pd
    import numpy as np
    high = data['high']
    low = data['low']
    close = data['close']
    tr = (high.combine(close.shift(1), max) - low.combine(close.shift(1), min))
    vm_plus = (high - low.shift(1)).abs()
    vm_minus = (low - high.shift(1)).abs()
    tr_sum = tr.rolling(int(period)).sum()
    vi_plus = vm_plus.rolling(int(period)).sum() / tr_sum.replace(0, pd.NA)
    vi_minus = vm_minus.rolling(int(period)).sum() / tr_sum.replace(0, pd.NA)
    if mode == 'plus':
        return vi_plus.fillna(0.0)
    if mode == 'minus':
        return vi_minus.fillna(0.0)
    # osc = vi+ - vi-
    return (vi_plus - vi_minus).fillna(0.0)


