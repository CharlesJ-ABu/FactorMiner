import pandas as pd
import numpy as np

def calculate(data, period=14, **kwargs):
    tr1 = data['high'] - data['low']
    tr2 = (data['high'] - data['close'].shift(1)).abs()
    tr3 = (data['low'] - data['close'].shift(1)).abs()
    tr = pd.concat([tr1,tr2,tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

