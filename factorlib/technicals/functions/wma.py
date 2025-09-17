import pandas as pd
import numpy as np

def calculate(data, period=20, **kwargs):
    weights = np.arange(1, period+1)
    return data['close'].rolling(window=period).apply(lambda x: np.dot(x, weights)/weights.sum(), raw=True)

