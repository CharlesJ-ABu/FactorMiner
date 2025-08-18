import pandas as pd
import numpy as np

def calculate(data, period=14, **kwargs):
    tp = (data['high']+data['low']+data['close'])/3
    mf = tp * data['volume']
    pos = mf.where(tp>tp.shift(1), 0).rolling(window=period).sum()
    neg = mf.where(tp<tp.shift(1), 0).rolling(window=period).sum()
    mr = pos/neg
    return 100 - (100/(1+mr))

