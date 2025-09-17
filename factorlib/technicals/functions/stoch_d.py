import pandas as pd
import numpy as np

def calculate(data, period=14, smooth=3, **kwargs):
    lowest = data['low'].rolling(window=period).min()
    highest = data['high'].rolling(window=period).max()
    k = (data['close']-lowest)/(highest-lowest)*100
    return k.rolling(window=smooth).mean()

