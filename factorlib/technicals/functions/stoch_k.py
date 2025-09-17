import pandas as pd
import numpy as np

def calculate(data, period=14, **kwargs):
    lowest = data['low'].rolling(window=period).min()
    highest = data['high'].rolling(window=period).max()
    return (data['close']-lowest)/(highest-lowest)*100

