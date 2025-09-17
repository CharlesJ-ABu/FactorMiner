import pandas as pd
import numpy as np

def calculate(data, period=14, **kwargs):
    highest = data['high'].rolling(window=period).max()
    lowest = data['low'].rolling(window=period).min()
    return (highest - data['close'])/(highest - lowest) * -100

