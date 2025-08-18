import pandas as pd
import numpy as np

def calculate(data, period=20, std_dev=2, **kwargs):
    sma = data['close'].rolling(window=period).mean()
    std = data['close'].rolling(window=period).std()
    return sma + std*std_dev

