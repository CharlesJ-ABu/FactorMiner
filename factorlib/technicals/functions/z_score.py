import pandas as pd
import numpy as np

def calculate(data, period=20, **kwargs):
    mean = data['close'].rolling(window=period).mean()
    std = data['close'].rolling(window=period).std()
    return (data['close']-mean)/std

