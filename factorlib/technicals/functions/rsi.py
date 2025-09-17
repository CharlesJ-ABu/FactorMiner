import pandas as pd
import numpy as np

def calculate(data, period=14, **kwargs):
    delta = data['close'].diff()
    gain = delta.where(delta>0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta<0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100/(1+rs))

