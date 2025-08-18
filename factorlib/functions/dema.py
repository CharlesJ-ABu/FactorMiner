import pandas as pd
import numpy as np

def calculate(data, period=21, **kwargs):
    ema1 = data['close'].ewm(span=period).mean()
    ema2 = ema1.ewm(span=period).mean()
    return 2*ema1 - ema2

