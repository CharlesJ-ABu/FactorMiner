import pandas as pd
import numpy as np

def calculate(data, fast_period=12, slow_period=26, signal_period=9, **kwargs):
    ema_fast = data['close'].ewm(span=fast_period).mean()
    ema_slow = data['close'].ewm(span=slow_period).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period).mean()
    return signal_line

