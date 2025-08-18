import pandas as pd
import numpy as np

def calculate(data, period=20, **kwargs):
    return data['close'].ewm(span=period).mean()

