import pandas as pd
import numpy as np

def calculate(data, **kwargs):
    direction = np.sign(data['close'].diff()).fillna(0)
    return (direction * data['volume']).cumsum()

