import pandas as pd
import numpy as np

def calculate(data, **kwargs):
    pv = (data['close']*data['volume']).cumsum()
    vv = data['volume'].cumsum()
    return pv / vv

