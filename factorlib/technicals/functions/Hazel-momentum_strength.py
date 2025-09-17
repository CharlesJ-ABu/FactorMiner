import pandas as pd
import numpy as np

def calculate(data: pd.DataFrame, window: int = 5, **kwargs) -> pd.Series:
    required = {"close", "volume"}
    if data is None or len(data) == 0 or not required.issubset(set(data.columns)):
        return pd.Series(dtype=float)
    c = pd.to_numeric(data["close"], errors="coerce")
    v = pd.to_numeric(data["volume"], errors="coerce")
    pm = (c - c.shift(window)) / c.shift(window)
    vm = (v - v.shift(window)) / v.shift(window)
    return pm.abs() * vm
