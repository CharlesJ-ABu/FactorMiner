import pandas as pd
import numpy as np

def calculate(data: pd.DataFrame, window: int = 20, **kwargs) -> pd.Series:
    required = {"close", "high", "low"}
    if data is None or len(data) == 0 or not required.issubset(set(data.columns)):
        return pd.Series(dtype=float)
    close = pd.to_numeric(data["close"], errors="coerce")
    high = pd.to_numeric(data["high"], errors="coerce")
    low = pd.to_numeric(data["low"], errors="coerce")
    high_max = high.rolling(window).max()
    low_min = low.rolling(window).min()
    denom = (high_max - low_min).replace(0, np.nan)
    return (close - low_min) / denom
