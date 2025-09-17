import pandas as pd
import numpy as np

def calculate(data: pd.DataFrame, window: int = 5, **kwargs) -> pd.Series:
    if data is None or len(data) == 0 or "volume" not in data.columns:
        return pd.Series(dtype=float)
    vol = pd.to_numeric(data["volume"], errors="coerce")
    mom = (vol - vol.shift(window)) / vol.shift(window)
    return mom - mom.shift(1)
