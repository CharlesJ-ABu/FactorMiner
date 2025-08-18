import pandas as pd
import numpy as np

def calculate(data: pd.DataFrame, window: int = 5, **kwargs) -> pd.Series:
    if data is None or len(data) == 0 or "volume" not in data.columns:
        return pd.Series(dtype=float)
    volume = pd.to_numeric(data["volume"], errors="coerce")
    base = volume.shift(window)
    return (volume - base) / base
