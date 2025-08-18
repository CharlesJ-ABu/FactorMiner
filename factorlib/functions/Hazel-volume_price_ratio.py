import pandas as pd
import numpy as np

def calculate(data: pd.DataFrame, **kwargs) -> pd.Series:
    if data is None or len(data) == 0 or "close" not in data.columns or "volume" not in data.columns:
        return pd.Series(dtype=float)
    close = pd.to_numeric(data["close"], errors="coerce")
    volume = pd.to_numeric(data["volume"], errors="coerce")
    denom = close.replace(0, np.nan)
    return volume / denom
