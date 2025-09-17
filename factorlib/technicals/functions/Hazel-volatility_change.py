import pandas as pd
import numpy as np

def calculate(data: pd.DataFrame, window: int = 10, **kwargs) -> pd.Series:
    if data is None or len(data) == 0 or "close" not in data.columns:
        return pd.Series(dtype=float)
    c = pd.to_numeric(data["close"], errors="coerce")
    vol = c.rolling(window).std() / c.rolling(window).mean()
    return vol - vol.shift(1)
