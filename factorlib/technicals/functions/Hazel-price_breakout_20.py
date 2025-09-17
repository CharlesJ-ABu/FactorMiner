import pandas as pd
import numpy as np

def calculate(data: pd.DataFrame, window: int = 20, **kwargs) -> pd.Series:
    if data is None or len(data) == 0 or "close" not in data.columns:
        return pd.Series(dtype=float)
    s = pd.to_numeric(data["close"], errors="coerce")
    thresh = s.rolling(window).max().shift(1)
    return (s > thresh).astype(int)
