import pandas as pd
import numpy as np

def calculate(data: pd.DataFrame, length: int = 3, **kwargs) -> pd.Series:
    if data is None or len(data) == 0 or "close" not in data.columns:
        return pd.Series(dtype=float)
    c = pd.to_numeric(data["close"], errors="coerce")
    cond = (c > c.shift(1)) & (c.shift(1) > c.shift(2)) & (c.shift(2) > c.shift(3))
    return cond.astype(int)
