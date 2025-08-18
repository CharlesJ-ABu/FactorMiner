import pandas as pd
import numpy as np

def calculate(data: pd.DataFrame, **kwargs) -> pd.Series:
    if data is None or len(data) == 0 or "close" not in data.columns:
        return pd.Series(dtype=float)
    c = pd.to_numeric(data["close"], errors="coerce")
    return ((c > c.shift(1)) & (c.shift(1) < c.shift(2))).astype(int)
