import pandas as pd
import numpy as np

def calculate(data: pd.DataFrame, **kwargs) -> pd.Series:
    # 返回 开盘价 列。若列缺失则返回 NaN 序列。
    if data is None or len(data) == 0:
        return pd.Series(dtype=float)
    if "open" not in data.columns:
        return pd.Series(np.nan, index=data.index)
    series = pd.to_numeric(data["open"], errors="coerce")
    return series
