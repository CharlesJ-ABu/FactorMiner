import pandas as pd
import numpy as np

def calculate(data: pd.DataFrame, mom_window: int = 5, trend_window: int = 10, pos_window: int = 20, sent_window: int = 10, **kwargs) -> pd.Series:
    required = {"close", "high", "low", "volume"}
    if data is None or len(data) == 0 or not required.issubset(set(data.columns)):
        return pd.Series(dtype=float)
    c = pd.to_numeric(data["close"], errors="coerce")
    h = pd.to_numeric(data["high"], errors="coerce")
    l = pd.to_numeric(data["low"], errors="coerce")
    v = pd.to_numeric(data["volume"], errors="coerce")

    price_mom = (c - c.shift(mom_window)) / c.shift(mom_window)
    vol_mom = (v - v.shift(mom_window)) / v.shift(mom_window)
    trend_strength = (c - c.shift(trend_window)).abs() / c.shift(trend_window)
    hmax = h.rolling(pos_window).max(); lmin = l.rolling(pos_window).min()
    pos = (c - lmin) / (hmax - lmin).replace(0, np.nan)
    mc = c.rolling(sent_window).mean(); mv = v.rolling(sent_window).mean()
    sentiment = ((c > mc) & (v > mv)).astype(float)

    return (price_mom * 0.3 + vol_mom * 0.2 + trend_strength * 0.2 + pos * 0.15 + sentiment * 0.15)
