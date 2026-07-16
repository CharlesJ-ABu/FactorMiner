"""Robust time-series operators available to custom GP and RL searches."""

import numpy as np
import pandas as pd

from core.miner.registry import OperatorRegistry


@OperatorRegistry.register(arity=1)
def custom_ts_decay(series: pd.Series) -> pd.Series:
    """A 5-bar linearly weighted mean for a Series or cross-asset DataFrame."""
    weights = [1, 2, 3, 4, 5]
    return series.rolling(5, min_periods=3).apply(
        lambda window: sum(value * weight for value, weight in zip(window, weights[-len(window):])) / sum(weights[-len(window):]),
        raw=True,
    )


@OperatorRegistry.register(arity=1)
def ts_zscore_20(series: pd.Series) -> pd.Series:
    """20-bar rolling Z-score, with zero-variance windows treated as missing."""
    rolling_mean = series.rolling(20, min_periods=8).mean()
    rolling_std = series.rolling(20, min_periods=8).std().replace(0, np.nan)
    return (series - rolling_mean) / rolling_std


@OperatorRegistry.register(arity=1)
def ts_delta_5(series: pd.Series) -> pd.Series:
    """Five-bar change, useful for short-horizon momentum and volume shocks."""
    return series.diff(5)


@OperatorRegistry.register(arity=1)
def ts_rank_20(series: pd.Series) -> pd.Series:
    """Percentile rank of the latest value within the trailing 20 observations."""
    return series.rolling(20, min_periods=8).apply(
        lambda window: pd.Series(window).rank(pct=True).iloc[-1],
        raw=False,
    )


@OperatorRegistry.register(arity=1)
def ts_volatility_20(series: pd.Series) -> pd.Series:
    """Rolling volatility of one-period percentage changes over 20 bars."""
    return series.pct_change(fill_method=None).rolling(20, min_periods=8).std()
