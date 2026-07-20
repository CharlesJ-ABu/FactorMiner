import numpy as np
import pandas as pd
from core.miner.registry import OperatorRegistry

@OperatorRegistry.register(arity=1)
def ts_frac_diff_05(series: pd.Series) -> pd.Series:
    """Fractional differencing with d=0.5 over a 20-bar rolling window."""
    d = 0.5
    window = 20
    weights = [1.0]
    for k in range(1, window):
        weights.append(-weights[-1] * (d - k + 1) / k)
    # Reverse weights to align with window elements (latest at the end)
    weights = np.array(weights[::-1])
    
    def frac_diff_window(x):
        return np.dot(x, weights[-len(x):])
        
    return series.rolling(window, min_periods=window).apply(frac_diff_window, raw=True)


@OperatorRegistry.register(arity=1)
def ts_hurst_60(series: pd.Series) -> pd.Series:
    """Rolling Hurst Exponent estimation over 60 bars using variance ratio."""
    def calc_hurst(x):
        # We need variance of diffs at various lags tau
        taus = np.array([2, 5, 10, 20])
        variances = []
        for tau in taus:
            # calculate diffs of lag tau
            diffs = x[tau:] - x[:-tau]
            if len(diffs) > 0:
                variances.append(np.var(diffs))
            else:
                variances.append(np.nan)
        
        valid = ~np.isnan(variances) & (np.array(variances) > 0)
        if sum(valid) < 2:
            return np.nan
            
        x_log = np.log(taus[valid])
        y_log = np.log(np.array(variances)[valid])
        
        # log(Var) = 2H * log(tau) + c
        # So slope = 2H => H = slope / 2
        poly = np.polyfit(x_log, y_log, 1)
        H = poly[0] / 2.0
        return H

    return series.rolling(60, min_periods=40).apply(calc_hurst, raw=True)


@OperatorRegistry.register(arity=1)
def ts_sampen_20(series: pd.Series) -> pd.Series:
    """Rolling Sample Entropy with m=2 over a 20-bar window."""
    def calc_sampen(x):
        m = 2
        N = len(x)
        r = 0.2 * np.std(x)
        
        if r == 0 or N < m + 2:
            return np.nan
            
        def count_matches(m_val):
            templates = np.array([x[i:i+m_val] for i in range(N-m_val)])
            diff = np.abs(templates[:, None, :] - templates[None, :, :])
            dist = np.max(diff, axis=2)
            np.fill_diagonal(dist, np.inf) # Ignore self-matching
            return np.sum(dist < r)
            
        B = count_matches(m)
        A = count_matches(m+1)
        
        if B == 0 or A == 0:
            return np.nan # Avoid log(0)
            
        return -np.log(A / B)

    return series.rolling(20, min_periods=20).apply(calc_sampen, raw=True)
