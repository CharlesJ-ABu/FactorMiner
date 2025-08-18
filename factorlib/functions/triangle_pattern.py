def calculate(data, window=50, **kwargs):
    import pandas as pd
    # 简化：上沿=高点滚动线性回归斜率， 下沿=低点滚动线性回归斜率，收敛(同号且绝对值变小)视为三角形分数
    high = data['high']
    low = data['low']
    def slope(s):
        import numpy as np
        x = np.arange(len(s))
        if len(s) < 2:
            return 0.0
        A = np.vstack([x, np.ones(len(x))]).T
        m, _ = np.linalg.lstsq(A, s.values, rcond=None)[0]
        return m
    up_slope = high.rolling(window=window).apply(slope, raw=False)
    down_slope = low.rolling(window=window).apply(slope, raw=False)
    score = (up_slope.abs() + down_slope.abs())
    return (-score).fillna(0.0)

