def calculate(data, window=5, **kwargs):
    import pandas as pd
    w = int(window)
    if w < 3:
        w = 3
    half = w // 2
    low = data['low']
    def is_fractal_down(i):
        if i < half or i+half >= len(low):
            return 0.0
        center = low.iloc[i]
        left = low.iloc[i-half:i]
        right = low.iloc[i+1:i+half+1]
        return float(center == min(left.min(), center) and center == min(right.min(), center))
    out = pd.Series(0.0, index=low.index)
    for i in range(len(low)):
        out.iloc[i] = is_fractal_down(i)
    return out


