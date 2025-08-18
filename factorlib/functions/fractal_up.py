def calculate(data, window=5, **kwargs):
    import pandas as pd
    w = int(window)
    if w < 3:
        w = 3
    half = w // 2
    high = data['high']
    def is_fractal_up(i):
        if i < half or i+half >= len(high):
            return 0.0
        center = high.iloc[i]
        left = high.iloc[i-half:i]
        right = high.iloc[i+1:i+half+1]
        return float(center == max(left.max(), center) and center == max(right.max(), center))
    out = pd.Series(0.0, index=high.index)
    for i in range(len(high)):
        out.iloc[i] = is_fractal_up(i)
    return out


