def calculate(data, period=25, **kwargs):
    import pandas as pd
    low = data['low']
    def aroon_down_window(s):
        idx = s.values.argmin()
        return (len(s)-1 - idx) / (len(s)-1) * 100 if len(s) > 1 else 0.0
    return low.rolling(window=period).apply(lambda s: 100 - aroon_down_window(s), raw=False).fillna(0.0)

