def calculate(data, period=25, **kwargs):
    import pandas as pd
    high = data['high']
    def aroon_up_window(s):
        idx = s.values.argmax()
        return (len(s)-1 - idx) / (len(s)-1) * 100 if len(s) > 1 else 0.0
    return high.rolling(window=period).apply(lambda s: 100 - aroon_up_window(s), raw=False).fillna(0.0)

