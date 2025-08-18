def calculate(data, period=20, **kwargs):
    import pandas as pd
    # 近period内的局部最低作为支撑估计
    return data['low'].rolling(window=period).min().fillna(0.0)

