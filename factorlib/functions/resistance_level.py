def calculate(data, period=20, **kwargs):
    import pandas as pd
    # 近period内的局部最高作为阻力估计
    return data['high'].rolling(window=period).max().fillna(0.0)

