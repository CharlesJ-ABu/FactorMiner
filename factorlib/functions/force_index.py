def calculate(data, period=13, **kwargs):
    import pandas as pd
    fi_raw = data['close'].diff() * data['volume']
    # 使用EMA平滑
    fi = fi_raw.ewm(span=int(period)).mean()
    return fi.fillna(0.0)


