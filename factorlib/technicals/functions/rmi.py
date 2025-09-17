def calculate(data, period=20, momentum=5, **kwargs):
    import pandas as pd
    close = data['close']
    mom = close.diff(int(momentum))
    up = mom.clip(lower=0)
    down = (-mom).clip(lower=0)
    up_ave = up.rolling(int(period)).mean()
    down_ave = down.rolling(int(period)).mean()
    rmi = 100 - (100 / (1 + (up_ave / down_ave.replace(0, pd.NA))))
    return rmi.fillna(0.0)


