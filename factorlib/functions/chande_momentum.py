def calculate(data, period=14, **kwargs):
    import pandas as pd
    close = data['close']
    diff = close.diff()
    up = diff.where(diff > 0, 0.0)
    down = (-diff).where(diff < 0, 0.0)
    cu = up.rolling(window=period).sum()
    cd = down.rolling(window=period).sum()
    cmo = (cu - cd) / (cu + cd).replace(0, pd.NA) * 100
    return cmo.fillna(0.0)

