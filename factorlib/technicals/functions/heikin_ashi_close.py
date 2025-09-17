def calculate(data, **kwargs):
    import pandas as pd
    ha_close = (data['open'] + data['high'] + data['low'] + data['close']) / 4
    return ha_close


