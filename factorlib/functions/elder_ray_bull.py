def calculate(data, ema_period=13, **kwargs):
    import pandas as pd
    ema = data['close'].ewm(span=ema_period).mean()
    bull_power = data['high'] - ema
    return bull_power.fillna(0.0)

