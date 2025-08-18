def calculate(data, ema_period=13, **kwargs):
    import pandas as pd
    ema = data['close'].ewm(span=ema_period).mean()
    bear_power = data['low'] - ema
    return bear_power.fillna(0.0)

