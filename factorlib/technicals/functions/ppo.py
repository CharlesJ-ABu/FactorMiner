def calculate(data, fast_period=12, slow_period=26, **kwargs):
    import pandas as pd
    ema_fast = data['close'].ewm(span=fast_period).mean()
    ema_slow = data['close'].ewm(span=slow_period).mean()
    ppo = (ema_fast - ema_slow) / ema_slow.replace(0, pd.NA) * 100
    return ppo.fillna(0.0)

