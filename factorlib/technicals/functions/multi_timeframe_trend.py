def calculate(data, fast=20, slow=50, **kwargs):
    import pandas as pd
    ema_fast = data['close'].ewm(span=fast).mean()
    ema_slow = data['close'].ewm(span=slow).mean()
    return (ema_fast - ema_slow).fillna(0.0)

