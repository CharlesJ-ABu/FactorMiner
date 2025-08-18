def calculate(data, body_small_ratio=0.3, **kwargs):
    import pandas as pd
    o = data['open']; c = data['close']
    body = (c - o).abs()
    prev_o = o.shift(1); prev_c = c.shift(1); prev_body = body.shift(1)
    prev2_o = o.shift(2); prev2_c = c.shift(2); prev2_body = body.shift(2)
    bull_long = prev2_c > prev2_o
    small = body.shift(1) <= (prev2_body.rolling(3).mean() * body_small_ratio)
    bear_drop = c < (prev2_o + prev2_c)/2
    signal = (bull_long & small & bear_drop).astype(float)
    return signal.fillna(0.0)

