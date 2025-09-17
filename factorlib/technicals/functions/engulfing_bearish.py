def calculate(data, body_ratio_min=0.5, **kwargs):
    import pandas as pd
    o = data['open']; c = data['close']
    prev_o = o.shift(1); prev_c = c.shift(1)
    prev_bull = prev_c > prev_o
    curr_bear = c < o
    engulf = (o >= prev_c) & (c <= prev_o)
    body_curr = (o - c).abs(); body_prev = (prev_c - prev_o).abs()
    strong = body_curr >= body_prev * float(body_ratio_min)
    signal = (prev_bull & curr_bear & engulf & strong).astype(float)
    return signal.fillna(0.0)

