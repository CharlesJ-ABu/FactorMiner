def calculate(data, body_ratio_min=0.5, **kwargs):
    import pandas as pd
    o = data['open']; c = data['close']
    prev_o = o.shift(1); prev_c = c.shift(1)
    prev_bear = prev_c < prev_o
    curr_bull = c > o
    engulf = (c >= prev_o) & (o <= prev_c)
    # 体量阈值：当前实体 >= 上根实体 * body_ratio_min
    body_curr = (c - o).abs(); body_prev = (prev_c - prev_o).abs()
    strong = body_curr >= body_prev * float(body_ratio_min)
    signal = (prev_bear & curr_bull & engulf & strong).astype(float)
    return signal.fillna(0.0)

