def calculate(data, upper_shadow_ratio=2.0, lower_shadow_max=0.3, **kwargs):
    import pandas as pd
    o = data['open']; c = data['close']; h = data['high']; l = data['low']
    body = (c - o).abs()
    upper_shadow = (h - o.combine(c, max))
    lower_shadow = (o.combine(c, min) - l)
    cond = (upper_shadow >= body * float(upper_shadow_ratio)) & (lower_shadow <= body * float(lower_shadow_max))
    return cond.astype(float).fillna(0.0)

