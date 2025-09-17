def calculate(data, lower_shadow_ratio=2.0, upper_shadow_max=0.3, **kwargs):
    import pandas as pd
    o = data['open']; c = data['close']; h = data['high']; l = data['low']
    body = (c - o).abs()
    lower_shadow = (o.combine(c, min) - l)
    upper_shadow = (h - o.combine(c, max))
    cond = (lower_shadow >= body * float(lower_shadow_ratio)) & (upper_shadow <= body * float(upper_shadow_max))
    return cond.astype(float).fillna(0.0)

