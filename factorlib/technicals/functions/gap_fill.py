def calculate(data, lookback=20, **kwargs):
    import pandas as pd
    # 简化版：若当前收盘价回到前一根收盘价与当根开盘价之间（缺口区域），则视为回补
    prev_close = data['close'].shift(1)
    open_ = data['open']
    close = data['close']
    filled = ((close - prev_close) * (open_ - prev_close) <= 0).astype(float)
    # 可选：要求在 lookback 内出现过跳空（此处简化直接输出）
    return filled.fillna(0.0)

