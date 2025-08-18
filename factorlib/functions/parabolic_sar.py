def calculate(data, af_start=0.02, af_step=0.02, af_max=0.2, **kwargs):
    import pandas as pd
    high = data['high']
    low = data['low']
    close = data['close']
    sar = pd.Series(index=close.index, dtype=float)
    trend_up = True
    af = af_start
    ep = high.iloc[0]
    sar.iloc[0] = low.iloc[0]
    for i in range(1, len(close)):
        prev_sar = sar.iloc[i-1]
        if trend_up:
            sar.iloc[i] = prev_sar + af * (ep - prev_sar)
            sar.iloc[i] = min(sar.iloc[i], low.iloc[i-1], low.iloc[i])
            if high.iloc[i] > ep:
                ep = high.iloc[i]
                af = min(af + af_step, af_max)
            if low.iloc[i] < sar.iloc[i]:
                trend_up = False
                sar.iloc[i] = ep
                ep = low.iloc[i]
                af = af_start
        else:
            sar.iloc[i] = prev_sar + af * (ep - prev_sar)
            sar.iloc[i] = max(sar.iloc[i], high.iloc[i-1], high.iloc[i])
            if low.iloc[i] < ep:
                ep = low.iloc[i]
                af = min(af + af_step, af_max)
            if high.iloc[i] > sar.iloc[i]:
                trend_up = True
                sar.iloc[i] = ep
                ep = high.iloc[i]
                af = af_start
    return sar.fillna(0.0)

