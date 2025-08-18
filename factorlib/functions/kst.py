def calculate(data, rp=(10,15,20,30), sp=(10,10,10,15), **kwargs):
    import pandas as pd
    close = data['close']
    r1 = close.pct_change(rp[0]).rolling(sp[0]).sum()
    r2 = close.pct_change(rp[1]).rolling(sp[1]).sum()
    r3 = close.pct_change(rp[2]).rolling(sp[2]).sum()
    r4 = close.pct_change(rp[3]).rolling(sp[3]).sum()
    kst = r1*1 + r2*2 + r3*3 + r4*4
    return kst.fillna(0.0)


