def calculate(data, period=10, multiplier=3, **kwargs):
    # 使用与 supertrend 相同的计算，输出方向（1/-1）
    import pandas as pd
    from factorlib.functions.supertrend import calculate as supertrend_calc
    st = supertrend_calc(data, period=period, multiplier=multiplier)
    return (data['close'] >= st).astype('int').replace({0: -1, 1: 1})

