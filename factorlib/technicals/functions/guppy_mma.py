def calculate(data, fast_periods=(3,5,8,10,12,15), slow_periods=(30,35,40,45,50,60), mode='spread', **kwargs):
    import pandas as pd
    close = data['close']
    fast_emas = [close.ewm(span=int(p)).mean() for p in fast_periods]
    slow_emas = [close.ewm(span=int(p)).mean() for p in slow_periods]
    if mode == 'avg_fast':
        return sum(fast_emas) / len(fast_emas)
    if mode == 'avg_slow':
        return sum(slow_emas) / len(slow_emas)
    # spread: 快均线平均 - 慢均线平均
    return (sum(fast_emas) / len(fast_emas)) - (sum(slow_emas) / len(slow_emas))


