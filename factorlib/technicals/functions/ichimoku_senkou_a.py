def calculate(data, tenkan_period=9, kijun_period=26, shift=26, **kwargs):
    high_t = data['high'].rolling(window=tenkan_period).max()
    low_t = data['low'].rolling(window=tenkan_period).min()
    tenkan = (high_t + low_t) / 2.0
    high_k = data['high'].rolling(window=kijun_period).max()
    low_k = data['low'].rolling(window=kijun_period).min()
    kijun = (high_k + low_k) / 2.0
    span_a = (tenkan + kijun) / 2.0
    return span_a.shift(int(shift)).fillna(0.0)

