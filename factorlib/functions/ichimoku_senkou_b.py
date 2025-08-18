def calculate(data, period=52, shift=26, **kwargs):
    high = data['high'].rolling(window=period).max()
    low = data['low'].rolling(window=period).min()
    span_b = (high + low) / 2.0
    return span_b.shift(int(shift)).fillna(0.0)

