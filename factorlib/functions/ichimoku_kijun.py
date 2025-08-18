def calculate(data, period=26, **kwargs):
    high = data['high'].rolling(window=period).max()
    low = data['low'].rolling(window=period).min()
    return (high + low) / 2.0

