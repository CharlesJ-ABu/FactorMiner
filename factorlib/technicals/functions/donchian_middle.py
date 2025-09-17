def calculate(data, period=20, **kwargs):
    lower = data['low'].rolling(window=period).min()
    upper = data['high'].rolling(window=period).max()
    return (lower + upper) / 2.0

