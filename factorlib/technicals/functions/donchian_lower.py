def calculate(data, period=20, **kwargs):
    return data['low'].rolling(window=period).min()

