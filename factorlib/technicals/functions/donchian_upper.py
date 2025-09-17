def calculate(data, period=20, **kwargs):
    return data['high'].rolling(window=period).max()

