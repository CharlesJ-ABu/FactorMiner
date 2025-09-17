def calculate(data, period=20, **kwargs):
    return data['volume'].rolling(window=period).mean().fillna(0.0)

