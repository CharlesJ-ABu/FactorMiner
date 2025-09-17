def calculate(data, period=12, **kwargs):
    return data['close'].pct_change(periods=period)

