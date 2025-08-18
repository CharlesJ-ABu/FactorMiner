
def calculate(data, period=10, **kwargs):
    return data["close"].pct_change(periods=period)
