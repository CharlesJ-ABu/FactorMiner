def calculate(data, period=20, multiplier=2, **kwargs):
    ema = data['close'].ewm(span=period).mean()
    tr1 = data['high'] - data['low']
    tr2 = (data['high'] - data['close'].shift(1)).abs()
    tr3 = (data['low'] - data['close'].shift(1)).abs()
    tr = (tr1.combine(tr2, max)).combine(tr3, max)
    atr = tr.rolling(window=period).mean()
    return ema - multiplier * atr

