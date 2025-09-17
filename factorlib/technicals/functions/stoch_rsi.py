def calculate(data, rsi_period=14, stoch_period=14, k_smooth=3, d_smooth=3, **kwargs):
    close = data['close']
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    lowest = rsi.rolling(window=stoch_period).min()
    highest = rsi.rolling(window=stoch_period).max()
    stoch_rsi = (rsi - lowest) / (highest - lowest) * 100
    k = stoch_rsi.rolling(window=k_smooth).mean()
    d = k.rolling(window=d_smooth).mean()
    output = kwargs.get('output', 'k')  # 'k' | 'd' | 'stoch_rsi'
    if output == 'd':
        return d
    elif output == 'stoch_rsi':
        return stoch_rsi
    else:
        return k

