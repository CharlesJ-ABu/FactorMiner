def calculate(data, period=14, **kwargs):
    import pandas as pd
    close = data['close']
    max_close = close.rolling(int(period)).max()
    drawdown_pct = (close - max_close) / max_close * 100
    ui = (drawdown_pct.pow(2).rolling(int(period)).mean()).pow(0.5)
    return ui.fillna(0.0)


