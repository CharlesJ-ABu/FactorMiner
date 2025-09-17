def calculate(data, oi_col='open_interest', price_col='close', window=50, **kwargs):
    import pandas as pd
    # 简化版：以持仓量变化最大处附近的价格作为潜在清算集中区
    if oi_col in data.columns and price_col in data.columns:
        d_oi = data[oi_col].diff().rolling(window=window).sum().abs()
        level = d_oi.rolling(window=window).apply(lambda x: x.argmax(), raw=True)
        # 将索引位置映射到价格（近似）
        price = data[price_col]
        lvl_series = price.rolling(window=window).apply(lambda s: s.iloc[-1], raw=False)
        return lvl_series.fillna(method='ffill').fillna(0.0)
    return pd.Series(0.0, index=data.index)

