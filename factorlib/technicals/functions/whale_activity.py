def calculate(data, large_tx_col='large_transactions', vol_col='volume', period=14, **kwargs):
    import pandas as pd
    # 大单数/成交量的比率及均线
    if large_tx_col in data.columns and vol_col in data.columns:
        ratio = (data[large_tx_col] / data[vol_col].replace(0, pd.NA))
        return ratio.rolling(window=period).mean().fillna(0.0)
    return pd.Series(0.0, index=data.index)

