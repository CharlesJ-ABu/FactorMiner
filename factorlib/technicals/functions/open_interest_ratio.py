def calculate(data, oi_col='open_interest', vol_col='volume', **kwargs):
    import pandas as pd
    if oi_col in data.columns and vol_col in data.columns:
        return (data[oi_col] / data[vol_col].replace(0, pd.NA)).fillna(0.0)
    return pd.Series(0.0, index=data.index)

