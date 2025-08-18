def calculate(data, perp_col='perp_price', spot_col='close', **kwargs):
    import pandas as pd
    if perp_col in data.columns and spot_col in data.columns:
        return ((data[perp_col] - data[spot_col]) / data[spot_col].replace(0, pd.NA)).fillna(0.0)
    return pd.Series(0.0, index=data.index)

