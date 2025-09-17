def calculate(data, long_col='long_accounts', short_col='short_accounts', **kwargs):
    import pandas as pd
    if long_col in data.columns and short_col in data.columns:
        denom = (data[long_col] + data[short_col]).replace(0, pd.NA)
        return (data[long_col] / denom).fillna(0.0)
    return pd.Series(0.0, index=data.index)

