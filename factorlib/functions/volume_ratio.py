def calculate(data, fast=5, slow=20, **kwargs):
    import pandas as pd
    vol = data['volume']
    return (vol.rolling(fast).mean() / vol.rolling(slow).mean().replace(0, pd.NA)).fillna(0.0)

