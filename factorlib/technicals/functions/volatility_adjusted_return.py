def calculate(data, period=20, **kwargs):
    """
    计算波动率调整收益率因子
    
    避免未来函数：使用滞后1期的波动率，确保在时间t只使用t-1及之前的信息
    
    Args:
        data: 包含OHLCV数据的DataFrame
        period: 波动率计算窗口，默认20期
        **kwargs: 其他参数
        
    Returns:
        波动率调整收益率Series
    """
    import pandas as pd
    
    # 计算收益率
    ret = data['close'].pct_change()
    
    # 使用滞后1期的波动率，避免未来函数
    # 在时间t，我们只能使用t-1及之前的信息
    vol = ret.shift(1).rolling(window=period).std()
    
    # 计算波动率调整收益率
    var = ret / vol.replace(0, pd.NA)
    
    # 填充NaN值
    return var.fillna(0.0)

