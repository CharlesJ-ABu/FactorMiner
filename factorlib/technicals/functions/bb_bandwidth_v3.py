
def calculate(data, period=20, std_dev=2):
    """
    计算布林带指标
    
    参数:
        - period: 周期
        - std_dev: 标准差倍数
    """
    # 1. 计算中轨(SMA)
    middle = data['close'].rolling(window=period).mean()
    
    # 2. 计算标准差
    std = data['close'].rolling(window=period).std()
    
    # 3. 计算上轨和下轨
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    
    # 4. 计算带宽
    bandwidth = (upper - lower) / middle
    
    return bandwidth  # 返回带宽
