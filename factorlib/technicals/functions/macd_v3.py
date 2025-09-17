
def calculate(data, fast_period=12, slow_period=26, signal_period=9):
    """
    计算MACD指标
    
    参数:
        - fast_period: 快线周期
        - slow_period: 慢线周期
        - signal_period: 信号线周期
    """
    # 1. 计算快线和慢线
    fast_ema = data['close'].ewm(span=fast_period).mean()
    slow_ema = data['close'].ewm(span=slow_period).mean()
    
    # 2. 计算MACD线
    macd_line = fast_ema - slow_ema
    
    # 3. 计算信号线
    signal_line = macd_line.ewm(span=signal_period).mean()
    
    # 4. 计算MACD柱状图
    histogram = macd_line - signal_line
    
    return macd_line  # 返回MACD线
