def calculate(data, shift=26, **kwargs):
    # 🚨 修复未来函数问题：
    # 原来的错误：使用 shift(-int(shift)) 获取未来数据
    # 修复后：使用历史数据，模拟Chikou Span的延迟显示效果
    
    # Chikou Span: 收盘价向前移位（历史数据）
    # 注意：这是为了模拟Chikou Span的显示效果，实际使用时需要延迟shift个周期
    return data['close'].shift(int(shift)).fillna(method='ffill').fillna(0.0)

