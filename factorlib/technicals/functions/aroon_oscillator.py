def calculate(data, period=25, **kwargs):
    from factorlib.functions.aroon_up import calculate as a_up
    from factorlib.functions.aroon_down import calculate as a_down
    up = a_up(data, period=period)
    down = a_down(data, period=period)
    return up - down

