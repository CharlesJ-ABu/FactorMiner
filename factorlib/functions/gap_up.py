def calculate(data, **kwargs):

    mode = kwargs.get('mode', 'open_vs_prev_close')  # 'open_vs_prev_close' | 'open_vs_prev_high' | 'strict'
    min_gap = float(kwargs.get('min_gap', 0.0))      # 相对阈值，例如 0.005 表示 0.5%
    return_type = kwargs.get('return_type', 'magnitude')  # 'magnitude' | 'bool'

    prev_close = data['close'].shift(1)
    prev_high = data['high'].shift(1)

    if mode == 'strict':
        # 当前最低价高于前一根最高价
        condition = data['low'] > prev_high
        gap_mag = (data['low'] - prev_high) / prev_high
    elif mode == 'open_vs_prev_high':
        # 开盘价高于前一根最高价
        condition = data['open'] > prev_high
        gap_mag = (data['open'] - prev_high) / prev_high
    else:
        # 默认：开盘价高于前一根收盘价
        condition = data['open'] > prev_close
        gap_mag = (data['open'] - prev_close) / prev_close

    if return_type == 'bool':
        if min_gap > 0:
            out = (gap_mag > min_gap) & condition
        else:
            out = condition
        return out.astype('float64').reindex(data.index).fillna(0.0)
    else:
        # 仅保留向上缺口的正幅度，其他置 0
        mag = gap_mag.where(condition, 0.0)
        if min_gap > 0:
            mag = mag.where(mag > min_gap, 0.0)
        return mag.reindex(data.index).fillna(0.0)

