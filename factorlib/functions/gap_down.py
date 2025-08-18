def calculate(data, mode='open_vs_prev_close', min_gap=0.0, return_type='magnitude', **kwargs):
    prev_close = data['close'].shift(1)
    prev_low = data['low'].shift(1)
    if mode == 'strict':
        condition = data['high'] < prev_low
        gap_mag = (prev_low - data['high']) / prev_low
    elif mode == 'open_vs_prev_low':
        condition = data['open'] < prev_low
        gap_mag = (prev_low - data['open']) / prev_low
    else:
        condition = data['open'] < prev_close
        gap_mag = (prev_close - data['open']) / prev_close

    if return_type == 'bool':
        if min_gap > 0:
            out = (gap_mag > min_gap) & condition
        else:
            out = condition
        return out.astype('float64').reindex(data.index).fillna(0.0)
    else:
        mag = gap_mag.where(condition, 0.0)
        if min_gap > 0:
            mag = mag.where(mag > min_gap, 0.0)
        return mag.reindex(data.index).fillna(0.0)

