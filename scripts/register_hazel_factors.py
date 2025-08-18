#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将从策略特征工程片段中可直接用 pandas 计算的特征提取为因子，并按 V3 透明存储写入 factorlib。

注意：所有因子 ID 使用前缀 "Hazel-"。

存储结构：
- factorlib/definitions/<factor_id>.json
- factorlib/functions/<factor_id>.py  (入口函数名: calculate)
"""

from typing import List, Dict
from pathlib import Path
import sys

# 确保可以导入项目包
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from factor_miner.core.factor_storage import TransparentFactorStorage


def _wrap_function(imports: List[str], body: str) -> str:
    """拼接函数源码。入口函数名固定为 calculate。"""
    header = "\n".join(imports) + "\n\n"
    return header + body.strip() + "\n"


def register_hazel_factors() -> None:
    storage = TransparentFactorStorage()

    # 通用导入（所有函数文件都会包含）
    base_imports = [
        "import pandas as pd",
        "import numpy as np",
    ]

    # 定义待注册的因子（均为函数型因子）
    # 每个项包含：factor_id, name, description, category, parameters, function_code
    factors: List[Dict] = []

    # 1. 原始价格/成交量（标准化为 series 返回）
    for col, zh in [
        ("close", "收盘价"), ("open", "开盘价"), ("high", "最高价"), ("low", "最低价"), ("volume", "成交量")
    ]:
        factor_id = f"Hazel-raw_{col}"
        name = f"Hazel 原始{zh}"
        description = f"直接返回K线的{zh}列。"
        code = f"""
def calculate(data: pd.DataFrame, **kwargs) -> pd.Series:
    # 返回 {zh} 列。若列缺失则返回 NaN 序列。
    if data is None or len(data) == 0:
        return pd.Series(dtype=float)
    if "{col}" not in data.columns:
        return pd.Series(np.nan, index=data.index)
    series = pd.to_numeric(data["{col}"], errors="coerce")
    return series
"""
        factors.append({
            "factor_id": factor_id,
            "name": name,
            "description": description,
            "category": "technical",
            "parameters": {},
            "imports": base_imports,
            "function_code": code,
        })

    # 2. 价格动量/成交量动量（窗口可调）
    factors.append({
        "factor_id": "Hazel-price_momentum_5",
        "name": "Hazel 价格动量(5)",
        "description": "(close - close.shift(5)) / close.shift(5)",
        "category": "technical",
        "parameters": {"window": 5},
        "imports": base_imports,
        "function_code": """
def calculate(data: pd.DataFrame, window: int = 5, **kwargs) -> pd.Series:
    # 相对 5 根K 的价格动量。
    if data is None or len(data) == 0 or "close" not in data.columns:
        return pd.Series(dtype=float)
    close = pd.to_numeric(data["close"], errors="coerce")
    base = close.shift(window)
    return (close - base) / base
""",
    })

    factors.append({
        "factor_id": "Hazel-volume_momentum_5",
        "name": "Hazel 成交量动量(5)",
        "description": "(volume - volume.shift(5)) / volume.shift(5)",
        "category": "technical",
        "parameters": {"window": 5},
        "imports": base_imports,
        "function_code": """
def calculate(data: pd.DataFrame, window: int = 5, **kwargs) -> pd.Series:
    if data is None or len(data) == 0 or "volume" not in data.columns:
        return pd.Series(dtype=float)
    volume = pd.to_numeric(data["volume"], errors="coerce")
    base = volume.shift(window)
    return (volume - base) / base
""",
    })

    # 3. 波动率（滚动标准差 / 滚动均值）
    factors.append({
        "factor_id": "Hazel-volatility_10",
        "name": "Hazel 波动率(10)",
        "description": "rolling(std,10) / rolling(mean,10) of close",
        "category": "technical",
        "parameters": {"window": 10},
        "imports": base_imports,
        "function_code": """
def calculate(data: pd.DataFrame, window: int = 10, **kwargs) -> pd.Series:
    if data is None or len(data) == 0 or "close" not in data.columns:
        return pd.Series(dtype=float)
    close = pd.to_numeric(data["close"], errors="coerce")
    ma = close.rolling(window).mean()
    std = close.rolling(window).std()
    return std / ma
""",
    })

    # 4. 趋势强度（|close - close.shift(window)| / close.shift(window)）
    factors.append({
        "factor_id": "Hazel-trend_strength_10",
        "name": "Hazel 趋势强度(10)",
        "description": "abs(close - close.shift(10)) / close.shift(10)",
        "category": "technical",
        "parameters": {"window": 10},
        "imports": base_imports,
        "function_code": """
def calculate(data: pd.DataFrame, window: int = 10, **kwargs) -> pd.Series:
    if data is None or len(data) == 0 or "close" not in data.columns:
        return pd.Series(dtype=float)
    close = pd.to_numeric(data["close"], errors="coerce")
    base = close.shift(window)
    return (close - base).abs() / base
""",
    })

    # 5. 价格位置（在 rolling(window) 最高最低区间的归一化位置）
    factors.append({
        "factor_id": "Hazel-price_position_20",
        "name": "Hazel 价格位置(20)",
        "description": "(close - low_min) / (high_max - low_min) over 20",
        "category": "technical",
        "parameters": {"window": 20},
        "imports": base_imports,
        "function_code": """
def calculate(data: pd.DataFrame, window: int = 20, **kwargs) -> pd.Series:
    required = {"close", "high", "low"}
    if data is None or len(data) == 0 or not required.issubset(set(data.columns)):
        return pd.Series(dtype=float)
    close = pd.to_numeric(data["close"], errors="coerce")
    high = pd.to_numeric(data["high"], errors="coerce")
    low = pd.to_numeric(data["low"], errors="coerce")
    high_max = high.rolling(window).max()
    low_min = low.rolling(window).min()
    denom = (high_max - low_min).replace(0, np.nan)
    return (close - low_min) / denom
""",
    })

    # 6. 成交量/价格比
    factors.append({
        "factor_id": "Hazel-volume_price_ratio",
        "name": "Hazel 成交量价格比",
        "description": "volume / close",
        "category": "technical",
        "parameters": {},
        "imports": base_imports,
        "function_code": """
def calculate(data: pd.DataFrame, **kwargs) -> pd.Series:
    if data is None or len(data) == 0 or "close" not in data.columns or "volume" not in data.columns:
        return pd.Series(dtype=float)
    close = pd.to_numeric(data["close"], errors="coerce")
    volume = pd.to_numeric(data["volume"], errors="coerce")
    denom = close.replace(0, np.nan)
    return volume / denom
""",
    })

    # 7/8/9. 不同周期的价格变化率
    for fid, w, zh in [
        ("Hazel-short_price_change_1", 1, "短期"),
        ("Hazel-medium_price_change_10", 10, "中期"),
        ("Hazel-long_price_change_20", 20, "长期"),
    ]:
        name = f"Hazel {zh}价格变化({w})"
        desc = f"(close - close.shift({w})) / close.shift({w})"
        code = f"""
def calculate(data: pd.DataFrame, window: int = {w}, **kwargs) -> pd.Series:
    if data is None or len(data) == 0 or "close" not in data.columns:
        return pd.Series(dtype=float)
    close = pd.to_numeric(data["close"], errors="coerce")
    base = close.shift(window)
    return (close - base) / base
"""
        factors.append({
            "factor_id": fid,
            "name": name,
            "description": desc,
            "category": "technical",
            "parameters": {"window": w},
            "imports": base_imports,
            "function_code": code,
        })

    # 10. 价格加速度：短期变化的差分
    factors.append({
        "factor_id": "Hazel-price_acceleration",
        "name": "Hazel 价格加速度",
        "description": "短期价格变化的一阶差分",
        "category": "technical",
        "parameters": {"window": 1},
        "imports": base_imports,
        "function_code": """
def calculate(data: pd.DataFrame, window: int = 1, **kwargs) -> pd.Series:
    if data is None or len(data) == 0 or "close" not in data.columns:
        return pd.Series(dtype=float)
    close = pd.to_numeric(data["close"], errors="coerce")
    change = (close - close.shift(window)) / close.shift(window)
    return change - change.shift(1)
""",
    })

    # 11. 成交量加速度：成交量动量差分
    factors.append({
        "factor_id": "Hazel-volume_acceleration",
        "name": "Hazel 成交量加速度",
        "description": "成交量动量的一阶差分（默认窗口5）",
        "category": "technical",
        "parameters": {"window": 5},
        "imports": base_imports,
        "function_code": """
def calculate(data: pd.DataFrame, window: int = 5, **kwargs) -> pd.Series:
    if data is None or len(data) == 0 or "volume" not in data.columns:
        return pd.Series(dtype=float)
    vol = pd.to_numeric(data["volume"], errors="coerce")
    mom = (vol - vol.shift(window)) / vol.shift(window)
    return mom - mom.shift(1)
""",
    })

    # 12/13/14. 突破/跌破（价格、成交量）
    for fid, col, zh, w in [
        ("Hazel-price_breakout_20", "close", "价格突破", 20),
        ("Hazel-price_breakdown_20", "close", "价格跌破", 20),
        ("Hazel-volume_breakout_20", "volume", "成交量突破", 20),
    ]:
        if "breakout" in fid:
            code = f"""
def calculate(data: pd.DataFrame, window: int = {w}, **kwargs) -> pd.Series:
    if data is None or len(data) == 0 or "{col}" not in data.columns:
        return pd.Series(dtype=float)
    s = pd.to_numeric(data["{col}"], errors="coerce")
    thresh = s.rolling(window).max().shift(1)
    return (s > thresh).astype(int)
"""
            desc = f"{zh}：{col} 大于其 {w} 滚动最大（前一根）"
        else:
            code = f"""
def calculate(data: pd.DataFrame, window: int = {w}, **kwargs) -> pd.Series:
    if data is None or len(data) == 0 or "{col}" not in data.columns:
        return pd.Series(dtype=float)
    s = pd.to_numeric(data["{col}"], errors="coerce")
    thresh = s.rolling(window).min().shift(1)
    return (s < thresh).astype(int)
"""
            desc = f"{zh}：{col} 小于其 {w} 滚动最小（前一根）"

        factors.append({
            "factor_id": fid,
            "name": f"Hazel {zh}({w})",
            "description": desc,
            "category": "technical",
            "parameters": {"window": w},
            "imports": base_imports,
            "function_code": code,
        })

    # 15. 价格反转（上升反转形态的简单刻画）
    factors.append({
        "factor_id": "Hazel-price_reversal",
        "name": "Hazel 价格反转",
        "description": "(close > close.shift(1)) & (close.shift(1) < close.shift(2))",
        "category": "technical",
        "parameters": {},
        "imports": base_imports,
        "function_code": """
def calculate(data: pd.DataFrame, **kwargs) -> pd.Series:
    if data is None or len(data) == 0 or "close" not in data.columns:
        return pd.Series(dtype=float)
    c = pd.to_numeric(data["close"], errors="coerce")
    return ((c > c.shift(1)) & (c.shift(1) < c.shift(2))).astype(int)
""",
    })

    # 16. 趋势一致性（连续3根上升）
    factors.append({
        "factor_id": "Hazel-trend_consistency_3",
        "name": "Hazel 趋势一致性(3)",
        "description": "close 连续三根递增",
        "category": "technical",
        "parameters": {"length": 3},
        "imports": base_imports,
        "function_code": """
def calculate(data: pd.DataFrame, length: int = 3, **kwargs) -> pd.Series:
    if data is None or len(data) == 0 or "close" not in data.columns:
        return pd.Series(dtype=float)
    c = pd.to_numeric(data["close"], errors="coerce")
    cond = (c > c.shift(1)) & (c.shift(1) > c.shift(2)) & (c.shift(2) > c.shift(3))
    return cond.astype(int)
""",
    })

    # 17. 市场情绪（价格与成交量均高于10期均值）
    factors.append({
        "factor_id": "Hazel-market_sentiment_10",
        "name": "Hazel 市场情绪(10)",
        "description": "(close > ma10_close) & (volume > ma10_volume)",
        "category": "technical",
        "parameters": {"window": 10},
        "imports": base_imports,
        "function_code": """
def calculate(data: pd.DataFrame, window: int = 10, **kwargs) -> pd.Series:
    required = {"close", "volume"}
    if data is None or len(data) == 0 or not required.issubset(set(data.columns)):
        return pd.Series(dtype=float)
    c = pd.to_numeric(data["close"], errors="coerce")
    v = pd.to_numeric(data["volume"], errors="coerce")
    mc = c.rolling(window).mean()
    mv = v.rolling(window).mean()
    return ((c > mc) & (v > mv)).astype(int)
""",
    })

    # 18. 波动率变化（相邻期波动率差分）
    factors.append({
        "factor_id": "Hazel-volatility_change",
        "name": "Hazel 波动率变化",
        "description": "volatility(10) 的一阶差分",
        "category": "technical",
        "parameters": {"window": 10},
        "imports": base_imports,
        "function_code": """
def calculate(data: pd.DataFrame, window: int = 10, **kwargs) -> pd.Series:
    if data is None or len(data) == 0 or "close" not in data.columns:
        return pd.Series(dtype=float)
    c = pd.to_numeric(data["close"], errors="coerce")
    vol = c.rolling(window).std() / c.rolling(window).mean()
    return vol - vol.shift(1)
""",
    })

    # 19. 成交量价格背离（价格上升而量下降记为 +1；相反为 -1，否则 0）
    factors.append({
        "factor_id": "Hazel-volume_price_divergence",
        "name": "Hazel 成交量价格背离",
        "description": "(price_mom>0 & vol_mom<0) -> 1; (price_mom<0 & vol_mom>0) -> -1; else 0",
        "category": "technical",
        "parameters": {"window": 5},
        "imports": base_imports,
        "function_code": """
def calculate(data: pd.DataFrame, window: int = 5, **kwargs) -> pd.Series:
    required = {"close", "volume"}
    if data is None or len(data) == 0 or not required.issubset(set(data.columns)):
        return pd.Series(dtype=float)
    c = pd.to_numeric(data["close"], errors="coerce")
    v = pd.to_numeric(data["volume"], errors="coerce")
    pm = (c - c.shift(window)) / c.shift(window)
    vm = (v - v.shift(window)) / v.shift(window)
    pos = ((pm > 0) & (vm < 0)).astype(int)
    neg = ((pm < 0) & (vm > 0)).astype(int)
    return pos - neg
""",
    })

    # 20. 支撑阻力（等价于价格位置，保留为独立因子名以便评估）
    factors.append({
        "factor_id": "Hazel-support_resistance_20",
        "name": "Hazel 支撑阻力(20)",
        "description": "(close - low_min) / (high_max - low_min) over 20",
        "category": "technical",
        "parameters": {"window": 20},
        "imports": base_imports,
        "function_code": """
def calculate(data: pd.DataFrame, window: int = 20, **kwargs) -> pd.Series:
    required = {"close", "high", "low"}
    if data is None or len(data) == 0 or not required.issubset(set(data.columns)):
        return pd.Series(dtype=float)
    close = pd.to_numeric(data["close"], errors="coerce")
    high = pd.to_numeric(data["high"], errors="coerce")
    low = pd.to_numeric(data["low"], errors="coerce")
    high_max = high.rolling(window).max()
    low_min = low.rolling(window).min()
    denom = (high_max - low_min).replace(0, np.nan)
    return (close - low_min) / denom
""",
    })

    # 21. 动量强度（|price_momentum| * volume_momentum）
    factors.append({
        "factor_id": "Hazel-momentum_strength",
        "name": "Hazel 动量强度",
        "description": "abs(price_mom) * volume_mom (window=5)",
        "category": "technical",
        "parameters": {"window": 5},
        "imports": base_imports,
        "function_code": """
def calculate(data: pd.DataFrame, window: int = 5, **kwargs) -> pd.Series:
    required = {"close", "volume"}
    if data is None or len(data) == 0 or not required.issubset(set(data.columns)):
        return pd.Series(dtype=float)
    c = pd.to_numeric(data["close"], errors="coerce")
    v = pd.to_numeric(data["volume"], errors="coerce")
    pm = (c - c.shift(window)) / c.shift(window)
    vm = (v - v.shift(window)) / v.shift(window)
    return pm.abs() * vm
""",
    })

    # 22. 综合技术得分（线性组合）
    factors.append({
        "factor_id": "Hazel-technical_score",
        "name": "Hazel 综合技术得分",
        "description": "0.3*price_mom + 0.2*vol_mom + 0.2*trend_strength + 0.15*price_position + 0.15*sentiment",
        "category": "technical",
        "parameters": {"mom_window": 5, "trend_window": 10, "pos_window": 20, "sent_window": 10},
        "imports": base_imports,
        "function_code": """
def calculate(data: pd.DataFrame, mom_window: int = 5, trend_window: int = 10, pos_window: int = 20, sent_window: int = 10, **kwargs) -> pd.Series:
    required = {"close", "high", "low", "volume"}
    if data is None or len(data) == 0 or not required.issubset(set(data.columns)):
        return pd.Series(dtype=float)
    c = pd.to_numeric(data["close"], errors="coerce")
    h = pd.to_numeric(data["high"], errors="coerce")
    l = pd.to_numeric(data["low"], errors="coerce")
    v = pd.to_numeric(data["volume"], errors="coerce")

    price_mom = (c - c.shift(mom_window)) / c.shift(mom_window)
    vol_mom = (v - v.shift(mom_window)) / v.shift(mom_window)
    trend_strength = (c - c.shift(trend_window)).abs() / c.shift(trend_window)
    hmax = h.rolling(pos_window).max(); lmin = l.rolling(pos_window).min()
    pos = (c - lmin) / (hmax - lmin).replace(0, np.nan)
    mc = c.rolling(sent_window).mean(); mv = v.rolling(sent_window).mean()
    sentiment = ((c > mc) & (v > mv)).astype(float)

    return (price_mom * 0.3 + vol_mom * 0.2 + trend_strength * 0.2 + pos * 0.15 + sentiment * 0.15)
""",
    })

    # 执行保存
    ok, fail = 0, 0
    for f in factors:
        code = _wrap_function(f.get("imports", base_imports), f["function_code"]) 
        success = storage.save_function_factor(
            factor_id=f["factor_id"],
            name=f["name"],
            function_code=code,
            entry_point="calculate",
            description=f["description"],
            category=f["category"],
            parameters=f.get("parameters", {}),
            # 头部 import 已由 _wrap_function 写入，这里不再重复
            imports=[],
        )
        if success:
            ok += 1
        else:
            fail += 1

    print(f"Hazel 因子注册完成：成功 {ok} 个，失败 {fail} 个。")


if __name__ == "__main__":
    register_hazel_factors()


