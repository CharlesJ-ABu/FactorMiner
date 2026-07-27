import logging
import math
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from core.evaluation.targets import ForwardReturnTarget

logger = logging.getLogger(__name__)


class InspectorMetricEngine:
    """
    因子审查指标引擎：计算全维度的因子统计量，包含 IC/RankIC、IR、t-stat、Lag 衰减及分组多空收益。
    """

    @staticmethod
    def calculate_comprehensive_metrics(
        factor_values: Any,
        returns: Any,
        close_prices: Any = None,
        lags: List[int] = [1, 2, 3, 5, 10],
        n_quantiles: int = 5,
        price_data: Any = None,
        target_spec: ForwardReturnTarget | Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """
        计算全套审查指标字典。
        """
        result = {
            "pearson": {},
            "spearman": {},
            "decay": {},
            "quantiles": {},
            "turnover": 0.0,
            "coverage": 0.0,
            "total_bars": 0,
            "valid_bars": 0,
        }

        if factor_values is None or returns is None:
            return result

        # 兼容性整理
        if isinstance(factor_values, pd.DataFrame) and isinstance(returns, pd.DataFrame):
            return InspectorMetricEngine._evaluate_cross_asset(factor_values, returns, lags, n_quantiles)
        
        if not isinstance(factor_values, pd.Series):
            factor_values = pd.Series(factor_values)
        if not isinstance(returns, pd.Series):
            returns = pd.Series(returns)

        # 1. 基础有效性与覆盖率
        total_bars = len(factor_values)
        valid_mask = factor_values.notna() & returns.notna()
        valid_bars = int(valid_mask.sum())
        coverage = valid_bars / total_bars if total_bars > 0 else 0.0

        result["total_bars"] = total_bars
        result["valid_bars"] = valid_bars
        result["coverage"] = coverage

        if valid_bars < 10:
            logger.warning("Too few valid bars (%d) for inspection.", valid_bars)
            return result

        clean_f = factor_values[valid_mask]
        clean_r = returns[valid_mask]

        # 2. 逐 bar / 滚动 相关性 (Rolling / Overall)
        # 为评估 IC 稳定性，计算 20-bar 滚动相关性序列
        rolling_pearson = clean_f.rolling(20, min_periods=10).corr(clean_r).dropna()
        rank_f = clean_f.rank()
        rank_r = clean_r.rank()
        rolling_spearman = rank_f.rolling(20, min_periods=10).corr(rank_r).dropna()

        # 如果总体计算：
        p_ic = clean_f.corr(clean_r, method="pearson")
        s_ic = clean_f.corr(clean_r, method="spearman")

        def calc_stats(series: pd.Series, overall_val: float):
            if series.empty or len(series) < 2:
                mean_v = overall_val if pd.notna(overall_val) else 0.0
                std_v = 0.0
            else:
                mean_v = float(series.mean())
                std_v = float(series.std())

            n = len(series) if not series.empty else 1
            ir = (mean_v / std_v * math.sqrt(252)) if std_v > 1e-9 else 0.0
            t_stat = (mean_v / (std_v / math.sqrt(n))) if std_v > 1e-9 and n > 1 else 0.0
            pos_ratio = (series > 0).mean() if not series.empty else (1.0 if overall_val > 0 else 0.0)

            return {
                "overall": float(overall_val) if pd.notna(overall_val) else 0.0,
                "mean": mean_v,
                "std": std_v,
                "ir": ir,
                "t_stat": t_stat,
                "pos_ratio": float(pos_ratio),
            }

        result["pearson"] = calc_stats(rolling_pearson, p_ic)
        result["spearman"] = calc_stats(rolling_spearman, s_ic)

        # 3. 因子衰减分析 (Lag IC Decay)
        # 优先保持 entry/exit/return_type 不变，仅改变 horizon。
        if isinstance(price_data, pd.DataFrame) and target_spec is not None:
            normalized_target = (
                target_spec
                if isinstance(target_spec, ForwardReturnTarget)
                else ForwardReturnTarget.from_config(target_spec)
            )
            for lag in lags:
                lag_ret = normalized_target.with_horizon(lag).build(price_data)
                mask_k = factor_values.notna() & lag_ret.notna()
                if mask_k.sum() >= 10:
                    ric_k = factor_values[mask_k].corr(
                        lag_ret[mask_k],
                        method="spearman",
                    )
                    result["decay"][f"Lag_{lag}"] = (
                        float(ric_k) if pd.notna(ric_k) else 0.0
                    )
                else:
                    result["decay"][f"Lag_{lag}"] = 0.0
        elif close_prices is not None and isinstance(close_prices, pd.Series):
            for lag in lags:
                lag_ret = close_prices.pct_change(lag).shift(-lag)
                mask_k = factor_values.notna() & lag_ret.notna()
                if mask_k.sum() >= 10:
                    ric_k = factor_values[mask_k].corr(lag_ret[mask_k], method="spearman")
                    result["decay"][f"Lag_{lag}"] = float(ric_k) if pd.notna(ric_k) else 0.0
                else:
                    result["decay"][f"Lag_{lag}"] = 0.0
        else:
            # 降级备用：以 returns.shift(-(lag-1)) 替代
            for lag in lags:
                lag_r = returns.shift(-(lag - 1))
                mask_k = factor_values.notna() & lag_r.notna()
                if mask_k.sum() >= 10:
                    ric_k = factor_values[mask_k].corr(lag_r[mask_k], method="spearman")
                    result["decay"][f"Lag_{lag}"] = float(ric_k) if pd.notna(ric_k) else 0.0
                else:
                    result["decay"][f"Lag_{lag}"] = 0.0

        # 4. 5-Quantile 分组收益分析
        try:
            quantiles = pd.qcut(clean_f, q=n_quantiles, labels=False, duplicates="drop")
            q_df = pd.DataFrame({"quantile": quantiles, "return": clean_r})
            q_returns = q_df.groupby("quantile")["return"].mean().to_dict()
            
            q_dict = {}
            for q_idx in range(n_quantiles):
                q_dict[f"Q{q_idx+1}"] = float(q_returns.get(q_idx, 0.0))
            
            # 结算 Long-Short 多空价差 (Q5 - Q1)
            q_first = q_returns.get(0, 0.0)
            q_last = q_returns.get(n_quantiles - 1, 0.0)
            q_dict["Long_Short"] = float(q_last - q_first)

            result["quantiles"] = q_dict
        except Exception as e:
            logger.debug("Quantile calculation failed: %s", e)
            result["quantiles"] = {f"Q{i+1}": 0.0 for i in range(n_quantiles)}
            result["quantiles"]["Long_Short"] = 0.0

        # 5. 因子换手率 (Turnover)
        diff_abs = clean_f.diff().abs().mean()
        denom = clean_f.abs().mean() + 1e-9
        result["turnover"] = float(diff_abs / denom) if pd.notna(diff_abs) else 0.0

        return result

    @staticmethod
    def _evaluate_cross_asset(
        factor_values: pd.DataFrame,
        returns: pd.DataFrame,
        lags: List[int],
        n_quantiles: int,
    ) -> Dict[str, Any]:
        """Cross-Asset 截面评估逻辑"""
        valid_mask = ~factor_values.isna() & ~returns.isna()
        total_bars = factor_values.size
        valid_bars = int(valid_mask.sum().sum())
        coverage = valid_bars / total_bars if total_bars > 0 else 0.0

        # 截面 IC：每个时间点跨资产算 Spearman / Pearson
        p_ics = factor_values.corrwith(returns, axis=1, method="pearson").dropna()
        s_ics = factor_values.corrwith(returns, axis=1, method="spearman").dropna()

        def calc_cs_stats(series: pd.Series):
            if series.empty:
                return {"overall": 0.0, "mean": 0.0, "std": 0.0, "ir": 0.0, "t_stat": 0.0, "pos_ratio": 0.0}
            mean_v = float(series.mean())
            std_v = float(series.std())
            n = len(series)
            ir = (mean_v / std_v * math.sqrt(252)) if std_v > 1e-9 else 0.0
            t_stat = (mean_v / (std_v / math.sqrt(n))) if std_v > 1e-9 and n > 1 else 0.0
            pos_r = float((series > 0).mean())
            return {
                "overall": mean_v,
                "mean": mean_v,
                "std": std_v,
                "ir": ir,
                "t_stat": t_stat,
                "pos_ratio": pos_r,
            }

        turnover = factor_values.diff().abs().mean().mean()

        decay = {}
        for lag in lags:
            lag_r = returns.shift(-(lag - 1))
            cs_decay = factor_values.corrwith(lag_r, axis=1, method="spearman").mean()
            decay[f"Lag_{lag}"] = float(cs_decay) if pd.notna(cs_decay) else 0.0

        return {
            "total_bars": total_bars,
            "valid_bars": valid_bars,
            "coverage": coverage,
            "pearson": calc_cs_stats(p_ics),
            "spearman": calc_cs_stats(s_ics),
            "decay": decay,
            "quantiles": {"Q1": 0.0, "Q5": 0.0, "Long_Short": 0.0},
            "turnover": float(turnover) if pd.notna(turnover) else 0.0,
        }
