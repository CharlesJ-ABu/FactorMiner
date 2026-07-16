"""Compute Inspector Phase II analytics from persisted factor snapshots only."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


class SnapshotAnalysisError(ValueError):
    """Raised when a stored factor snapshot cannot support a real analysis."""


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _points(series: pd.Series) -> list[Dict[str, Any]]:
    return [
        {"timestamp": pd.Timestamp(timestamp).isoformat(), "value": value}
        for timestamp, raw_value in series.items()
        if (value := _finite_number(raw_value)) is not None
    ]


def _quantile_labels(values: pd.Series, count: int = 5) -> pd.Series:
    ranked = values.rank(method="first")
    unique = ranked.nunique()
    buckets = min(count, int(unique))
    if buckets < 2:
        return pd.Series(index=values.index, dtype="Int64")
    return pd.qcut(ranked, q=buckets, labels=False, duplicates="drop").astype("Int64") + 1


def _sequential_analysis(frame: pd.DataFrame, window: int) -> Dict[str, Any]:
    ordered = frame.sort_values("timestamp").copy()
    ordered["rolling_ic"] = ordered["factor"].rolling(window, min_periods=max(5, window // 2)).corr(
        ordered["forward_return"]
    )
    turnover = ordered.set_index("timestamp")["factor"].diff().abs()
    labels = _quantile_labels(ordered["factor"])
    ordered["quantile"] = labels
    buckets = ordered.dropna(subset=["quantile"]).groupby("quantile")["forward_return"].agg(["mean", "count"])
    quantiles = [
        {
            "bucket": f"Q{int(bucket)}",
            "mean_return": _finite_number(row["mean"]) or 0.0,
            "count": int(row["count"]),
        }
        for bucket, row in buckets.iterrows()
    ]
    return {
        "mode": "sequential_single",
        "rolling_ic": _points(ordered.set_index("timestamp")["rolling_ic"]),
        "turnover": _points(turnover),
        "quantiles": quantiles,
    }


def _cross_asset_analysis(frame: pd.DataFrame, window: int) -> Dict[str, Any]:
    ordered = frame.sort_values(["timestamp", "asset"]).copy()
    ic_by_time: Dict[pd.Timestamp, float] = {}
    turnover_by_time: Dict[pd.Timestamp, float] = {}
    bucket_rows: list[pd.DataFrame] = []

    for timestamp, group in ordered.groupby("timestamp", sort=True):
        valid = group.dropna(subset=["factor", "forward_return"])
        if len(valid) >= 2 and valid["factor"].nunique() > 1 and valid["forward_return"].nunique() > 1:
            ic_by_time[timestamp] = valid["factor"].corr(valid["forward_return"])
        labels = _quantile_labels(valid["factor"])
        if not labels.empty:
            bucket_rows.append(valid.assign(quantile=labels))

    factor_matrix = ordered.pivot(index="timestamp", columns="asset", values="factor")
    turnover = factor_matrix.diff().abs().mean(axis=1)
    turnover_by_time.update(turnover.to_dict())
    raw_ic = pd.Series(ic_by_time, dtype=float).sort_index()
    rolling_ic = raw_ic.rolling(window, min_periods=max(3, window // 3)).mean()

    if bucket_rows:
        bucket_frame = pd.concat(bucket_rows)
        buckets = bucket_frame.groupby("quantile")["forward_return"].agg(["mean", "count"])
    else:
        buckets = pd.DataFrame(columns=["mean", "count"])
    quantiles = [
        {
            "bucket": f"Q{int(bucket)}",
            "mean_return": _finite_number(row["mean"]) or 0.0,
            "count": int(row["count"]),
        }
        for bucket, row in buckets.iterrows()
    ]
    return {
        "mode": "cross_asset",
        "rolling_ic": _points(rolling_ic),
        "turnover": _points(pd.Series(turnover_by_time).sort_index()),
        "quantiles": quantiles,
    }


def analyze_factor_snapshot(snapshot: pd.DataFrame, rolling_window: int = 20) -> Dict[str, Any]:
    """Return a JSON-safe, real-data Tearsheet payload for one factor."""
    required_columns = {"timestamp", "factor", "forward_return"}
    missing = required_columns.difference(snapshot.columns)
    if missing:
        raise SnapshotAnalysisError(f"Snapshot is missing required columns: {', '.join(sorted(missing))}.")

    frame = snapshot.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame["factor"] = pd.to_numeric(frame["factor"], errors="coerce")
    frame["forward_return"] = pd.to_numeric(frame["forward_return"], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "factor", "forward_return"])
    if frame.empty:
        raise SnapshotAnalysisError("Snapshot has no aligned finite factor/forward-return observations.")

    window = max(5, min(int(rolling_window), len(frame)))
    analysis = _cross_asset_analysis(frame, window) if "asset" in frame.columns else _sequential_analysis(frame, window)
    bucket_values = [item["mean_return"] for item in analysis["quantiles"]]
    analysis["summary"] = {
        "observations": int(len(frame)),
        "start": frame["timestamp"].min().isoformat(),
        "end": frame["timestamp"].max().isoformat(),
        "rolling_window": window,
        "latest_rolling_ic": analysis["rolling_ic"][-1]["value"] if analysis["rolling_ic"] else None,
        "mean_turnover": _finite_number(pd.Series([item["value"] for item in analysis["turnover"]]).mean()),
        "quantile_spread": (max(bucket_values) - min(bucket_values)) if len(bucket_values) >= 2 else None,
    }
    return analysis
