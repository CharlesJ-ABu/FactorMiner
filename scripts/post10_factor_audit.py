"""Run the Post 10 multi-asset audit for the saved Temporal NN winner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from user_workspace.custom_miners.my_custom_nn_temporal import TemporalICFactorModel


FACTOR_ID = "fac_1151729b"
MODEL_FILE = "factor_db/models/nn_fdc0914ec631a88e.npz"
SNAPSHOT_FILE = "factor_db/values/fac_1151729b.parquet"
ASSETS = ("BTC", "ETH", "SOL", "BNB", "XRP")
START = pd.Timestamp("2025-08-02 00:00:00")
END = pd.Timestamp("2025-08-15 00:00:00")
HORIZONS = (1, 3, 5, 10, 15, 30, 60)
DELAYS = (0, 1, 3, 5, 10, 15, 30, 60)
QUANTILES = 5


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def load_market_data(project_root: Path, asset: str) -> pd.DataFrame:
    path = (
        project_root
        / "data"
        / "binance"
        / "futures"
        / f"{asset}_USDT_USDT-1m-futures.feather"
    )
    frame = pd.read_feather(path)
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.set_index("date")
    else:
        frame.index = pd.to_datetime(frame.index, errors="coerce")
    if frame.index.tz is not None:
        frame.index = frame.index.tz_convert("UTC").tz_localize(None)
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    frame = frame.loc[(frame.index >= START) & (frame.index <= END)].copy()
    numeric = ["open", "high", "low", "close", "volume"]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    return frame


def daily_stat_ci(values: pd.Series, rng: np.random.Generator) -> tuple[float, float]:
    clean = values.dropna().to_numpy(dtype=float)
    if len(clean) < 2:
        return (float("nan"), float("nan"))
    samples = rng.choice(clean, size=(2000, len(clean)), replace=True).mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def corr(frame: pd.DataFrame, method: str = "spearman") -> float:
    valid = frame.dropna(subset=["factor", "return"])
    if len(valid) < 10 or valid["factor"].nunique() < 2 or valid["return"].nunique() < 2:
        return float("nan")
    return float(valid["factor"].corr(valid["return"], method=method))


def quantile_table(frame: pd.DataFrame) -> pd.DataFrame:
    valid = frame.dropna(subset=["factor", "return"]).copy()
    ranked = valid["factor"].rank(method="first")
    valid["quantile"] = pd.qcut(
        ranked, q=QUANTILES, labels=range(1, QUANTILES + 1)
    ).astype(int)
    result = (
        valid.groupby("quantile", observed=True)["return"]
        .agg(["mean", "count"])
        .reset_index()
    )
    return result


def daily_quantile_spread(frame: pd.DataFrame) -> pd.Series:
    rows: dict[pd.Timestamp, float] = {}
    for day, group in frame.groupby(frame.index.floor("D")):
        try:
            table = quantile_table(group)
        except ValueError:
            continue
        means = table.set_index("quantile")["mean"]
        if 1 in means and QUANTILES in means:
            rows[day] = float(means.loc[QUANTILES] - means.loc[1])
    return pd.Series(rows, dtype=float)


def hourly_ic(frame: pd.DataFrame) -> pd.Series:
    values: dict[pd.Timestamp, float] = {}
    for hour, group in frame.groupby(frame.index.floor("h")):
        value = corr(group)
        if np.isfinite(value):
            values[hour] = value
    return pd.Series(values, dtype=float).sort_index()


def subperiod_rows(asset: str, frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    boundaries = np.linspace(0, len(frame), 5, dtype=int)
    for idx in range(4):
        part = frame.iloc[boundaries[idx] : boundaries[idx + 1]]
        rows.append(
            {
                "asset": asset,
                "block": idx + 1,
                "start": part.index.min(),
                "end": part.index.max(),
                "observations": int(part[["factor", "return"]].dropna().shape[0]),
                "rank_ic_5m": corr(part),
            }
        )
    return rows


def regime_rows(asset: str, frame: pd.DataFrame) -> list[dict[str, Any]]:
    work = frame.copy()
    work["trend_240m"] = work["close"].pct_change(240)
    work["vol_240m"] = work["close"].pct_change().rolling(240, min_periods=120).std()
    vol_median = work["vol_240m"].median()
    masks = {
        "uptrend_240m": work["trend_240m"] >= 0,
        "downtrend_240m": work["trend_240m"] < 0,
        "high_vol_240m": work["vol_240m"] >= vol_median,
        "low_vol_240m": work["vol_240m"] < vol_median,
    }
    rows = []
    for regime, mask in masks.items():
        part = work.loc[mask]
        rows.append(
            {
                "asset": asset,
                "regime": regime,
                "observations": int(part[["factor", "return"]].dropna().shape[0]),
                "rank_ic_5m": corr(part),
            }
        )
    return rows


def audit_asset(
    asset: str,
    market: pd.DataFrame,
    model: TemporalICFactorModel,
    channel: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    factor = model.predict_channel(market, channel).rename("factor")
    work = market.copy()
    work["factor"] = factor
    for horizon in HORIZONS:
        work[f"return_{horizon}m"] = work["close"].shift(-horizon) / work["close"] - 1.0
    for delay in DELAYS:
        work[f"delayed_return_{delay}m"] = (
            work["close"].shift(-(delay + 5)) / work["close"].shift(-delay) - 1.0
        )
    work["return"] = work["return_5m"]

    total = len(work)
    valid = work[["factor", "return"]].dropna()
    coverage = len(valid) / total if total else 0.0
    daily_ic = valid.groupby(valid.index.floor("D")).apply(
        lambda group: corr(group), include_groups=False
    )
    daily_low, daily_high = daily_stat_ci(daily_ic, rng)
    hour_ic = hourly_ic(valid)
    rolling_24h = hour_ic.rolling(24, min_periods=12).mean()
    q_table = quantile_table(valid)
    q_means = q_table.set_index("quantile")["mean"]
    spread = float(q_means.loc[QUANTILES] - q_means.loc[1])
    q_monotonicity = float(
        pd.Series(q_means.index, dtype=float).corr(
            pd.Series(q_means.to_numpy(), dtype=float), method="spearman"
        )
    )
    daily_spread = daily_quantile_spread(valid)
    spread_low, spread_high = daily_stat_ci(daily_spread, rng)

    factor_valid = work["factor"].dropna()
    turnover_raw = float(factor_valid.diff().abs().mean())
    turnover = float(turnover_raw / (factor_valid.abs().mean() + 1e-12))
    sign_flip = float(
        (np.sign(factor_valid).diff().abs() == 2).mean()
    )
    autocorr = {
        lag: float(factor_valid.autocorr(lag=lag))
        for lag in (1, 5, 15, 60)
    }

    engineered = model.adapter.engineer_features(market)
    feature_z = (engineered - model.feature_mean) / model.feature_std
    finite_z = feature_z.replace([np.inf, -np.inf], np.nan)
    shift_rate = float((finite_z.abs() > 5).sum().sum() / finite_z.notna().sum().sum())

    horizon_rows = []
    for horizon in HORIZONS:
        sample = work[["factor", f"return_{horizon}m"]].rename(
            columns={f"return_{horizon}m": "return"}
        )
        horizon_rows.append(
            {
                "asset": asset,
                "horizon_minutes": horizon,
                "pearson_ic": corr(sample, "pearson"),
                "rank_ic": corr(sample, "spearman"),
                "observations": int(sample.dropna().shape[0]),
            }
        )

    delay_rows = []
    for delay in DELAYS:
        sample = work[["factor", f"delayed_return_{delay}m"]].rename(
            columns={f"delayed_return_{delay}m": "return"}
        )
        delay_rows.append(
            {
                "asset": asset,
                "delay_minutes": delay,
                "forward_window_minutes": 5,
                "pearson_ic": corr(sample, "pearson"),
                "rank_ic": corr(sample, "spearman"),
                "observations": int(sample.dropna().shape[0]),
            }
        )

    quantile_rows = [
        {
            "asset": asset,
            "quantile": int(row.quantile),
            "mean_forward_return_5m": float(row.mean),
            "observations": int(row.count),
        }
        for row in q_table.itertuples(index=False)
    ]
    rolling_rows = [
        {
            "asset": asset,
            "timestamp": timestamp,
            "hourly_rank_ic": finite(hour_ic.get(timestamp)),
            "rolling_24h_mean_rank_ic": finite(value),
        }
        for timestamp, value in rolling_24h.items()
    ]
    summary = {
        "asset": asset,
        "bars": total,
        "start": work.index.min().isoformat(),
        "end": work.index.max().isoformat(),
        "coverage": coverage,
        "pearson_ic_5m": corr(valid, "pearson"),
        "rank_ic_5m": corr(valid, "spearman"),
        "daily_rank_ic_mean": float(daily_ic.mean()),
        "daily_rank_ic_median": float(daily_ic.median()),
        "daily_rank_ic_positive_ratio": float((daily_ic > 0).mean()),
        "daily_rank_ic_ci_low": daily_low,
        "daily_rank_ic_ci_high": daily_high,
        "rolling_24h_positive_ratio": float((rolling_24h.dropna() > 0).mean()),
        "rolling_24h_min": float(rolling_24h.min()),
        "rolling_24h_max": float(rolling_24h.max()),
        "quantile_spread_q5_minus_q1": spread,
        "quantile_spread_daily_ci_low": spread_low,
        "quantile_spread_daily_ci_high": spread_high,
        "quantile_monotonicity_spearman": q_monotonicity,
        "turnover_raw_1m": turnover_raw,
        "turnover_normalized_1m": turnover,
        "sign_flip_rate_1m": sign_flip,
        "factor_autocorr_1m": autocorr[1],
        "factor_autocorr_5m": autocorr[5],
        "factor_autocorr_15m": autocorr[15],
        "factor_autocorr_60m": autocorr[60],
        "feature_abs_z_gt_5_rate": shift_rate,
    }
    return {
        "summary": summary,
        "horizons": horizon_rows,
        "delays": delay_rows,
        "quantiles": quantile_rows,
        "rolling": rolling_rows,
        "subperiods": subperiod_rows(asset, valid),
        "regimes": regime_rows(asset, work),
        "work": work,
    }


def reproduce_snapshot(project_root: Path, btc_work: pd.DataFrame) -> dict[str, Any]:
    snapshot = pd.read_parquet(project_root / SNAPSHOT_FILE).copy()
    snapshot["timestamp"] = pd.to_datetime(snapshot["timestamp"], errors="coerce")
    snapshot = snapshot.set_index("timestamp").sort_index()
    joined = snapshot[["factor", "forward_return"]].join(
        btc_work[["factor", "return_5m"]],
        how="inner",
        lsuffix="_snapshot",
        rsuffix="_replay",
    ).dropna()
    factor_diff = (joined["factor_snapshot"] - joined["factor_replay"]).abs()
    return_diff = (joined["forward_return"] - joined["return_5m"]).abs()
    return {
        "snapshot_rows": int(len(snapshot)),
        "joined_rows": int(len(joined)),
        "factor_max_abs_diff": float(factor_diff.max()),
        "factor_mean_abs_diff": float(factor_diff.mean()),
        "forward_return_max_abs_diff": float(return_diff.max()),
        "factor_equivalent_within_1e_10": bool((factor_diff <= 1e-10).all()),
    }


def gate_decisions(summary: pd.DataFrame, delays: pd.DataFrame, subperiods: pd.DataFrame) -> list[dict[str, str]]:
    coverage_ok = bool((summary["coverage"] >= 0.99).all())
    rolling_ok = bool(
        (summary["rank_ic_5m"] > 0).all()
        and (summary["daily_rank_ic_positive_ratio"] >= 0.60).all()
        and (subperiods["rank_ic_5m"] > 0).all()
    )
    delay_short = delays.loc[delays["delay_minutes"].isin([0, 5, 10])]
    delay_ok = bool((delay_short["rank_ic"] > 0).all())
    quantile_ok = bool(
        (summary["quantile_spread_q5_minus_q1"] > 0).all()
        and (summary["quantile_monotonicity_spearman"] >= 0.5).all()
    )
    return [
        {
            "gate": "覆盖率",
            "status": "通过" if coverage_ok else "存疑",
            "rule": "五个标的有效覆盖率均不低于 99%",
        },
        {
            "gate": "滚动 IC 稳定性",
            "status": "通过" if rolling_ok else "未通过",
            "rule": "各标的总体 RankIC>0、日 IC 正值比例≥60%、四分段均为正",
        },
        {
            "gate": "IC 延迟衰减",
            "status": "条件通过" if delay_ok else "未通过",
            "rule": "固定 5m 未来收益窗口延后 0/5/10 分钟后，各标的 RankIC 均不翻负",
        },
        {
            "gate": "分位收益",
            "status": "通过" if quantile_ok else "未通过",
            "rule": "各标的 Q5-Q1>0 且五分位单调性 Spearman≥0.5",
        },
        {
            "gate": "换手率",
            "status": "仅描述",
            "rule": "缺少手续费/滑点模型，不把信号换手直接判为可交易",
        },
    ]


def save_charts(
    output: Path,
    summary: pd.DataFrame,
    horizons: pd.DataFrame,
    delays: pd.DataFrame,
    quantiles: pd.DataFrame,
    rolling: pd.DataFrame,
    regimes: pd.DataFrame,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = ["#1d4ed8", "#0891b2", "#0f766e", "#d97706", "#dc2626"]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(summary))
    ax.bar(x - 0.18, summary["pearson_ic_5m"], 0.36, label="Pearson IC", color="#1d4ed8")
    ax.bar(x + 0.18, summary["rank_ic_5m"], 0.36, label="RankIC", color="#f97316")
    ax.axhline(0, color="#111827", linewidth=0.8)
    ax.set_xticks(x, summary["asset"])
    ax.set_ylabel("5-minute forward correlation")
    ax.set_title("Zero-shot multi-asset IC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "01_cross_asset_ic.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for color, (asset, group) in zip(colors, horizons.groupby("asset", sort=False)):
        ax.plot(group["horizon_minutes"], group["rank_ic"], marker="o", label=asset, color=color)
    ax.axhline(0, color="#111827", linewidth=0.8)
    ax.set_xlabel("Forward horizon (minutes)")
    ax.set_ylabel("RankIC")
    ax.set_title("True forward-horizon decay")
    ax.legend(ncol=5)
    fig.tight_layout()
    fig.savefig(output / "02_horizon_decay.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for color, (asset, group) in zip(colors, delays.groupby("asset", sort=False)):
        ax.plot(group["delay_minutes"], group["rank_ic"], marker="o", label=asset, color=color)
    ax.axhline(0, color="#111827", linewidth=0.8)
    ax.set_xlabel("Signal delay before the fixed 5-minute return window")
    ax.set_ylabel("RankIC")
    ax.set_title("Fixed-window IC delay decay")
    ax.legend(ncol=5)
    fig.tight_layout()
    fig.savefig(output / "02b_delay_decay.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    for color, (asset, group) in zip(colors, rolling.groupby("asset", sort=False)):
        ax.plot(
            pd.to_datetime(group["timestamp"]),
            group["rolling_24h_mean_rank_ic"],
            label=asset,
            color=color,
            linewidth=1.5,
        )
    ax.axhline(0, color="#111827", linewidth=0.8)
    ax.set_ylabel("24-hour mean of hourly RankIC")
    ax.set_title("Rolling IC stability")
    ax.legend(ncol=5)
    fig.tight_layout()
    fig.savefig(output / "03_rolling_ic.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for color, (asset, group) in zip(colors, quantiles.groupby("asset", sort=False)):
        ax.plot(
            group["quantile"],
            group["mean_forward_return_5m"] * 10_000,
            marker="o",
            label=asset,
            color=color,
        )
    ax.axhline(0, color="#111827", linewidth=0.8)
    ax.set_xticks(range(1, QUANTILES + 1))
    ax.set_xlabel("Factor quantile")
    ax.set_ylabel("Mean 5-minute forward return (bp)")
    ax.set_title("Quantile return shape")
    ax.legend(ncol=5)
    fig.tight_layout()
    fig.savefig(output / "04_quantile_returns.png", dpi=180)
    plt.close(fig)

    pivot = regimes.pivot(index="asset", columns="regime", values="rank_ic_5m")
    fig, ax = plt.subplots(figsize=(10, 5))
    image = ax.imshow(pivot.to_numpy(), cmap="RdYlBu", aspect="auto", vmin=-0.08, vmax=0.08)
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=25, ha="right")
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    for row in range(len(pivot.index)):
        for col in range(len(pivot.columns)):
            value = pivot.iloc[row, col]
            ax.text(col, row, f"{value:.3f}", ha="center", va="center", fontsize=9)
    ax.set_title("5-minute RankIC by market regime")
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(output / "05_regime_heatmap.png", dpi=180)
    plt.close(fig)


def markdown_report(
    summary: pd.DataFrame,
    horizons: pd.DataFrame,
    gates: list[dict[str, str]],
    reproduction: dict[str, Any],
) -> str:
    summary_rows = []
    for row in summary.itertuples(index=False):
        summary_rows.append(
            f"| {row.asset} | {row.coverage:.2%} | {row.pearson_ic_5m:.4f} | "
            f"{row.rank_ic_5m:.4f} | {row.daily_rank_ic_positive_ratio:.1%} | "
            f"{row.quantile_spread_q5_minus_q1 * 10_000:.3f} | "
            f"{row.quantile_monotonicity_spearman:.2f} | "
            f"{row.turnover_raw_1m:.3f} / {row.turnover_normalized_1m:.3f} |"
        )
    gate_rows = [
        f"| {item['gate']} | {item['status']} | {item['rule']} |" for item in gates
    ]
    decay_lines = []
    for asset, group in horizons.groupby("asset", sort=False):
        pairs = ", ".join(
            f"{int(row.horizon_minutes)}m={row.rank_ic:.4f}"
            for row in group.itertuples(index=False)
        )
        decay_lines.append(f"- {asset}: {pairs}")

    return "\n".join(
        [
            "# 第 10 篇研究底稿：IC 0.0335 之后的五道审查",
            "",
            "## 审查口径",
            "",
            f"- 因子：`{FACTOR_ID}`，Temporal NN 已保存模型，固定通道 4。",
            "- 训练来源：BTC/USDT:USDT；ETH、SOL、BNB、XRP 均为零微调迁移，不是重新训练结果。",
            f"- 统一窗口：{START} 至 {END}，Binance U 本位永续，1 分钟。",
            "- 标签：真实未来 1/3/5/10/15/30/60 分钟收益；主口径为未来 5 分钟。",
            "- 不包含手续费、滑点、资金费率与组合净值，因此换手率只做描述，不作可交易结论。",
            "",
            "## 快照复现",
            "",
            f"- 原快照 {reproduction['snapshot_rows']:,} 行；时间对齐后 {reproduction['joined_rows']:,} 行。",
            f"- 模型回放因子最大绝对误差：{reproduction['factor_max_abs_diff']:.3e}。",
            f"- 未来收益最大绝对误差：{reproduction['forward_return_max_abs_diff']:.3e}。",
            f"- 因子 1e-10 精度内数值等价：{reproduction['factor_equivalent_within_1e_10']}。",
            "",
            "## 五关结论",
            "",
            "| 审查关卡 | 结论 | 预先固定的判定规则 |",
            "|---|---|---|",
            *gate_rows,
            "",
            "## 跨标的主结果",
            "",
            "| 标的 | 覆盖率 | Pearson IC | RankIC | 日 IC 正值比例 | Q5-Q1 (bp) | 分位单调性 | 1m 原始/归一化换手 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
            *summary_rows,
            "",
            "## 预测周期衰减（RankIC）",
            "",
            *decay_lines,
            "",
            "## 解释边界",
            "",
            "- 这是候选因子审查，不是收益回测。",
            "- 多标的迁移可以检验特征模式的可迁移性，但不能替代 walk-forward 或真正未触碰的后续留出集。",
            "- 日 IC 置信区间按日重采样，避免把 1 分钟观测误当成完全独立样本。",
            "- 分位结果按全窗口排序，同时另存逐日 Q5-Q1 置信区间。",
            "- 换手率没有交易成本模型配合时，不应写成“已经可落地”。",
            "",
            "## 图表",
            "",
            "- `01_cross_asset_ic.png`：跨标的 IC。",
            "- `02_horizon_decay.png`：不同累计未来周期响应。",
            "- `02b_delay_decay.png`：固定 5 分钟窗口的真正延迟衰减。",
            "- `03_rolling_ic.png`：小时 RankIC 的 24 小时滚动均值。",
            "- `04_quantile_returns.png`：五分位未来收益形状。",
            "- `05_regime_heatmap.png`：趋势与波动状态下的 RankIC。",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    project_root = PROJECT_ROOT
    rng = np.random.default_rng(20260724)

    payload = (project_root / MODEL_FILE).read_bytes()
    model = TemporalICFactorModel.from_artifact(payload)
    channel = 4

    results = []
    for asset in ASSETS:
        market = load_market_data(project_root, asset)
        if len(market) != 18_721:
            raise RuntimeError(
                f"{asset} expected 18,721 bars in the audit window, found {len(market)}"
            )
        results.append(audit_asset(asset, market, model, channel, rng))
        model.clear_prediction_cache()

    summary = pd.DataFrame(item["summary"] for item in results)
    horizons = pd.DataFrame(row for item in results for row in item["horizons"])
    delays = pd.DataFrame(row for item in results for row in item["delays"])
    quantiles = pd.DataFrame(row for item in results for row in item["quantiles"])
    rolling = pd.DataFrame(row for item in results for row in item["rolling"])
    subperiods = pd.DataFrame(row for item in results for row in item["subperiods"])
    regimes = pd.DataFrame(row for item in results for row in item["regimes"])
    reproduction = reproduce_snapshot(project_root, results[0]["work"])
    gates = gate_decisions(summary, delays, subperiods)

    summary.to_csv(output / "asset_summary.csv", index=False)
    horizons.to_csv(output / "horizon_decay.csv", index=False)
    delays.to_csv(output / "delay_decay.csv", index=False)
    quantiles.to_csv(output / "quantile_returns.csv", index=False)
    rolling.to_csv(output / "rolling_ic.csv", index=False)
    subperiods.to_csv(output / "subperiod_stability.csv", index=False)
    regimes.to_csv(output / "regime_stability.csv", index=False)
    (output / "audit_summary.json").write_text(
        json.dumps(
            {
                "factor_id": FACTOR_ID,
                "model_file": MODEL_FILE,
                "channel": channel,
                "window": [START.isoformat(), END.isoformat()],
                "assets": list(ASSETS),
                "reproduction": reproduction,
                "gates": gates,
                "asset_summary": summary.to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    (output / "research_report.md").write_text(
        markdown_report(summary, horizons, gates, reproduction),
        encoding="utf-8",
    )
    save_charts(output, summary, horizons, delays, quantiles, rolling, regimes)
    print(summary.to_string(index=False))
    print(json.dumps({"gates": gates, "reproduction": reproduction}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
