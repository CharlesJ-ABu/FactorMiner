"""Evaluate recorded LLM candidates on the configured untouched test window."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from core.data_feed.real_client import RealDataClient
from core.inspector.resolver import FactorResolver
from core.storage.factor_storage import LocalFactorStorage
from user_workspace.experiment_tools.summarize_llm_run import resolve_run_dir


def coverage(value: pd.Series) -> float:
    numeric = value.to_numpy(dtype=float, na_value=np.nan)
    return float(np.isfinite(numeric).sum() / numeric.size) if numeric.size else 0.0


def metrics(factor: pd.Series, returns: pd.Series) -> dict[str, float]:
    if not factor.index.equals(returns.index):
        raise ValueError("Factor and return indexes do not match")
    valid = pd.concat(
        [factor.rename("factor"), returns.rename("return")],
        axis=1,
    ).replace([np.inf, -np.inf], np.nan).dropna()
    turnover = float(factor.diff().abs().mean())
    result = {
        "coverage": coverage(factor),
        "observations": int(len(valid)),
        "IC": float(valid["factor"].corr(valid["return"], method="pearson")),
        "RankIC": float(valid["factor"].corr(valid["return"], method="spearman")),
        "Turnover": turnover,
    }
    result["fitness_score"] = abs(result["RankIC"]) * 100.0
    return result


def stored_factor_map(candidate_ids: set[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for metadata_file in (PROJECT_ROOT / "factor_db" / "metadata").glob("*.json"):
        payload = json.loads(metadata_file.read_text(encoding="utf-8"))
        provenance = payload.get("logic_reference", {}).get("provenance", {})
        candidate_id = provenance.get("candidate_id")
        if candidate_id in candidate_ids:
            result[candidate_id] = payload["factor_id"]
    return result


def read_candidate_rows(run_dir: Path) -> list[dict[str, str]]:
    with (run_dir / "candidate_results.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        return list(csv.DictReader(handle))


def semantic_flags(code: str, timeframe: str) -> list[str]:
    lowered = code.lower()
    flags: list[str] = []
    if timeframe.endswith("m") and any(
        phrase in lowered for phrase in ("daily", "day's", "10-day", "across stocks")
    ):
        flags.append("comment_timeframe_or_universe_mismatch")
    if "shift(10)" in lowered and "10-day" in lowered and timeframe == "1m":
        flags.append("shift_10_is_10_minutes_not_10_days")
    return flags


def audit(run_path: Path, config_path: Path) -> Path:
    run_dir = resolve_run_dir(run_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    data_client = RealDataClient(config)
    train_data = data_client.get_data()
    train_returns = data_client.get_returns()
    test_data = data_client.get_test_data()
    test_returns = data_client.get_test_returns()

    candidate_rows = read_candidate_rows(run_dir)
    candidate_ids = {row["candidate_id"] for row in candidate_rows}
    factor_map = stored_factor_map(candidate_ids)
    resolver = FactorResolver(storage=LocalFactorStorage())
    timeframe = config.get("data_feeds", {}).get("timeframe", "")

    audit_rows: list[dict[str, Any]] = []
    test_outputs: dict[str, pd.Series] = {}
    for row in candidate_rows:
        candidate_id = row["candidate_id"]
        factor_id = factor_map.get(candidate_id)
        if not factor_id:
            raise ValueError(f"No persisted Factor ID found for candidate {candidate_id}")
        expression = resolver.resolve(factor_id=factor_id)
        train_factor = expression.compute(train_data)
        test_factor = expression.compute(test_data)
        train_metrics = metrics(train_factor, train_returns)
        test_metrics = metrics(test_factor, test_returns)
        test_outputs[candidate_id] = test_factor
        flags = semantic_flags(row["code"], timeframe)
        audit_rows.append(
            {
                "candidate_id": candidate_id,
                "factor_id": factor_id,
                "epoch": int(row["epoch"]),
                "train_IC": train_metrics["IC"],
                "train_RankIC": train_metrics["RankIC"],
                "train_coverage": train_metrics["coverage"],
                "test_IC": test_metrics["IC"],
                "test_RankIC": test_metrics["RankIC"],
                "test_coverage": test_metrics["coverage"],
                "test_Turnover": test_metrics["Turnover"],
                "rankic_sign_preserved": bool(
                    np.sign(train_metrics["RankIC"]) == np.sign(test_metrics["RankIC"])
                ),
                "semantic_flags": "|".join(flags),
                "code": row["code"],
                "output_sha256": row["output_sha256"],
            }
        )

    audit_frame = pd.DataFrame(audit_rows).sort_values(
        "test_RankIC", ascending=False
    )
    audit_frame.to_csv(run_dir / "holdout_results.csv", index=False)

    output_frame = pd.DataFrame(test_outputs)
    similarity = output_frame.corr(method="spearman")
    similarity.to_csv(run_dir / "holdout_similarity_spearman.csv")

    exact_groups: dict[str, list[str]] = defaultdict(list)
    for row in audit_rows:
        exact_groups[row["output_sha256"]].append(row["candidate_id"])
    repeated_groups = [
        members for members in exact_groups.values() if len(members) > 1
    ]
    upper = similarity.where(
        np.triu(np.ones(similarity.shape), k=1).astype(bool)
    )
    highly_similar_pairs = []
    for left in upper.index:
        for right in upper.columns:
            value = upper.loc[left, right]
            if pd.notna(value) and abs(float(value)) >= 0.95:
                highly_similar_pairs.append(
                    {
                        "left": left,
                        "right": right,
                        "spearman": float(value),
                    }
                )

    summary = {
        "run_id": json.loads(
            (run_dir / "manifest.json").read_text(encoding="utf-8")
        ).get("run_id"),
        "candidate_count": len(audit_rows),
        "persisted_factor_count": len(factor_map),
        "exact_output_group_count": len(exact_groups),
        "repeated_exact_output_groups": repeated_groups,
        "high_similarity_pair_count_abs_ge_0_95": len(highly_similar_pairs),
        "highly_similar_pairs": highly_similar_pairs,
        "semantic_flagged_candidates": [
            row["candidate_id"] for row in audit_rows if row["semantic_flags"]
        ],
        "test_rankic_positive_count": int((audit_frame["test_RankIC"] > 0).sum()),
        "rankic_sign_preserved_count": int(
            audit_frame["rankic_sign_preserved"].sum()
        ),
        "best_test_candidate": audit_frame.iloc[0][
            ["candidate_id", "factor_id", "test_IC", "test_RankIC"]
        ].to_dict(),
    }
    (run_dir / "holdout_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    report_rows = [
        f"| {row.factor_id} | {row.epoch} | {row.train_RankIC:.4f} | "
        f"{row.test_RankIC:.4f} | {row.test_coverage:.2%} | "
        f"{'是' if row.rankic_sign_preserved else '否'} |"
        for row in audit_frame.itertuples(index=False)
    ]
    report = "\n".join(
        [
            "# LLM 候选测试窗口复评",
            "",
            "## 口径",
            "",
            f"- 训练窗口：{config['data_feeds']['mine_period']}",
            f"- 测试窗口：{config['data_feeds']['test_period']}",
            "- 标签：下一根 1 分钟 K 线收益。",
            "- 测试窗口没有参与 LLM Prompt 或 Reflection。",
            "",
            "## 结果",
            "",
            "| Factor | 轮次 | 训练 RankIC | 测试 RankIC | 测试覆盖率 | 符号保持 |",
            "|---|---:|---:|---:|---:|---:|",
            *report_rows,
            "",
            "## 同质化与语义检查",
            "",
            f"- 9 个代码候选只形成 {len(exact_groups)} 组不同训练输出。",
            f"- 完全重复输出组：{len(repeated_groups)} 组。",
            f"- 测试期 |Spearman|≥0.95 的候选对：{len(highly_similar_pairs)} 对。",
            f"- 注释与 1m/BTC 实验口径冲突的候选："
            f"{len(summary['semantic_flagged_candidates'])} 个。",
            "",
            "## 边界",
            "",
            "- 这是固定测试窗口复评，不是交易策略回测。",
            "- 没有手续费、滑点、资金费率或组合构建。",
            "- 多段代码产生相同输出时，不能因为注释不同就视为独立发现。",
            "",
        ]
    )
    (run_dir / "holdout_report.md").write_text(report, encoding="utf-8")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_path", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "user_workspace" / "configs" / "configLLM_experiment.json",
    )
    args = parser.parse_args()
    print(audit(args.run_path, args.config.resolve()))


if __name__ == "__main__":
    main()
