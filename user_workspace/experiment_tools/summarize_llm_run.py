"""Summarize one recorded LLM mining run into research-ready artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def resolve_run_dir(path: Path) -> Path:
    candidate = path.resolve()
    if (candidate / "manifest.json").is_file():
        return candidate
    runs = sorted(
        (
            child
            for child in candidate.iterdir()
            if child.is_dir() and (child / "manifest.json").is_file()
        ),
        key=lambda child: child.name,
    )
    if not runs:
        raise FileNotFoundError(f"No recorded run found under {candidate}")
    return runs[-1]


def load_events(run_dir: Path) -> list[dict[str, Any]]:
    event_file = run_dir / "events.jsonl"
    if not event_file.is_file():
        return []
    return [
        json.loads(line)
        for line in event_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def candidate_rows(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    for event in events:
        candidate_id = event.get("candidate_id")
        if not candidate_id:
            continue
        row = candidates.setdefault(
            candidate_id,
            {
                "candidate_id": candidate_id,
                "epoch": event.get("epoch"),
                "candidate_index": event.get("candidate_index"),
                "generation_source": "",
                "status": "generated",
                "code": "",
                "error_type": "",
                "error_message": "",
                "IC": "",
                "RankIC": "",
                "Turnover": "",
                "coverage": "",
                "fitness_score": "",
                "lines_of_code": "",
                "deterministic_replay_equal": "",
                "output_sha256": "",
            },
        )
        event_type = event.get("event_type")
        if event_type == "candidate_generated":
            row["candidate_index"] = event.get("candidate_index")
            row["generation_source"] = event.get("generation_source", "")
            row["code"] = event.get("extracted_code", "")
        elif event_type == "candidate_evaluated":
            row["status"] = event.get("status", "")
            row["code"] = event.get("code", row["code"])
            row["error_type"] = event.get("error_type", "")
            row["error_message"] = event.get("error_message", "")
            metrics = event.get("metrics", {})
            diagnostics = event.get("diagnostics", {})
            for key in ("IC", "RankIC", "Turnover", "fitness_score"):
                row[key] = metrics.get(key, "")
            for key in (
                "coverage",
                "lines_of_code",
                "deterministic_replay_equal",
                "output_sha256",
            ):
                row[key] = diagnostics.get(key, "")
        elif event_type == "candidate_filtered":
            row["status"] = "filtered"
            row["error_type"] = event.get("reason", "")
    return sorted(
        candidates.values(),
        key=lambda row: (
            row["epoch"] if row["epoch"] is not None else -1,
            row["candidate_index"] if row["candidate_index"] is not None else -1,
        ),
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else [
        "candidate_id",
        "epoch",
        "candidate_index",
        "generation_source",
        "status",
        "code",
        "error_type",
        "error_message",
        "IC",
        "RankIC",
        "Turnover",
        "coverage",
        "fitness_score",
        "lines_of_code",
        "deterministic_replay_equal",
        "output_sha256",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def markdown_report(
    manifest: dict[str, Any],
    events: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> str:
    event_counts = Counter(event.get("event_type", "unknown") for event in events)
    status_counts = Counter(row["status"] for row in rows)
    blocked = next(
        (
            event.get("reason", "")
            for event in events
            if event.get("event_type") == "run_blocked"
        ),
        "",
    )
    successful = [row for row in rows if row["status"] == "success"]
    successful.sort(
        key=lambda row: (
            float(row["fitness_score"])
            if row["fitness_score"] not in ("", None)
            else float("-inf")
        ),
        reverse=True,
    )
    table_rows = []
    for row in successful[:10]:
        table_rows.append(
            f"| {row['candidate_id']} | {row['epoch']} | "
            f"{float(row['IC']):.4f} | {float(row['RankIC']):.4f} | "
            f"{float(row['coverage']):.2%} | {float(row['fitness_score']):.4f} |"
        )
    if not table_rows:
        table_rows.append("| — | — | — | — | — | — |")
    feeds = manifest.get("data_feeds", {})
    llm = manifest.get("llm", {})
    lines = [
        "# LLM 因子挖掘实验底稿",
        "",
        "## 实验状态",
        "",
        f"- Run ID：`{manifest.get('run_id', 'legacy-run')}`",
        f"- 开始时间：{manifest.get('started_at', '')}",
        f"- 模型：`{llm.get('model', '')}`",
        f"- 标的：{', '.join(feeds.get('pairs', []))}",
        f"- 周期：{feeds.get('timeframe', '')}",
        f"- 候选状态：{dict(status_counts)}",
        f"- 事件计数：{dict(event_counts)}",
    ]
    if blocked:
        lines.extend(["", f"> 本次运行被阻塞：{blocked}"])
    lines.extend(
        [
            "",
            "## 成功候选",
            "",
            "| Candidate | 轮次 | IC | RankIC | 覆盖率 | Fitness |",
            "|---|---:|---:|---:|---:|---:|",
            *table_rows,
            "",
            "## 证据边界",
            "",
            "- `candidate_generated` 保存模型原始回答与提取后的代码。",
            "- `candidate_evaluated` 保存执行状态、错误或指标，以及确定性回放检查。",
            "- `reflection_updated` 保存进入下一轮 Prompt 的成功与失败记忆。",
            "- 候选因子不等于可交易策略；本报告不包含手续费、滑点或组合回测。",
            "",
        ]
    )
    return "\n".join(lines)


def summarize(path: Path) -> Path:
    run_dir = resolve_run_dir(path)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    events = load_events(run_dir)
    rows = candidate_rows(events)
    write_csv(run_dir / "candidate_results.csv", rows)
    summary = {
        "run_id": manifest.get("run_id"),
        "event_counts": dict(
            Counter(event.get("event_type", "unknown") for event in events)
        ),
        "candidate_status_counts": dict(Counter(row["status"] for row in rows)),
        "candidate_count": len(rows),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_dir / "research_report.md").write_text(
        markdown_report(manifest, events, rows),
        encoding="utf-8",
    )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_path", type=Path)
    args = parser.parse_args()
    print(summarize(args.run_path))


if __name__ == "__main__":
    main()
