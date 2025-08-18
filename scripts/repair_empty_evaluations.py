#!/usr/bin/env python3
"""
修复 factorlib/evaluations 下的空/损坏评估文件：
- 若JSON为空或无法解析，则重写为有效的空结构：
  {"factor_id": "<id>", "evaluations": []}
"""

import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    eval_dir = root / "factorlib" / "evaluations"
    if not eval_dir.exists():
        print(f"目录不存在: {eval_dir}")
        return

    fixed = 0
    for file in eval_dir.glob("*_evaluation.json"):
        try:
            text = file.read_text(encoding="utf-8")
            if not text.strip():
                raise ValueError("空文件")
            _ = json.loads(text)
            continue  # 已是有效JSON
        except Exception:
            # 重写为空结构
            factor_id = file.stem.replace("_evaluation", "")
            payload = {
                "factor_id": factor_id,
                "evaluations": []
            }
            with open(file, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            fixed += 1
            print(f"已修复: {file.name}")

    print(f"完成。修复文件数: {fixed}")


if __name__ == "__main__":
    main()


