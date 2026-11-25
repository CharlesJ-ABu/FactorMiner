import argparse
import json
import logging
import os
import sys

from core.inspector.engine import FactorInspectorEngine
from core.utils.dynamic_loader import load_user_modules

logger = logging.getLogger("FactorInspectorCLI")


def parse_args():
    parser = argparse.ArgumentParser(description="FactorMiner V4 CLI - Factor Inspector Engine")
    parser.add_argument("--factor", type=str, default=None, help="Factor ID stored in factor_db")
    parser.add_argument("--ast", type=str, default=None, help="Raw AST string or dictionary literal")
    parser.add_argument("--code", type=str, default=None, help="Python factor code string")
    parser.add_argument("--config", type=str, default=None, help="Optional: Path to base JSON configuration file")
    parser.add_argument("--pairs", type=str, default=None, help="Comma-separated symbols (e.g. BTC/USDT:USDT,ETH/USDT:USDT)")
    parser.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--timeframe", type=str, default=None, help="Timeframe (e.g. 5m, 1h, 1d)")
    parser.add_argument("--user-dir", type=str, default="user_workspace", help="Path to user workspace directory")
    return parser.parse_args()


def run_inspector(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # 1. 动态加载用户自定义算子与模型
    if os.path.exists(args.user_dir):
        logger.info("Loading user modules from %s...", args.user_dir)
        load_report = load_user_modules(args.user_dir)
        if load_report.errors:
            logger.error("Inspection startup failed while loading user modules:\n%s", "\n".join(load_report.errors))
            sys.exit(1)

    base_config = {}
    if args.config is not None:
        try:
            with open(args.config, "r", encoding="utf-8") as f:
                base_config = json.load(f)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            logger.error("Failed to load base config %r: %s", args.config, exc)
            sys.exit(1)
        if not isinstance(base_config, dict):
            logger.error("Base config %r must be a JSON object.", args.config)
            sys.exit(1)

    # 2. 解析 CLI 命令行参数覆盖
    pairs_list = [p.strip() for p in args.pairs.split(",")] if args.pairs else None
    periods = [[args.start, args.end]] if (args.start and args.end) else None

    # 3. 运行引擎
    engine = FactorInspectorEngine(base_config=base_config)
    try:
        results = engine.inspect(
            factor_id=args.factor,
            ast_str=args.ast,
            code_str=args.code,
            pairs=pairs_list,
            periods=periods,
            timeframe=args.timeframe,
        )
    except Exception as e:
        logger.error("Inspection failed: %s", e, exc_info=True)
        sys.exit(1)

    failures = {pair: metrics["error"] for pair, metrics in results.items() if "error" in metrics}
    if failures:
        details = "\n".join(f"- {pair}: {error}" for pair, error in failures.items())
        logger.error("Inspection failed for %d of %d requested pairs:\n%s", len(failures), len(results), details)
        sys.exit(1)


if __name__ == "__main__":
    args = parse_args()
    run_inspector(args)
