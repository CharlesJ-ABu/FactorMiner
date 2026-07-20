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
        load_user_modules(args.user_dir)

    base_config = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, "r") as f:
            try:
                base_config = json.load(f)
            except Exception as e:
                logger.warning("Failed to load base config %s: %s", args.config, e)

    # 2. 解析 CLI 命令行参数覆盖
    pairs_list = [p.strip() for p in args.pairs.split(",")] if args.pairs else None
    periods = [[args.start, args.end]] if (args.start and args.end) else None

    # 3. 运行引擎
    engine = FactorInspectorEngine(base_config=base_config)
    try:
        engine.inspect(
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


if __name__ == "__main__":
    args = parse_args()
    run_inspector(args)
