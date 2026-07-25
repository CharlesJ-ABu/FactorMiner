import argparse
import logging
import json
import os
import sys

from core.utils.dynamic_loader import load_user_modules
from core.miner.director import FactorMinerDirector
from core.startup_validation import StartupValidationError, validate_mining_startup

# Removed MockDataClient in favor of RealDataClient


def parse_args():
    parser = argparse.ArgumentParser(description="FactorMiner V4 CLI - Freqtrade Paradigm")
    parser.add_argument("--miner", type=str, required=True, help="Name of the miner class/paradigm (e.g., GP, LLM, MyCustomGP)")
    parser.add_argument("--config", type=str, required=True, help="Path to the JSON configuration file")
    parser.add_argument("--iterations", type=int, default=None, help="Optional: Number of iterations (overrides config)")
    parser.add_argument("--user-dir", type=str, default="user_workspace", help="Path to the user workspace directory")
    return parser.parse_args()

def run_miner(args):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("FactorMinerCLI")
    
    # 1. 解析配置文件
    if not os.path.exists(args.config):
        logger.error(f"Config file {args.config} not found!")
        sys.exit(1)
        
    with open(args.config, 'r') as f:
        try:
            config_dict = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse config file: {e}")
            sys.exit(1)
            
    # 覆盖配置中的 paradigm
    config_dict["paradigm"] = args.miner
    if args.iterations is not None:
        config_dict["max_iterations"] = args.iterations
    
    # 2. 动态加载用户目录 (Freqtrade 范式的核心！)
    logger.info(f"Loading user modules from {args.user_dir}...")
    load_report = load_user_modules(args.user_dir)

    try:
        validate_mining_startup(config_dict, load_report)
    except StartupValidationError as exc:
        logger.error("%s", exc)
        sys.exit(1)
    
    # 3. 初始化并启动
    selected_miner = config_dict["paradigm"]
    logger.info(f"Starting FactorMiner execution for miner: {selected_miner}")
    from core.data_feed.real_client import RealDataClient
    
    mining_mode = config_dict.get("data_feeds", {}).get("mining_mode", "sequential_single")
    pairs = config_dict.get("data_feeds", {}).get("pairs", [])
    
    try:
        if mining_mode == "sequential_single" and pairs:
            all_mined_factors = []
            
            # 顺序挖掘每一个品种
            for pair in pairs:
                logger.info(f"=== Mining Target: {pair} ===")
                # 为了让 RealDataClient 只加载当前 pair 的数据，覆盖 config
                import copy
                pair_config = copy.deepcopy(config_dict)
                pair_config["data_feeds"]["pairs"] = [pair]
                
                data_client = RealDataClient(pair_config)
                director = FactorMinerDirector(pair_config, data_client)
                
                iters = pair_config.get("max_iterations", 5)
                best_candidates = director.run(max_iterations=iters)
                
                factor_ids = director.get_latest_factor_ids()
                for f_id, cand in zip(factor_ids, best_candidates):
                    all_mined_factors.append((f_id, pair, cand))
            
            # 打印全局大表
            if all_mined_factors:
                try:
                    from rich.console import Console
                    from rich.table import Table
                    
                    console = Console()
                    console.print("\n[bold green]====== 🏆 FINAL MINING SUMMARY ======[/bold green]\n")
                    
                    # 按照 target_pair 分组
                    from collections import defaultdict
                    grouped_factors = defaultdict(list)
                    for item in all_mined_factors:
                        grouped_factors[item[1]].append(item)
                        
                    for pair, factors in grouped_factors.items():
                        table = Table(
                            title=f"Target Pair: {pair}",
                            show_header=True, 
                            header_style="bold blue"
                        )
                        table.add_column("Rank", style="dim", width=6)
                        table.add_column("Factor ID", width=12)
                        table.add_column("Paradigm")
                        table.add_column("Logic / Expression", overflow="fold")
                        table.add_column("Fitness Score", justify="right")
                        table.add_column("IC", justify="right")
                        
                        # 按照 fitness score 降序排列
                        factors.sort(key=lambda x: x[2].metrics.get('fitness_score', 0), reverse=True)
                        
                        for idx, (f_id, _, cand) in enumerate(factors):
                            metrics = cand.metrics
                            fitness = f"{metrics.get('fitness_score', 0):.4f}"
                            ic = f"{metrics.get('IC', 0):.4f}"
                            paradigm = selected_miner
                            logic = cand.to_display_string(max_length=None)
                            
                            table.add_row(str(idx + 1), f_id, paradigm, logic, fitness, ic)
                            
                        console.print(table)
                        console.print("\n")
                except ImportError:
                    pass
                
        else:
            # cross_asset 模式或者未配置 pairs
            logger.info("=== Mining Target: ALL (Cross-Asset) ===")
            data_client = RealDataClient(config_dict)
            director = FactorMinerDirector(config_dict, data_client)
            best_candidates = director.run(max_iterations=args.iterations)
            
            try:
                from rich.console import Console
                from rich.table import Table
                
                console = Console()
                console.print("\n[bold green]====== 🏆 FINAL MINING SUMMARY ======[/bold green]\n")
                
                table = Table(
                    title="Target Pair: ALL (Cross-Asset)",
                    show_header=True, 
                    header_style="bold blue"
                )
                table.add_column("Rank", style="dim", width=6)
                table.add_column("Factor ID", width=12)
                table.add_column("Paradigm")
                table.add_column("Logic / Expression", overflow="fold")
                table.add_column("Fitness Score", justify="right")
                table.add_column("IC", justify="right")
                
                # 获取所有的因子并且按照 fitness 排序
                factor_ids = director.get_latest_factor_ids()
                candidates_with_id = list(zip(factor_ids, best_candidates))
                candidates_with_id.sort(key=lambda x: x[1].metrics.get('fitness_score', 0), reverse=True)
                
                for idx, (f_id, cand) in enumerate(candidates_with_id):
                    metrics = cand.metrics
                    fitness = f"{metrics.get('fitness_score', 0):.4f}"
                    ic = f"{metrics.get('IC', 0):.4f}"
                    paradigm = selected_miner
                    logic = cand.to_display_string(max_length=None)
                    
                    table.add_row(str(idx + 1), f_id, paradigm, logic, fitness, ic)
                    
                console.print(table)
                console.print("\n")
            except ImportError:
                pass
            
        logger.info("FactorMiner execution completed successfully.")
    except Exception as e:
        logger.error(f"Execution failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    args = parse_args()
    run_miner(args)
