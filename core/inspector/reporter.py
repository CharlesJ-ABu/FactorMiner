import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class InspectorReporter:
    """
    因子审查报告渲染器：利用 rich 库构建美观的终端审查报告与面板卡片。
    """

    @staticmethod
    def _format_quantiles(quantiles: Dict[str, Any]) -> tuple[str, str]:
        groups = " | ".join(
            f"{key}: " + (f"{value * 10000:.1f}bps" if value is not None else "N/A")
            for key, value in quantiles.items() if key != "Long_Short"
        )
        spread = quantiles.get("Long_Short")
        return groups or "N/A", f"{spread * 10000:+.1f} bps" if spread is not None else "N/A"

    @staticmethod
    def print_inspection_report(
        expression_str: str,
        results_by_pair: Dict[str, Dict[str, Any]],
    ) -> None:
        """在终端格式化输出因子审查报告"""
        try:
            from rich.console import Console
            from rich.panel import Panel
            from rich.table import Table
            from rich.text import Text

            console = Console()

            # 1. 标题面板
            console.print("\n")
            header_text = Text(f"🔬 Factor Inspection Report\n\nExpression: {expression_str}", style="bold white on blue")
            console.print(Panel(header_text, title="FactorMiner Inspector", expand=False))

            # 2. 逐交易对打印指标大表
            for pair, metrics in results_by_pair.items():
                if "error" in metrics:
                    console.print(Panel(
                        Text(f"Inspection unavailable\n{metrics['error']}", style="bold red"),
                        title=f"Symbol Inspection: {pair}",
                        expand=False,
                    ))
                    console.print("\n")
                    continue
                table = Table(
                    title=f"📊 Symbol Inspection: [bold cyan]{pair}[/bold cyan]",
                    show_header=True,
                    header_style="bold magenta",
                )

                table.add_column("Metric Category", style="bold yellow", width=22)
                table.add_column("Indicator / Metric Name", width=28)
                table.add_column("Value", justify="right", style="bold green")

                # Data Coverage
                cov = metrics.get("coverage", 0.0) * 100
                valid_b = metrics.get("valid_bars", 0)
                total_b = metrics.get("total_bars", 0)
                table.add_row("Data Coverage", "Valid Bars / Total", f"{valid_b} / {total_b} ({cov:.1f}%)")

                # Spearman RankIC
                s_metrics = metrics.get("spearman", {})
                table.add_row("RankIC (Spearman)", "Overall RankIC", f"{s_metrics.get('overall', 0.0):.4f}")
                table.add_row("", "Mean RankIC", f"{s_metrics.get('mean', 0.0):.4f}")
                table.add_row("", "RankIC Std", f"{s_metrics.get('std', 0.0):.4f}")
                table.add_row("", "RankIC IR (Annualized)", f"{s_metrics.get('ir', 0.0):.2f}")
                table.add_row("", "RankIC t-stat", f"{s_metrics.get('t_stat', 0.0):.2f}")
                table.add_row("", "Positive IC Ratio", f"{s_metrics.get('pos_ratio', 0.0)*100:.1f}%")

                # Pearson IC
                p_metrics = metrics.get("pearson", {})
                table.add_row("Pearson IC", "Overall Pearson IC", f"{p_metrics.get('overall', 0.0):.4f}")
                table.add_row("", "Pearson IC IR", f"{p_metrics.get('ir', 0.0):.2f}")

                # Lag Decay
                decay = metrics.get("decay", {})
                decay_str = " | ".join([f"{k}: {v:.4f}" for k, v in decay.items()])
                table.add_row("Lag IC Decay", "RankIC across Lags", decay_str if decay_str else "N/A")

                # Quantiles & Long-Short
                quantiles = metrics.get("quantiles", {})
                q_str, spread_str = InspectorReporter._format_quantiles(quantiles)
                table.add_row("Quantile Returns", "Group Means (Q1..Q5)", q_str)
                table.add_row("", "Long-Short Spread (Q5-Q1)", f"[bold gold1]{spread_str}[/bold gold1]")

                # Turnover
                turnover = metrics.get("turnover", 0.0)
                table.add_row("Factor Turnover", "One-period Turnover Rate", f"{turnover:.4f}")

                console.print(table)
                console.print("\n")

        except ImportError:
            logger.info("Rich library not installed. Falling back to plain text printing.")
            print(f"=== Factor Inspection Report: {expression_str} ===")
            for pair, metrics in results_by_pair.items():
                print(f"Pair: {pair}")
                if "error" in metrics:
                    print(f"  Inspection unavailable: {metrics['error']}")
                    continue
                print(f"  RankIC Overall: {metrics.get('spearman', {}).get('overall', 0.0):.4f}")
                print(f"  RankIC IR: {metrics.get('spearman', {}).get('ir', 0.0):.2f}")
                print(f"  RankIC t-stat: {metrics.get('spearman', {}).get('t_stat', 0.0):.2f}")
                print(f"  Decay: {metrics.get('decay')}")
                q_str, spread_str = InspectorReporter._format_quantiles(metrics.get("quantiles", {}))
                print(f"  Quantiles: {q_str}")
                print(f"  Long-Short Spread (Q5-Q1): {spread_str}")
