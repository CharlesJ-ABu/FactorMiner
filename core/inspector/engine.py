import copy
import logging
from typing import Any, Dict, List, Union

from core.data_feed.real_client import RealDataClient
from core.inspector.metrics import InspectorMetricEngine
from core.inspector.reporter import InspectorReporter
from core.inspector.resolver import FactorResolver

logger = logging.getLogger(__name__)


class FactorInspectorEngine:
    """
    因子审查引擎指挥中心：负责协调数据拉取、因子求解、全维度指标评估与报告渲染。
    """

    def __init__(self, base_config: Dict[str, Any] = None):
        self.base_config = base_config or {}
        self.resolver = FactorResolver()

    def inspect(
        self,
        factor_id: str = None,
        ast_str: str = None,
        code_str: str = None,
        ast_dict: Dict = None,
        pairs: List[str] = None,
        periods: List[List[str]] = None,
        timeframe: str = None,
        exchange: str = "binance",
        instrument_type: str = "futures",
    ) -> Dict[str, Dict[str, Any]]:
        """
        对指定因子在指定的币种和时期进行审查。
        """
        # 1. 因子求解
        expression = self.resolver.resolve(
            factor_id=factor_id,
            ast_str=ast_str,
            code_str=code_str,
            ast_dict=ast_dict,
        )
        expression_display = expression.to_display_string()
        logger.info("Inspecting Factor Expression: %s", expression_display)

        # 2. 参数处理
        df_cfg = self.base_config.get("data_feeds", {})
        target_pairs = pairs or df_cfg.get("pairs", ["BTC/USDT:USDT"])
        target_periods = periods or df_cfg.get("mine_period", [["2025-07-20", "2025-08-15"]])
        target_tf = timeframe or df_cfg.get("timeframe", "5m")
        target_exchange = exchange or df_cfg.get("exchange", "binance")
        target_inst = instrument_type or df_cfg.get("instrument_type", "futures")

        results_by_pair: Dict[str, Dict[str, Any]] = {}

        # 3. 逐个交易对加载数据并审查
        for pair in target_pairs:
            logger.info("Inspecting pair: %s (Timeframe: %s)", pair, target_tf)
            cfg = {
                "data_feeds": {
                    "exchange": target_exchange,
                    "instrument_type": target_inst,
                    "timeframe": target_tf,
                    "pairs": [pair],
                    "mine_period": target_periods,
                    "mining_mode": "sequential_single",
                }
            }

            data_client = RealDataClient(cfg)
            data = data_client.get_data()
            returns = data_client.get_returns()

            if data is None or (hasattr(data, "empty") and data.empty):
                logger.warning("No data available for pair %s in specified period.", pair)
                results_by_pair[pair] = {"coverage": 0.0, "total_bars": 0, "valid_bars": 0}
                continue

            try:
                factor_values = expression.compute(data)
                close_prices = data["close"] if "close" in data.columns else None

                metrics = InspectorMetricEngine.calculate_comprehensive_metrics(
                    factor_values=factor_values,
                    returns=returns,
                    close_prices=close_prices,
                )
                results_by_pair[pair] = metrics
            except Exception as e:
                logger.error("Failed to evaluate factor on pair %s: %s", pair, e, exc_info=True)
                results_by_pair[pair] = {"error": str(e), "coverage": 0.0}

        # 4. 打印格式化终端报告
        InspectorReporter.print_inspection_report(expression_display, results_by_pair)

        return results_by_pair
