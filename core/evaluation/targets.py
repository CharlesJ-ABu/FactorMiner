"""Canonical, config-driven prediction targets shared by mining and inspection."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping

import numpy as np
import pandas as pd


class TargetConfigError(ValueError):
    """Raised when a prediction-target configuration is invalid."""


@dataclass(frozen=True)
class ForwardReturnTarget:
    """Define one point-in-time forward-return label."""

    type: str = "forward_return"
    entry_price: str = "current_close"
    exit_price: str = "close"
    horizon_bars: int = 1
    return_type: str = "simple"

    @classmethod
    def from_config(
        cls,
        target_config: Mapping[str, Any] | None,
    ) -> "ForwardReturnTarget":
        if target_config is None:
            return cls()
        if not isinstance(target_config, Mapping):
            raise TargetConfigError("target must be an object when provided.")

        allowed = {
            "type",
            "entry_price",
            "exit_price",
            "horizon_bars",
            "return_type",
        }
        unknown = sorted(set(target_config).difference(allowed))
        if unknown:
            raise TargetConfigError(
                f"Unknown target field(s): {', '.join(unknown)}."
            )

        values = {
            "type": target_config.get("type", cls.type),
            "entry_price": target_config.get("entry_price", cls.entry_price),
            "exit_price": target_config.get("exit_price", cls.exit_price),
            "horizon_bars": target_config.get("horizon_bars", cls.horizon_bars),
            "return_type": target_config.get("return_type", cls.return_type),
        }
        if values["type"] != "forward_return":
            raise TargetConfigError("target.type must be 'forward_return'.")
        if values["entry_price"] not in {"current_close", "next_open"}:
            raise TargetConfigError(
                "target.entry_price must be 'current_close' or 'next_open'."
            )
        if values["exit_price"] != "close":
            raise TargetConfigError(
                "target.exit_price currently supports only 'close'."
            )
        horizon = values["horizon_bars"]
        if (
            isinstance(horizon, bool)
            or not isinstance(horizon, int)
            or horizon <= 0
        ):
            raise TargetConfigError("target.horizon_bars must be a positive integer.")
        if values["return_type"] not in {"simple", "log"}:
            raise TargetConfigError(
                "target.return_type must be 'simple' or 'log'."
            )
        return cls(**values)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def with_horizon(self, horizon_bars: int) -> "ForwardReturnTarget":
        if (
            isinstance(horizon_bars, bool)
            or not isinstance(horizon_bars, int)
            or horizon_bars <= 0
        ):
            raise TargetConfigError("target.horizon_bars must be a positive integer.")
        return replace(self, horizon_bars=horizon_bars)

    def definition(self) -> str:
        entry = "close[t]" if self.entry_price == "current_close" else "open[t+1]"
        exit_value = f"close[t+{self.horizon_bars}]"
        ratio = f"{exit_value} / {entry}"
        return f"log({ratio})" if self.return_type == "log" else f"{ratio} - 1"

    def build(self, data: pd.DataFrame) -> pd.Series:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Forward-return targets require a pandas.DataFrame.")

        required = {"close"}
        if self.entry_price == "next_open":
            required.add("open")
        missing = sorted(required.difference(data.columns))
        if missing:
            raise TargetConfigError(
                f"Target requires missing price column(s): {', '.join(missing)}."
            )

        close = pd.to_numeric(data["close"], errors="coerce")
        exit_values = close.shift(-self.horizon_bars)
        if self.entry_price == "current_close":
            entry_values = close
        else:
            entry_values = pd.to_numeric(data["open"], errors="coerce").shift(-1)

        ratio = exit_values.div(entry_values.replace(0.0, np.nan))
        returns = np.log(ratio) if self.return_type == "log" else ratio.sub(1.0)
        returns = returns.replace([np.inf, -np.inf], np.nan)
        returns.name = "returns"
        return returns


def target_from_config(config: Mapping[str, Any] | None) -> ForwardReturnTarget:
    """Normalize a full FactorMiner config into the canonical target contract."""
    if config is None:
        return ForwardReturnTarget()
    if not isinstance(config, Mapping):
        raise TargetConfigError("FactorMiner config must be an object.")
    return ForwardReturnTarget.from_config(config.get("target"))
