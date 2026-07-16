"""Shared AST operator discovery, validation, generation and evaluation."""

from __future__ import annotations

from typing import Any, Dict, Iterable

import pandas as pd

from core.miner.registry import OperatorRegistry


class OperatorRuntimeError(ValueError):
    """Raised when an AST references an unavailable or invalid operator."""


def _safe_divide(left: Any, right: Any) -> Any:
    if hasattr(right, "replace"):
        right = right.replace(0, 1e-9)
    elif right == 0:
        right = 1e-9
    return left / right


def _rolling_mean(series: Any) -> Any:
    return series.rolling(20, min_periods=10).mean()


def _rolling_std(series: Any) -> Any:
    return series.rolling(20, min_periods=10).std()


BUILTIN_OPERATORS: Dict[str, Dict[str, Any]] = {
    "add": {"func": lambda left, right: left + right, "arity": 2},
    "sub": {"func": lambda left, right: left - right, "arity": 2},
    "mul": {"func": lambda left, right: left * right, "arity": 2},
    "div": {"func": _safe_divide, "arity": 2},
    "ts_mean": {"func": _rolling_mean, "arity": 1},
    "ts_std": {"func": _rolling_std, "arity": 1},
}


def get_operator_spec(name: str) -> Dict[str, Any]:
    if name in BUILTIN_OPERATORS:
        return BUILTIN_OPERATORS[name]
    if name in OperatorRegistry._registry:
        return OperatorRegistry._registry[name]

    available = sorted({*BUILTIN_OPERATORS, *OperatorRegistry._registry})
    raise OperatorRuntimeError(
        f"Unknown operator '{name}'. Register it under user_workspace/custom_operators "
        f"or choose one of: {', '.join(available)}."
    )


def resolve_operator_specs(operator_names: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    if not isinstance(operator_names, (list, tuple)):
        raise OperatorRuntimeError("search_space.allowed_operators must be a non-empty list of operator names.")
    names = list(operator_names)
    if not names:
        raise OperatorRuntimeError("At least one operator must be configured in search_space.allowed_operators.")

    specs: Dict[str, Dict[str, Any]] = {}
    for name in names:
        if not isinstance(name, str) or not name:
            raise OperatorRuntimeError("Every configured operator name must be a non-empty string.")
        spec = get_operator_spec(name)
        arity = spec.get("arity")
        func = spec.get("func")
        if arity not in (1, 2) or not callable(func):
            raise OperatorRuntimeError(
                f"Operator '{name}' has invalid registration metadata; expected callable func and arity 1 or 2."
            )
        specs[name] = spec
    return specs


def configured_operator_names(config: Dict[str, Any], default: Iterable[str] = ("add", "sub", "mul", "div")) -> list[str]:
    return list(config.get("search_space", {}).get("allowed_operators", list(default)))


def _zero_like_data(data: Any) -> Any:
    if isinstance(data, dict):
        if not data:
            return pd.DataFrame()
        sample = next(iter(data.values()))
        return pd.DataFrame(0.0, index=sample.index, columns=sample.columns)
    if isinstance(data, pd.DataFrame):
        return pd.Series(0.0, index=data.index)
    return pd.Series(dtype=float)


def _terminal_value(node: Any, data: Any) -> Any:
    if isinstance(node, str):
        if isinstance(data, dict):
            if node in data:
                return data[node]
            return _zero_like_data(data)
        if isinstance(data, pd.DataFrame) and node in data.columns:
            return data[node]
        return _zero_like_data(data)
    if isinstance(node, (int, float)):
        zero = _zero_like_data(data)
        return zero + node
    raise OperatorRuntimeError(f"Invalid AST terminal {node!r}; expected a stream name or numeric constant.")


def evaluate_ast(node: Any, data: Any) -> Any:
    """Evaluate a unary/binary AST against sequential or cross-asset data."""
    if not isinstance(node, dict) or "op" not in node:
        return _terminal_value(node, data)

    op_name = node["op"]
    spec = get_operator_spec(op_name)
    left_node = node.get("left")
    if left_node is None:
        raise OperatorRuntimeError(f"Operator '{op_name}' is missing its left operand.")
    left = evaluate_ast(left_node, data)

    if spec["arity"] == 1:
        args = (left,)
    else:
        if "right" not in node or node.get("right") is None:
            raise OperatorRuntimeError(f"Binary operator '{op_name}' is missing its right operand.")
        args = (left, evaluate_ast(node["right"], data))

    try:
        result = spec["func"](*args)
    except Exception as exc:
        raise OperatorRuntimeError(f"Operator '{op_name}' failed during AST evaluation: {exc}") from exc

    if not isinstance(result, (pd.Series, pd.DataFrame)):
        raise OperatorRuntimeError(
            f"Operator '{op_name}' returned {type(result).__name__}; operators must return pandas.Series or pandas.DataFrame."
        )
    return result
