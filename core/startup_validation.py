"""Fail-fast validation for CLI and WebUI mining launches."""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable

from core.evaluation.targets import TargetConfigError, target_from_config
from core.miner.operator_runtime import OperatorRuntimeError, resolve_operator_specs
from core.miner.registry import EvaluatorRegistry, MinerRegistry
from core.utils.dynamic_loader import ModuleLoadReport


class StartupValidationError(ValueError):
    """A user-facing aggregation of launch configuration problems."""


logger = logging.getLogger(__name__)

NATIVE_MINERS = {"GP", "RL", "LLM"}
DEPRECATED_MINER_ALIASES = {"DL": "NN"}


def normalize_deprecated_miner(config: Dict[str, Any]) -> str | None:
    """Mutate deprecated public miner names to their supported replacement."""
    paradigm = config.get("paradigm")
    replacement = DEPRECATED_MINER_ALIASES.get(paradigm)
    if replacement:
        logger.warning(
            "Miner '%s' is deprecated and will be removed; using '%s'. "
            "Update the configuration.",
            paradigm,
            replacement,
        )
        config["paradigm"] = replacement
    return replacement


def _as_error_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        return [f"{label} must be a non-empty list."]
    return []


def validate_mining_startup(config: Dict[str, Any], load_report: ModuleLoadReport | None = None) -> None:
    """Validate dynamic extensions and the parts of config that select them."""
    errors: list[str] = []
    normalize_deprecated_miner(config)

    if load_report:
        errors.extend(load_report.errors)

    paradigm = config.get("paradigm")
    if not isinstance(paradigm, str) or not paradigm:
        errors.append("Miner name is missing. Supply --miner (CLI) or select a Miner in Launchpad.")
    elif paradigm not in NATIVE_MINERS and paradigm not in MinerRegistry._registry:
        available = sorted({*NATIVE_MINERS, *MinerRegistry._registry})
        errors.append(
            f"Unknown Miner '{paradigm}'. Available Miners: {', '.join(available)}."
        )

    iterations = config.get("max_iterations", 1)
    if not isinstance(iterations, int) or iterations <= 0:
        errors.append("max_iterations must be a positive integer.")

    try:
        target_from_config(config)
    except TargetConfigError as exc:
        errors.append(str(exc))

    data_feeds = config.get("data_feeds")
    if not isinstance(data_feeds, dict):
        errors.append("data_feeds must be an object containing pairs and required_streams.")
    else:
        errors.extend(_as_error_list(data_feeds.get("pairs"), "data_feeds.pairs"))
        errors.extend(_as_error_list(data_feeds.get("required_streams"), "data_feeds.required_streams"))

    search_space = config.get("search_space", {})
    if search_space and not isinstance(search_space, dict):
        errors.append("search_space must be an object when provided.")
    elif isinstance(search_space, dict) and "allowed_operators" in search_space:
        try:
            resolve_operator_specs(search_space["allowed_operators"])
        except OperatorRuntimeError as exc:
            errors.append(str(exc))

    fitness = config.get("fitness", {})
    if fitness and not isinstance(fitness, dict):
        errors.append("fitness must be an object when provided.")
    elif isinstance(fitness, dict) and fitness.get("hook"):
        hook_name = fitness["hook"]
        if hook_name not in EvaluatorRegistry._registry:
            available = sorted(EvaluatorRegistry._registry)
            suffix = ", ".join(available) if available else "none loaded"
            errors.append(
                f"Unknown Fitness Hook '{hook_name}'. Available Hooks: {suffix}."
            )

    if errors:
        detail = "\n".join(f"- {error}" for error in errors)
        raise StartupValidationError(f"Mining startup validation failed:\n{detail}")
