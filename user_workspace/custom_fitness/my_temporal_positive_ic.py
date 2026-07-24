"""Signed RankIC fitness for the MyTemporalNN learning template."""

from core.miner.registry import EvaluatorRegistry


@EvaluatorRegistry.register_fitness_hook("my_temporal_positive_ic")
def temporal_positive_ic_fitness(factor_values, returns, base_metrics):
    rank_ic = float(base_metrics.get("RankIC", 0.0))
    if hasattr(factor_values, "dropna"):
        valid_count = int(factor_values.dropna().count())
        total_count = int(factor_values.size)
        coverage = valid_count / total_count if total_count else 0.0
    else:
        coverage = 1.0
    penalty = (coverage / 0.20) ** 2 if coverage < 0.20 else 1.0
    return rank_ic * 100.0 * penalty
