"""Reference custom NN miner.

Users may replace the model, loss, or training loop in this file.  The shared
engine only requires final expressions to return a Series/DataFrame and models
to expose ``predict_channel()``, ``clone()`` and ``export_artifact()``.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, List

import numpy as np
import pandas as pd

from core.evaluation.targets import target_from_config
from core.miner.entities import EvaluationFeedback
from core.miner.expressions import FactorExpressionTensor
from core.miner.nn import NumpyMLPFactorModel
from core.miner.paradigms.base import BaseFactorMiner
from core.miner.registry import MinerRegistry

logger = logging.getLogger(__name__)


class NNTrainingTensor:
    """Training-only evaluator payload; it is never persisted as a factor."""

    def __init__(self, data: Any):
        self.data = data
        self.requires_grad = True


class MyNNExpression(FactorExpressionTensor):
    """One materialized output channel of a custom neural model."""

    def compute(self, data: Any):
        if self.model_instance is None:
            raise RuntimeError("NN expression has no model instance.")
        if self.channel_idx < 0:
            return NNTrainingTensor(data)
        return self.model_instance.predict_channel(data, self.channel_idx)


@MinerRegistry.register("NN")
@MinerRegistry.register("MyCustomNN")
class MyCustomNNMiner(BaseFactorMiner):
    """Extensible reference NN miner with OOS selection and portable artifacts."""

    def initialize_search_space(self) -> None:
        feeds = self.config.get("data_feeds", {})
        self.target_spec = target_from_config(self.config)
        self.terminals = feeds.get("required_streams", ["close", "volume"])
        self.mining_mode = feeds.get("mining_mode", "sequential_single")
        self.hidden_dim = int(self.config.get("hidden_dim", 8))
        self.lr = float(self.config.get("learning_rate", 0.01))
        self.random_seed = int(self.config.get("random_seed", 42))
        self.epochs_per_iteration = int(
            self.config.get("nn_training_epochs", 50)
        )
        self.min_samples = int(self.config.get("nn_min_samples", 30))
        self.l2 = float(self.config.get("nn_l2", 1e-4))
        self.diversity_penalty = float(
            self.config.get("nn_diversity_penalty", 1e-3)
        )
        self.gradient_clip = float(self.config.get("nn_gradient_clip", 5.0))
        self.early_stopping_patience = int(
            self.config.get("nn_early_stopping_patience", 20)
        )
        if self.lr <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.epochs_per_iteration <= 0:
            raise ValueError("nn_training_epochs must be positive.")
        if self.min_samples < 2:
            raise ValueError("nn_min_samples must be at least 2.")
        if int(self.config.get("top_k_factors", 5)) <= 0:
            raise ValueError("top_k_factors must be positive.")
        max_corr = float(self.config.get("max_corr", 0.95))
        if not 0.0 <= max_corr <= 1.0:
            raise ValueError("max_corr must be between 0 and 1.")
        self.model = self.build_model()
        self.epoch = 0
        self._head_archive: dict[str, FactorExpressionTensor] = {}

    def build_model(self):
        """Build the reference model.

        Replacing only this method also requires ``adapter.prepare()`` and
        ``train()`` because this Miner owns the training loop.  Replacing the
        training loop removes that requirement; persisted models only need the
        shared prediction/artifact protocol documented at module level.
        """
        return NumpyMLPFactorModel(
            self.terminals,
            self.hidden_dim,
            self.mining_mode,
            self.random_seed,
        )

    def build_target_returns(
        self,
        data: Any,
        default_returns: Any,
        split: str,
    ) -> Any:
        """Customization point for alternate prediction horizons or labels."""
        return default_returns

    def get_forward_return_definition(self) -> str:
        return self.target_spec.definition()

    def generate_candidates(self) -> List[FactorExpressionTensor]:
        return [
            MyNNExpression(
                model_version_id=f"nn_training_{self.epoch}",
                channel_idx=-1,
                model_instance=self.model,
            )
        ]

    def evaluate_candidates(
        self, candidates: List[FactorExpressionTensor]
    ) -> EvaluationFeedback:
        if not self.evaluator:
            raise RuntimeError("MyCustomNN requires an evaluator.")
        return self.evaluator.evaluate(candidates)

    @staticmethod
    def _has_split(data: Any, returns: Any) -> bool:
        if isinstance(data, pd.DataFrame):
            return not data.empty and isinstance(returns, pd.Series) and not returns.empty
        if isinstance(data, dict):
            return (
                bool(data)
                and isinstance(returns, pd.DataFrame)
                and not returns.empty
            )
        return False

    @staticmethod
    def _time_index(data: Any) -> pd.Index:
        if isinstance(data, pd.DataFrame):
            return data.index
        if isinstance(data, dict):
            for value in data.values():
                if isinstance(value, pd.DataFrame):
                    return value.index
        return pd.Index([])

    @staticmethod
    def _flat_factor_values(values: Any) -> pd.Series:
        if isinstance(values, pd.DataFrame):
            return values.stack().dropna()
        if isinstance(values, pd.Series):
            return values.dropna()
        return pd.Series(dtype=float)

    def _select_diverse_heads(
        self, heads: List[FactorExpressionTensor]
    ) -> List[FactorExpressionTensor]:
        ranked = sorted(
            heads,
            key=lambda head: head.metrics.get("fitness_score", float("-inf")),
            reverse=True,
        )
        selected: List[FactorExpressionTensor] = []
        max_corr = float(self.config.get("max_corr", 0.95))
        top_k = int(self.config.get("top_k_factors", 5))
        min_fitness = float(
            self.config.get("nn_min_fitness", float("-inf"))
        )

        for candidate in ranked:
            if candidate.metrics.get("fitness_score", float("-inf")) < min_fitness:
                continue
            candidate_values = getattr(candidate, "_selection_values", None)
            candidate_flat = self._flat_factor_values(candidate_values)
            if candidate_flat.empty:
                continue

            redundant = False
            for accepted in selected:
                accepted_values = getattr(accepted, "_selection_values", None)
                accepted_flat = self._flat_factor_values(accepted_values)
                aligned = pd.concat(
                    [candidate_flat.rename("candidate"), accepted_flat.rename("accepted")],
                    axis=1,
                    join="inner",
                ).dropna()
                if len(aligned) >= self.min_samples:
                    correlation = aligned["candidate"].corr(aligned["accepted"])
                    if pd.notna(correlation) and abs(float(correlation)) >= max_corr:
                        redundant = True
                        break
            if not redundant:
                selected.append(candidate)
            if len(selected) >= top_k:
                break
        return selected

    def _score_heads(
        self,
        heads: List[FactorExpressionTensor],
        train_data: Any,
        train_returns: Any,
        test_data: Any,
        test_returns: Any,
    ) -> List[FactorExpressionTensor]:
        train_feedback = self.evaluator.evaluate_on(
            heads, train_data, train_returns
        )
        train_metrics = {id(head): dict(head.metrics) for head in heads if head.metrics}

        has_test = self._has_split(test_data, test_returns)
        test_errors = []
        if has_test:
            for head in heads:
                head.metrics = {}
            test_feedback = self.evaluator.evaluate_on(
                heads, test_data, test_returns
            )
            test_errors = test_feedback.execution_status

        valid_heads = []
        for head in heads:
            final_metrics = dict(head.metrics) if has_test else train_metrics.get(id(head), {})
            if not final_metrics:
                continue
            train = train_metrics.get(id(head), {})
            for key in ("IC", "RankIC", "Turnover", "coverage", "fitness_score"):
                if key in train:
                    final_metrics[f"train_{key}"] = float(train[key])
            final_metrics["out_of_sample"] = 1.0 if has_test else 0.0
            head.metrics = final_metrics
            selection_data = test_data if has_test else train_data
            head._selection_values = head.compute(selection_data)
            head.evaluation_split = "test" if has_test else "mine"
            head.evaluation_returns = test_returns if has_test else train_returns
            head.evaluation_target = self.target_spec
            head.forward_return_definition = self.get_forward_return_definition()
            valid_heads.append(head)

        if train_feedback.execution_status:
            logger.warning(
                "MyCustomNN: %s training-split channel evaluations failed.",
                len(train_feedback.execution_status),
            )
        if test_errors:
            logger.warning(
                "MyCustomNN: %s test-split channel evaluations failed.",
                len(test_errors),
            )
        return valid_heads

    def update_model(
        self,
        candidates: List[FactorExpressionTensor],
        feedback: EvaluationFeedback,
    ) -> None:
        if feedback.execution_status or feedback.raw_outputs is None:
            detail = (
                feedback.execution_status[0].get("traceback", "")
                if feedback.execution_status
                else "No training tensor was returned."
            )
            raise RuntimeError(f"MyCustomNN forward pass failed. {detail}".strip())

        train_data = self.evaluator.data_client.get_data()
        train_returns = self.build_target_returns(
            train_data,
            self.evaluator.data_client.get_returns(),
            "mine",
        )
        test_data = self.evaluator.data_client.get_test_data()
        test_returns = self.build_target_returns(
            test_data,
            self.evaluator.data_client.get_test_returns(),
            "test",
        )
        if self._has_split(test_data, test_returns):
            overlap = self._time_index(train_data).intersection(
                self._time_index(test_data)
            )
            if not overlap.empty:
                raise RuntimeError(
                    "MyCustomNN mine_period and test_period overlap. "
                    "Use a disjoint test split for honest out-of-sample selection."
                )

        prepared = self.model.adapter.prepare(train_data, train_returns)
        if len(prepared.X) < self.min_samples:
            raise RuntimeError(
                f"MyCustomNN requires at least {self.min_samples} aligned training "
                f"samples, received {len(prepared.X)}."
            )

        loss = self.model.train(
            prepared.X,
            prepared.y,
            epochs=self.epochs_per_iteration,
            learning_rate=self.lr,
            l2=self.l2,
            diversity=self.diversity_penalty,
            gradient_clip=self.gradient_clip,
            patience=self.early_stopping_patience,
        )

        snapshot = self.model.clone()
        artifact = snapshot.export_artifact()
        model_version_id = f"nn_{hashlib.sha256(artifact.payload).hexdigest()[:16]}"
        heads = []
        for channel_idx in range(self.hidden_dim):
            head = MyNNExpression(
                model_version_id=model_version_id,
                channel_idx=channel_idx,
                model_instance=snapshot,
            )
            head.logic_hash = hashlib.sha256(
                f"{model_version_id}:{channel_idx}".encode("utf-8")
            ).hexdigest()
            heads.append(head)

        valid_heads = self._score_heads(
            heads, train_data, train_returns, test_data, test_returns
        )
        if not valid_heads:
            raise RuntimeError("MyCustomNN produced no valid evaluated output channels.")

        for head in valid_heads:
            self._head_archive[head.logic_hash] = head
        self.state.population = self._select_diverse_heads(
            list(self._head_archive.values())
        )
        if not self.state.population:
            raise RuntimeError("MyCustomNN produced no channels after diversity filtering.")

        best = self.state.population[0]
        logger.info(
            "MyCustomNN epoch %s: loss=%.8f, retained=%s, best_channel=%s, "
            "fitness=%.6f, OOS=%s.",
            self.epoch,
            loss,
            len(self.state.population),
            best.channel_idx,
            best.metrics.get("fitness_score", 0.0),
            bool(best.metrics.get("out_of_sample")),
        )
        self.epoch += 1

    def _get_best_factors(self) -> List[FactorExpressionTensor]:
        """NN owns its cross-epoch archive; other paradigms keep the base policy."""
        return list(self.state.population)

    def mine(self, n_iterations: int, progress_callback=None):
        result = super().mine(n_iterations, progress_callback=progress_callback)
        if not result:
            raise RuntimeError("MyCustomNN completed without any valid factors.")
        return result
