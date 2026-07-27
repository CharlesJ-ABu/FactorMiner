"""Teaching template: temporal feature NN optimized toward out-of-sample IC."""

from __future__ import annotations

from io import BytesIO
import json
import threading
from typing import Any, Sequence

import numpy as np
import pandas as pd

from core.miner.nn import (
    NNDataAdapter,
    NNModelArtifact,
    NumpyMLPFactorModel,
    register_nn_model_loader,
)
from core.evaluation.targets import target_from_config
from core.miner.registry import MinerRegistry
from user_workspace.custom_miners.my_custom_nn import MyCustomNNMiner


class TemporalFeatureAdapter(NNDataAdapter):
    """Build stationary, backward-looking features without future leakage."""

    FEATURE_VERSION = "temporal_ohlcv_v1"
    FEATURE_NAMES = [
        "return_1",
        "return_3",
        "return_5",
        "return_15",
        "return_30",
        "return_60",
        "volatility_5",
        "volatility_15",
        "volatility_60",
        "volume_log",
        "volume_change_1",
        "volume_change_5",
        "volume_z20",
        "volume_z60",
        "price_z20",
        "price_z60",
        "intrabar_range",
        "candle_body",
        "minute_sin",
        "minute_cos",
    ]

    def __init__(self):
        super().__init__(self.FEATURE_NAMES, "sequential_single")

    @staticmethod
    def _rolling_zscore(values: pd.Series, window: int) -> pd.Series:
        mean = values.rolling(window, min_periods=window).mean()
        std = values.rolling(window, min_periods=window).std()
        return (values - mean) / std.replace(0.0, np.nan)

    def engineer_features(self, data: Any) -> pd.DataFrame:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("MyTemporalNN currently supports sequential DataFrame data.")
        required = {"close", "volume"}
        missing = sorted(required.difference(data.columns))
        if missing:
            raise ValueError(f"Temporal NN input is missing columns: {missing}")

        close = pd.to_numeric(data["close"], errors="coerce")
        volume = pd.to_numeric(data["volume"], errors="coerce").clip(lower=0)
        returns = close.pct_change()
        features = pd.DataFrame(index=data.index)
        for horizon in (1, 3, 5, 15, 30, 60):
            features[f"return_{horizon}"] = close.pct_change(horizon)
        for window in (5, 15, 60):
            features[f"volatility_{window}"] = returns.rolling(
                window, min_periods=window
            ).std()

        features["volume_log"] = np.log1p(volume)
        features["volume_change_1"] = volume.pct_change()
        features["volume_change_5"] = volume.pct_change(5)
        features["volume_z20"] = self._rolling_zscore(volume, 20)
        features["volume_z60"] = self._rolling_zscore(volume, 60)
        features["price_z20"] = self._rolling_zscore(close, 20)
        features["price_z60"] = self._rolling_zscore(close, 60)

        if {"high", "low"}.issubset(data.columns):
            high = pd.to_numeric(data["high"], errors="coerce")
            low = pd.to_numeric(data["low"], errors="coerce")
            features["intrabar_range"] = (high - low) / close.replace(0.0, np.nan)
        else:
            features["intrabar_range"] = 0.0

        if "open" in data.columns:
            open_price = pd.to_numeric(data["open"], errors="coerce")
            features["candle_body"] = (
                close - open_price
            ) / open_price.replace(0.0, np.nan)
        else:
            features["candle_body"] = 0.0

        if isinstance(data.index, pd.DatetimeIndex):
            minute = data.index.hour * 60 + data.index.minute
            phase = 2.0 * np.pi * np.asarray(minute, dtype=float) / 1440.0
            features["minute_sin"] = np.sin(phase)
            features["minute_cos"] = np.cos(phase)
        else:
            features["minute_sin"] = 0.0
            features["minute_cos"] = 0.0
        return features.loc[:, self.FEATURE_NAMES]

    def prepare(self, data: Any, returns: Any = None):
        engineered = self.engineer_features(data)
        return self._prepare_sequential(engineered, returns)


class TemporalICFactorModel(NumpyMLPFactorModel):
    """MLP with a directly tradable prediction channel and Pearson-IC loss."""

    MODEL_FORMAT = "numpy_temporal_ic_mlp_v1"
    SCHEMA_VERSION = 1

    def __init__(
        self,
        output_channels: int,
        prediction_horizon: int = 5,
        ic_loss_weight: float = 0.7,
        random_seed: int = 42,
    ):
        if output_channels < 2:
            raise ValueError("Temporal NN requires at least two output channels.")
        self.output_channels = int(output_channels)
        self.prediction_horizon = int(prediction_horizon)
        self.ic_loss_weight = float(ic_loss_weight)
        latent_dim = self.output_channels - 1
        super().__init__(
            TemporalFeatureAdapter.FEATURE_NAMES,
            latent_dim,
            "sequential_single",
            random_seed,
        )
        self.adapter = TemporalFeatureAdapter()
        self.target_mean = 0.0
        self.target_std = 1.0
        self.target_scaler_fitted = False
        self._prediction_cache = {}
        self._cache_lock = threading.Lock()

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        epochs: int,
        learning_rate: float,
        l2: float = 1e-4,
        diversity: float = 1e-3,
        gradient_clip: float = 5.0,
        patience: int = 20,
    ) -> float:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if len(X) != len(y) or not len(y):
            raise ValueError("Temporal NN features and targets must be aligned.")
        if not self.scaler_fitted:
            self.fit_scaler(X)
        if not self.target_scaler_fitted:
            self.target_mean = float(np.mean(y))
            self.target_std = float(np.std(y))
            if self.target_std < 1e-12:
                self.target_std = 1.0
            self.target_scaler_fitted = True

        normalized = self.transform(X)
        target = (y - self.target_mean) / self.target_std
        target_centered = target - target.mean()
        target_norm = float(np.linalg.norm(target_centered)) + 1e-12
        alpha = min(max(self.ic_loss_weight, 0.0), 1.0)
        best_loss = float("inf")
        best_parameters = None
        stale_epochs = 0

        for _ in range(max(1, int(epochs))):
            channels = np.tanh(normalized @ self.W1 + self.b1)
            prediction = channels @ self.W2 + self.b2
            error = prediction - target
            mse = float(np.mean(error ** 2))

            prediction_centered = prediction - prediction.mean()
            prediction_norm = float(np.linalg.norm(prediction_centered)) + 1e-12
            correlation = float(
                np.dot(prediction_centered, target_centered)
                / (prediction_norm * target_norm)
            )
            loss = (1.0 - alpha) * mse + alpha * (1.0 - correlation)

            d_mse = (2.0 / len(target)) * error
            d_correlation = (
                target_centered / (prediction_norm * target_norm)
                - correlation
                * prediction_centered
                / (prediction_norm ** 2)
            )
            d_prediction = (1.0 - alpha) * d_mse - alpha * d_correlation

            dW2 = channels.T @ d_prediction + l2 * self.W2
            db2 = float(np.sum(d_prediction))
            d_channels = d_prediction[:, None] * self.W2[None, :]
            d_pre_activation = d_channels * (1.0 - channels ** 2)
            dW1 = normalized.T @ d_pre_activation + l2 * self.W1
            db1 = np.sum(d_pre_activation, axis=0)

            gram = self.W1.T @ self.W1
            off_diagonal = gram - np.diag(np.diag(gram))
            dW1 += diversity * 4.0 * (self.W1 @ off_diagonal)

            gradients = [dW1, db1, dW2, np.asarray(db2)]
            total_norm = np.sqrt(
                sum(float(np.sum(gradient ** 2)) for gradient in gradients)
            )
            scale = min(1.0, float(gradient_clip) / (total_norm + 1e-12))
            self.W1 -= learning_rate * dW1 * scale
            self.b1 -= learning_rate * db1 * scale
            self.W2 -= learning_rate * dW2 * scale
            self.b2 -= learning_rate * db2 * scale

            if loss + 1e-10 < best_loss:
                best_loss = loss
                best_parameters = (
                    self.W1.copy(),
                    self.b1.copy(),
                    self.W2.copy(),
                    float(self.b2),
                )
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= max(1, int(patience)):
                    break

        if best_parameters is not None:
            self.W1, self.b1, self.W2, self.b2 = best_parameters
        self.clear_prediction_cache()
        return best_loss

    def _predict_all(self, data: Any):
        cache_key = id(data)
        cached = self._prediction_cache.get(cache_key)
        if cached is not None and cached[0] is data:
            return cached[1], cached[2]
        with self._cache_lock:
            cached = self._prediction_cache.get(cache_key)
            if cached is not None and cached[0] is data:
                return cached[1], cached[2]
            prepared = self.adapter.prepare(data)
            channels, prediction = self._forward(prepared.X)
            outputs = np.column_stack([prediction, channels])
            self._prediction_cache = {cache_key: (data, prepared, outputs)}
            return prepared, outputs

    def predict_channel(self, data: Any, channel_idx: int) -> Any:
        if channel_idx < 0 or channel_idx >= self.output_channels:
            raise IndexError(f"Channel {channel_idx} is outside the model output.")
        prepared, outputs = self._predict_all(data)
        return prepared.restore_channel(outputs[:, channel_idx])

    def clear_prediction_cache(self) -> None:
        self._prediction_cache = {}

    def clone(self) -> "TemporalICFactorModel":
        clone = TemporalICFactorModel(
            self.output_channels,
            self.prediction_horizon,
            self.ic_loss_weight,
            self.random_seed,
        )
        for name in ("W1", "b1", "W2", "feature_mean", "feature_std"):
            setattr(clone, name, getattr(self, name).copy())
        clone.b2 = float(self.b2)
        clone.scaler_fitted = bool(self.scaler_fitted)
        clone.target_mean = float(self.target_mean)
        clone.target_std = float(self.target_std)
        clone.target_scaler_fitted = bool(self.target_scaler_fitted)
        return clone

    def get_model_spec(self):
        return {
            "schema_version": self.SCHEMA_VERSION,
            "feature_version": TemporalFeatureAdapter.FEATURE_VERSION,
            "features": list(TemporalFeatureAdapter.FEATURE_NAMES),
            "output_channels": self.output_channels,
            "hidden_dim": self.hidden_dim,
            "prediction_horizon": self.prediction_horizon,
            "ic_loss_weight": self.ic_loss_weight,
            "mining_mode": "sequential_single",
            "random_seed": self.random_seed,
        }

    def _artifact_payload(self) -> bytes:
        if not self.scaler_fitted or not self.target_scaler_fitted:
            raise RuntimeError("Cannot export an unfitted Temporal NN model.")
        buffer = BytesIO()
        np.savez_compressed(
            buffer,
            metadata=np.asarray(json.dumps(self.get_model_spec(), sort_keys=True)),
            W1=self.W1,
            b1=self.b1,
            W2=self.W2,
            b2=np.asarray(self.b2),
            feature_mean=self.feature_mean,
            feature_std=self.feature_std,
            target_mean=np.asarray(self.target_mean),
            target_std=np.asarray(self.target_std),
        )
        return buffer.getvalue()

    def export_artifact(self) -> NNModelArtifact:
        return NNModelArtifact(
            payload=self._artifact_payload(),
            model_format=self.MODEL_FORMAT,
            extension=".npz",
            metadata=self.get_model_spec(),
        )

    @classmethod
    def from_artifact(cls, payload: bytes) -> "TemporalICFactorModel":
        with np.load(BytesIO(payload), allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            if metadata.get("schema_version") != cls.SCHEMA_VERSION:
                raise ValueError(
                    f"Unsupported Temporal NN schema: "
                    f"{metadata.get('schema_version')}"
                )
            if (
                metadata.get("feature_version")
                != TemporalFeatureAdapter.FEATURE_VERSION
            ):
                raise ValueError(
                    f"Unsupported temporal feature schema: "
                    f"{metadata.get('feature_version')}"
                )
            model = cls(
                int(metadata["output_channels"]),
                int(metadata["prediction_horizon"]),
                float(metadata["ic_loss_weight"]),
                int(metadata["random_seed"]),
            )
            model.W1 = archive["W1"].copy()
            model.b1 = archive["b1"].copy()
            model.W2 = archive["W2"].copy()
            model.b2 = float(archive["b2"].item())
            model.feature_mean = archive["feature_mean"].copy()
            model.feature_std = archive["feature_std"].copy()
            model.target_mean = float(archive["target_mean"].item())
            model.target_std = float(archive["target_std"].item())
            model.scaler_fitted = True
            model.target_scaler_fitted = True
        return model


register_nn_model_loader(
    TemporalICFactorModel.MODEL_FORMAT,
    TemporalICFactorModel.from_artifact,
)


@MinerRegistry.register("MyTemporalNN")
class MyTemporalNNMiner(MyCustomNNMiner):
    """Custom NN learning template with temporal features and IC-aware loss."""

    def initialize_search_space(self) -> None:
        if (
            self.config.get("data_feeds", {}).get(
                "mining_mode", "sequential_single"
            )
            != "sequential_single"
        ):
            raise ValueError("MyTemporalNN currently supports sequential_single only.")
        if "target" not in self.config:
            legacy_horizon = int(self.config.get("nn_prediction_horizon", 5))
            self.config["target"] = {
                "type": "forward_return",
                "entry_price": "current_close",
                "exit_price": "close",
                "horizon_bars": legacy_horizon,
                "return_type": "simple",
            }
        self.target_spec = target_from_config(self.config)
        self.prediction_horizon = self.target_spec.horizon_bars
        self.ic_loss_weight = float(self.config.get("nn_ic_loss_weight", 0.7))
        if self.prediction_horizon <= 0:
            raise ValueError("target.horizon_bars must be positive.")
        if not 0.0 <= self.ic_loss_weight <= 1.0:
            raise ValueError("nn_ic_loss_weight must be between 0 and 1.")
        super().initialize_search_space()

    def build_model(self):
        return TemporalICFactorModel(
            output_channels=self.hidden_dim,
            prediction_horizon=self.prediction_horizon,
            ic_loss_weight=self.ic_loss_weight,
            random_seed=self.random_seed,
        )

    def build_target_returns(
        self,
        data: Any,
        default_returns: Any,
        split: str,
    ) -> pd.Series:
        if not isinstance(data, pd.DataFrame):
            return pd.Series(dtype=float)
        return self.target_spec.build(data)

    def get_forward_return_definition(self) -> str:
        return self.target_spec.definition()

    def update_model(self, candidates, feedback) -> None:
        super().update_model(candidates, feedback)
        # Selection values are stored on each head; release cached full matrices
        # from every frozen snapshot so long runs remain memory-bounded.
        seen_models = set()
        for head in self._head_archive.values():
            frozen_model = getattr(head, "model_instance", None)
            if id(frozen_model) in seen_models:
                continue
            seen_models.add(id(frozen_model))
            clearer = getattr(frozen_model, "clear_prediction_cache", None)
            if callable(clearer):
                clearer()
