"""Stable extension contracts and data adapters for neural factor miners."""

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
import json
from typing import Any, Callable, Dict, Optional, Protocol, Sequence, runtime_checkable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class NNModelArtifact:
    """Portable model payload returned by a custom neural model."""

    payload: bytes
    model_format: str
    extension: str = ".npz"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def logic_reference(self, filename: str) -> Dict[str, Any]:
        return {
            "model_file": filename,
            "model_format": self.model_format,
            **self.metadata,
        }


@runtime_checkable
class NNFactorModelProtocol(Protocol):
    """The only model-facing contract required by the shared engine."""

    def predict_channel(self, data: Any, channel_idx: int) -> Any:
        ...

    def export_artifact(self) -> NNModelArtifact:
        ...

    def clone(self) -> "NNFactorModelProtocol":
        ...


@dataclass
class PreparedNNData:
    """A dense model matrix plus enough information to restore pandas output."""

    X: np.ndarray
    y: Optional[np.ndarray]
    mode: str
    sample_index: pd.Index
    full_index: pd.Index
    full_columns: Optional[pd.Index] = None

    def restore_channel(self, values: np.ndarray) -> Any:
        values = np.asarray(values, dtype=float).reshape(-1)
        if len(values) != len(self.sample_index):
            raise ValueError(
                f"Prediction length {len(values)} does not match prepared samples "
                f"{len(self.sample_index)}."
            )

        series = pd.Series(values, index=self.sample_index, dtype=float)
        if self.mode == "cross_asset":
            result = series.unstack("asset")
            return result.reindex(index=self.full_index, columns=self.full_columns)
        return series.reindex(self.full_index)


class NNDataAdapter:
    """Convert sequential and cross-asset feeds into a common sample matrix."""

    def __init__(self, features: Sequence[str], mode: str = "sequential_single"):
        features = [str(feature) for feature in features]
        if not features:
            raise ValueError("NN data adapter requires at least one feature.")
        if len(set(features)) != len(features):
            raise ValueError("NN feature names must be unique.")
        if mode not in {"sequential_single", "cross_asset"}:
            raise ValueError(f"Unsupported NN mining mode: {mode}")
        self.features = features
        self.mode = mode

    def prepare(self, data: Any, returns: Any = None) -> PreparedNNData:
        if self.mode == "cross_asset":
            return self._prepare_cross_asset(data, returns)
        return self._prepare_sequential(data, returns)

    def _prepare_sequential(
        self, data: Any, returns: Any = None
    ) -> PreparedNNData:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Sequential NN data must be a pandas DataFrame.")

        missing = [feature for feature in self.features if feature not in data.columns]
        if missing:
            raise ValueError(f"NN input is missing required features: {missing}")

        feature_frame = data.loc[:, self.features].apply(pd.to_numeric, errors="coerce")
        full_index = feature_frame.index
        target = None
        if returns is not None:
            if not isinstance(returns, pd.Series):
                raise TypeError("Sequential NN returns must be a pandas Series.")
            target = pd.to_numeric(returns, errors="coerce").reindex(full_index)

        valid = np.isfinite(feature_frame.to_numpy(dtype=float)).all(axis=1)
        if target is not None:
            valid &= np.isfinite(target.to_numpy(dtype=float))

        valid_index = full_index[valid]
        X = feature_frame.loc[valid_index].to_numpy(dtype=float)
        y = target.loc[valid_index].to_numpy(dtype=float) if target is not None else None
        return PreparedNNData(X, y, self.mode, valid_index, full_index)

    def _prepare_cross_asset(
        self, data: Any, returns: Any = None
    ) -> PreparedNNData:
        if not isinstance(data, dict):
            raise TypeError(
                "Cross-asset NN data must be a feature-to-DataFrame mapping."
            )
        missing = [feature for feature in self.features if feature not in data]
        if missing:
            raise ValueError(f"NN input is missing required features: {missing}")

        frames = []
        for feature in self.features:
            frame = data[feature]
            if not isinstance(frame, pd.DataFrame):
                raise TypeError(
                    f"Cross-asset feature '{feature}' must be a pandas DataFrame."
                )
            frames.append(frame.apply(pd.to_numeric, errors="coerce"))

        common_index = frames[0].index
        common_columns = frames[0].columns
        for frame in frames[1:]:
            common_index = common_index.intersection(frame.index, sort=False)
            common_columns = common_columns.intersection(frame.columns, sort=False)

        target = None
        if returns is not None:
            if not isinstance(returns, pd.DataFrame):
                raise TypeError("Cross-asset NN returns must be a pandas DataFrame.")
            common_index = common_index.intersection(returns.index, sort=False)
            common_columns = common_columns.intersection(returns.columns, sort=False)
            target = returns.reindex(index=common_index, columns=common_columns)
            target = target.apply(pd.to_numeric, errors="coerce")

        if common_index.empty or common_columns.empty:
            raise ValueError("Cross-asset NN features and returns have no common samples.")

        aligned = [
            frame.reindex(index=common_index, columns=common_columns)
            for frame in frames
        ]
        full_sample_index = pd.MultiIndex.from_product(
            [common_index, common_columns], names=["timestamp", "asset"]
        )
        columns = [
            frame.stack().reindex(full_sample_index).rename(feature)
            for feature, frame in zip(self.features, aligned)
        ]
        feature_frame = pd.concat(columns, axis=1)

        target_series = None
        if target is not None:
            target_series = target.stack().reindex(full_sample_index)

        valid = np.isfinite(feature_frame.to_numpy(dtype=float)).all(axis=1)
        if target_series is not None:
            valid &= np.isfinite(target_series.to_numpy(dtype=float))

        valid_index = full_sample_index[valid]
        X = feature_frame.loc[valid_index].to_numpy(dtype=float)
        y = (
            target_series.loc[valid_index].to_numpy(dtype=float)
            if target_series is not None
            else None
        )
        return PreparedNNData(
            X,
            y,
            self.mode,
            valid_index,
            common_index,
            common_columns,
        )


class NumpyMLPFactorModel:
    """Small, serializable MLP whose hidden channels are candidate factors."""

    MODEL_FORMAT = "numpy_mlp_factor_v1"
    SCHEMA_VERSION = 1

    def __init__(
        self,
        features: Sequence[str],
        hidden_dim: int,
        mode: str = "sequential_single",
        random_seed: int = 42,
    ):
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        self.features = [str(feature) for feature in features]
        self.hidden_dim = int(hidden_dim)
        self.mode = mode
        self.random_seed = int(random_seed)
        self.adapter = NNDataAdapter(self.features, self.mode)

        rng = np.random.default_rng(self.random_seed)
        input_dim = len(self.features)
        self.W1 = rng.normal(0.0, np.sqrt(2.0 / (input_dim + hidden_dim)), (input_dim, hidden_dim))
        self.b1 = rng.normal(0.0, 0.05, hidden_dim)
        self.W2 = rng.normal(0.0, 1.0 / np.sqrt(hidden_dim), hidden_dim)
        self.b2 = 0.0
        self.feature_mean = np.zeros(input_dim, dtype=float)
        self.feature_std = np.ones(input_dim, dtype=float)
        self.scaler_fitted = False

    def fit_scaler(self, X: np.ndarray) -> None:
        if X.ndim != 2 or X.shape[1] != len(self.features):
            raise ValueError("NN training matrix has an incompatible feature dimension.")
        self.feature_mean = np.mean(X, axis=0)
        self.feature_std = np.std(X, axis=0)
        self.feature_std = np.where(self.feature_std < 1e-12, 1.0, self.feature_std)
        self.scaler_fitted = True

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.scaler_fitted:
            raise RuntimeError("NN feature scaler has not been fitted.")
        return (np.asarray(X, dtype=float) - self.feature_mean) / self.feature_std

    def _forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        normalized = self.transform(X)
        channels = np.tanh(normalized @ self.W1 + self.b1)
        prediction = channels @ self.W2 + self.b2
        return channels, prediction

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
            raise ValueError("NN training features and targets must be non-empty and aligned.")
        if not self.scaler_fitted:
            self.fit_scaler(X)

        normalized = self.transform(X)
        best_loss = float("inf")
        best_parameters = None
        stale_epochs = 0

        for _ in range(max(1, int(epochs))):
            channels = np.tanh(normalized @ self.W1 + self.b1)
            prediction = channels @ self.W2 + self.b2
            error = prediction - y
            loss = float(np.mean(error ** 2))

            d_prediction = (2.0 / len(y)) * error
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
            total_norm = np.sqrt(sum(float(np.sum(gradient ** 2)) for gradient in gradients))
            scale = min(1.0, float(gradient_clip) / (total_norm + 1e-12))
            self.W1 -= learning_rate * dW1 * scale
            self.b1 -= learning_rate * db1 * scale
            self.W2 -= learning_rate * dW2 * scale
            self.b2 -= learning_rate * db2 * scale

            if loss + 1e-12 < best_loss:
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
        return best_loss

    def predict_channel(self, data: Any, channel_idx: int) -> Any:
        if channel_idx < 0 or channel_idx >= self.hidden_dim:
            raise IndexError(f"Channel {channel_idx} is outside the model output.")
        prepared = self.adapter.prepare(data)
        channels, _ = self._forward(prepared.X)
        return prepared.restore_channel(channels[:, channel_idx])

    def clone(self) -> "NumpyMLPFactorModel":
        clone = NumpyMLPFactorModel(
            self.features, self.hidden_dim, self.mode, self.random_seed
        )
        for name in ("W1", "b1", "W2", "feature_mean", "feature_std"):
            setattr(clone, name, getattr(self, name).copy())
        clone.b2 = float(self.b2)
        clone.scaler_fitted = bool(self.scaler_fitted)
        return clone

    def get_model_spec(self) -> Dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "features": list(self.features),
            "hidden_dim": self.hidden_dim,
            "mining_mode": self.mode,
            "random_seed": self.random_seed,
        }

    def get_parameter_count(self) -> int:
        return int(self.W1.size + self.b1.size + self.W2.size + 1)

    def _artifact_payload(self) -> bytes:
        if not self.scaler_fitted:
            raise RuntimeError("Cannot export an unfitted NN model.")
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
    def from_artifact(cls, payload: bytes) -> "NumpyMLPFactorModel":
        with np.load(BytesIO(payload), allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            if metadata.get("schema_version") != cls.SCHEMA_VERSION:
                raise ValueError(
                    f"Unsupported NN model schema: {metadata.get('schema_version')}"
                )
            model = cls(
                metadata["features"],
                int(metadata["hidden_dim"]),
                metadata["mining_mode"],
                int(metadata["random_seed"]),
            )
            model.W1 = archive["W1"].copy()
            model.b1 = archive["b1"].copy()
            model.W2 = archive["W2"].copy()
            model.b2 = float(archive["b2"].item())
            model.feature_mean = archive["feature_mean"].copy()
            model.feature_std = archive["feature_std"].copy()
            model.scaler_fitted = True
        return model


_NN_MODEL_LOADERS: Dict[str, Callable[[bytes], NNFactorModelProtocol]] = {
    NumpyMLPFactorModel.MODEL_FORMAT: NumpyMLPFactorModel.from_artifact,
}


def register_nn_model_loader(
    model_format: str,
    loader: Callable[[bytes], NNFactorModelProtocol],
) -> None:
    """Register a loader for a user-defined portable NN model format."""
    if not model_format or not callable(loader):
        raise ValueError("NN model loader registration requires a format and callable.")
    _NN_MODEL_LOADERS[model_format] = loader


def load_nn_model(model_format: str, payload: bytes) -> NNFactorModelProtocol:
    """Restore a built-in or user-registered NN model artifact."""
    loader = _NN_MODEL_LOADERS.get(model_format)
    if loader is None:
        available = ", ".join(sorted(_NN_MODEL_LOADERS)) or "none"
        raise ValueError(
            f"Unknown NN model format '{model_format}'. Available formats: {available}."
        )
    return loader(payload)
