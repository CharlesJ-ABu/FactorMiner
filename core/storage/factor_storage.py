import os
import json
import logging
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any, List
import dataclasses

from core.miner.entities import FactorMetadata

logger = logging.getLogger(__name__)

_SENSITIVE_PROVENANCE_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "password",
    "secret",
    "token",
}


def _sanitize_provenance(value: Any) -> Any:
    """Remove credentials recursively before generation metadata reaches disk."""
    if isinstance(value, dict):
        return {
            key: _sanitize_provenance(item)
            for key, item in value.items()
            if not any(
                marker in str(key).lower().replace("-", "_")
                for marker in _SENSITIVE_PROVENANCE_KEYS
            )
        }
    if isinstance(value, list):
        return [_sanitize_provenance(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_provenance(item) for item in value]
    return value


class FactorStorageInterface(ABC):
    @abstractmethod
    def save_gp_factor(self, ast_dict: Dict, metadata: FactorMetadata) -> bool: pass
    
    @abstractmethod
    def save_rl_factor(self, action_sequence: List[str], agent_snapshot_bytes: bytes, metadata: FactorMetadata) -> bool: pass
    
    @abstractmethod
    def save_llm_factor(
        self,
        python_code: str,
        reflection_log: str,
        metadata: FactorMetadata,
        provenance: Dict[str, Any] = None,
    ) -> bool: pass
    
    @abstractmethod
    def save_model_weights(self, model_version_id: str, model_weights: bytes) -> bool: pass

    def save_model_artifact(
        self,
        model_version_id: str,
        payload: bytes,
        extension: str = ".npz",
    ) -> str:
        """Persist a portable NN artifact and return its relative filename."""
        raise NotImplementedError

    def load_model_artifact(self, filename: str) -> bytes:
        """Load a portable NN artifact by its storage-relative filename."""
        raise NotImplementedError

    def load_llm_source(self, filename: str) -> str:
        """Load persisted LLM source by its storage-relative filename."""
        raise NotImplementedError
    
    def save_nn_factor_channel(
        self,
        model_version_id: str,
        channel_index: int,
        metadata: FactorMetadata,
        artifact_reference: Dict[str, Any] = None,
    ) -> bool:
        """Persist an NN channel; legacy backends may implement the old alias."""
        return self.save_dl_factor_channel(
            model_version_id,
            channel_index,
            metadata,
            artifact_reference,
        )

    def save_dl_factor_channel(
        self,
        model_version_id: str,
        channel_index: int,
        metadata: FactorMetadata,
        artifact_reference: Dict[str, Any] = None,
    ) -> bool:
        """Deprecated storage-backend compatibility hook."""
        raise NotImplementedError(
            "Implement save_nn_factor_channel(); save_dl_factor_channel() is deprecated."
        )

    @abstractmethod
    def save_factor_values(self, factor_id: str, values_df: Any) -> bool: pass

    @abstractmethod
    def save_factor_snapshot(
        self,
        factor_id: str,
        factor_values: Any,
        forward_returns: Any,
        data_lineage: Dict[str, Any],
    ) -> bool: pass

    @abstractmethod
    def get_metadata(self, factor_id: str) -> FactorMetadata: pass

    @abstractmethod
    def list_metadata(self) -> List[FactorMetadata]: pass

    @abstractmethod
    def update_lifecycle_status(self, factor_id: str, lifecycle_status: str) -> FactorMetadata: pass

    @abstractmethod
    def load_factor_values(self, factor_id: str) -> Any: pass
    
    @abstractmethod
    def get_all_logic_hashes(self) -> set[str]: pass


class LocalFactorStorage(FactorStorageInterface):
    """
    V4 本地异构存储实现
    """
    def __init__(self, db_root="factor_db"):
        self.meta_dir = os.path.join(db_root, "metadata")
        self.val_dir = os.path.join(db_root, "values")
        self.weights_dir = os.path.join(db_root, "weights")
        self.models_dir = os.path.join(db_root, "models")
        self.src_dir = os.path.join(db_root, "sources")
        
        for d in [self.meta_dir, self.val_dir, self.weights_dir, self.models_dir, self.src_dir]:
            os.makedirs(d, exist_ok=True)
            
    def _save_meta(self, metadata: FactorMetadata):
        path = os.path.join(self.meta_dir, f"{metadata.factor_id}.json")
        with open(path, "w") as f:
            json.dump(dataclasses.asdict(metadata), f, indent=4)
            
    def save_gp_factor(self, ast_dict: Dict, metadata: FactorMetadata) -> bool:
        metadata.logic_reference = {"type": "json_ast", "ast": ast_dict}
        self._save_meta(metadata)
        logger.info(f"Saved GP metadata for {metadata.factor_id}")
        return True

    def save_rl_factor(self, action_sequence: List[str], agent_snapshot_bytes: bytes, metadata: FactorMetadata) -> bool:
        weight_file = f"{metadata.factor_id}_agent.pt"
        weight_path = os.path.join(self.weights_dir, weight_file)
        if agent_snapshot_bytes:
            with open(weight_path, "wb") as f:
                f.write(agent_snapshot_bytes)
                
        metadata.logic_reference = {"type": "rl_actions", "actions": action_sequence, "weights_file": weight_file}
        self._save_meta(metadata)
        logger.info(f"Saved RL metadata for {metadata.factor_id}")
        return True

    def save_llm_factor(
        self,
        python_code: str,
        reflection_log: str,
        metadata: FactorMetadata,
        provenance: Dict[str, Any] = None,
    ) -> bool:
        code_file = f"{metadata.factor_id}.py"
        code_path = os.path.join(self.src_dir, code_file)
        with open(code_path, "w") as f:
            f.write(python_code)
            
        metadata.logic_reference = {
            "type": "python_source",
            "source_file": code_file,
            "reflection": reflection_log,
            "provenance": _sanitize_provenance(provenance or {}),
        }
        self._save_meta(metadata)
        logger.info(f"Saved LLM metadata and source for {metadata.factor_id}")
        return True

    def save_model_weights(self, model_version_id: str, model_weights: bytes) -> bool:
        weight_path = os.path.join(self.weights_dir, f"{model_version_id}.pt")
        if not os.path.exists(weight_path):
            with open(weight_path, "wb") as f:
                f.write(model_weights)
            logger.info(f"Saved NN model weights for {model_version_id}")
        return True

    def save_model_artifact(
        self,
        model_version_id: str,
        payload: bytes,
        extension: str = ".npz",
    ) -> str:
        if not extension.startswith(".") or "/" in extension or "\\" in extension:
            raise ValueError(f"Invalid model artifact extension: {extension}")
        filename = f"{model_version_id}{extension}"
        model_path = os.path.join(self.models_dir, filename)
        if not os.path.exists(model_path):
            with open(model_path, "wb") as model_file:
                model_file.write(payload)
            logger.info("Saved NN model artifact for %s", model_version_id)
        return filename

    def load_model_artifact(self, filename: str) -> bytes:
        safe_filename = os.path.basename(filename)
        if safe_filename != filename:
            raise ValueError("Model artifact filename must not contain a path.")
        model_path = os.path.join(self.models_dir, safe_filename)
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"NN model artifact not found: {safe_filename}")
        with open(model_path, "rb") as model_file:
            return model_file.read()

    def load_llm_source(self, filename: str) -> str:
        safe_filename = os.path.basename(filename)
        if safe_filename != filename:
            raise ValueError("LLM source filename must not contain a path.")
        source_path = os.path.join(self.src_dir, safe_filename)
        if not os.path.isfile(source_path):
            raise FileNotFoundError(f"LLM source not found: {safe_filename}")
        with open(source_path, "r", encoding="utf-8") as source_file:
            return source_file.read()

    def save_nn_factor_channel(
        self,
        model_version_id: str,
        channel_index: int,
        metadata: FactorMetadata,
        artifact_reference: Dict[str, Any] = None,
    ) -> bool:
        metadata.logic_reference = {
            "type": "nn_channel",
            "model_version": model_version_id,
            "channel": channel_index,
            **(artifact_reference or {}),
        }
        self._save_meta(metadata)
        logger.info(f"Saved NN channel metadata for {metadata.factor_id}")
        return True

    def save_dl_factor_channel(
        self,
        model_version_id: str,
        channel_index: int,
        metadata: FactorMetadata,
        artifact_reference: Dict[str, Any] = None,
    ) -> bool:
        """Compatibility alias for pre-NN storage callers."""
        logger.warning(
            "save_dl_factor_channel() is deprecated; use save_nn_factor_channel()."
        )
        return self.save_nn_factor_channel(
            model_version_id,
            channel_index,
            metadata,
            artifact_reference,
        )

    def save_factor_values(self, factor_id: str, values_df: Any) -> bool:
        """Legacy helper retained for callers that only have factor values."""
        if isinstance(values_df, pd.Series):
            returns = pd.Series(index=values_df.index, dtype=float)
        elif isinstance(values_df, pd.DataFrame):
            returns = pd.DataFrame(index=values_df.index, columns=values_df.columns, dtype=float)
        else:
            logger.warning("No tabular factor values to save for %s", factor_id)
            return False
        return self.save_factor_snapshot(factor_id, values_df, returns, {})

    @staticmethod
    def _snapshot_frame(factor_values: Any, forward_returns: Any) -> pd.DataFrame:
        """Normalize sequential and cross-asset values into a stable parquet schema."""
        if isinstance(factor_values, pd.DataFrame) and isinstance(forward_returns, pd.DataFrame):
            factors = factor_values.rename_axis(index="timestamp", columns="asset").stack().rename("factor")
            returns = forward_returns.rename_axis(index="timestamp", columns="asset").stack().rename("forward_return")
            snapshot = pd.concat([factors, returns], axis=1).reset_index()
        elif isinstance(factor_values, pd.Series) and isinstance(forward_returns, pd.Series):
            snapshot = pd.concat(
                [
                    factor_values.rename("factor"),
                    forward_returns.rename("forward_return"),
                ],
                axis=1,
            ).reset_index()
            snapshot = snapshot.rename(columns={snapshot.columns[0]: "timestamp"})
        else:
            raise TypeError(
                "Factor snapshot requires factor values and forward returns with matching pandas Series or DataFrame types."
            )

        snapshot["timestamp"] = pd.to_datetime(snapshot["timestamp"], errors="coerce")
        snapshot["factor"] = pd.to_numeric(snapshot["factor"], errors="coerce")
        snapshot["forward_return"] = pd.to_numeric(snapshot["forward_return"], errors="coerce")
        snapshot = snapshot.dropna(subset=["timestamp"]).sort_values(
            ["timestamp", "asset"] if "asset" in snapshot.columns else ["timestamp"]
        )
        return snapshot

    def save_factor_snapshot(
        self,
        factor_id: str,
        factor_values: Any,
        forward_returns: Any,
        data_lineage: Dict[str, Any],
    ) -> bool:
        try:
            snapshot = self._snapshot_frame(factor_values, forward_returns)
            if snapshot.empty:
                logger.warning("No aligned snapshot rows to save for %s", factor_id)
                return False

            path = os.path.join(self.val_dir, f"{factor_id}.parquet")
            snapshot.to_parquet(path, index=False)

            metadata = self.get_metadata(factor_id)
            if metadata:
                metadata.data_lineage = {
                    **data_lineage,
                    "snapshot_file": os.path.basename(path),
                    "snapshot_schema": list(snapshot.columns),
                    "snapshot_rows": int(len(snapshot)),
                    "snapshot_start": snapshot["timestamp"].min().isoformat(),
                    "snapshot_end": snapshot["timestamp"].max().isoformat(),
                }
                self._save_meta(metadata)

            logger.info("Saved factor snapshot for %s to %s", factor_id, path)
            return True
        except Exception as exc:
            logger.warning("Failed to save factor snapshot for %s: %s", factor_id, exc)
            return False

    def get_metadata(self, factor_id: str) -> FactorMetadata:
        path = os.path.join(self.meta_dir, f"{factor_id}.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
            return FactorMetadata(**data)
        return None

    def list_metadata(self) -> List[FactorMetadata]:
        """Return every readable factor record for the Inspector catalog."""
        metadata_items = []
        for filename in os.listdir(self.meta_dir):
            if not filename.endswith(".json"):
                continue
            factor_id = os.path.splitext(filename)[0]
            try:
                metadata = self.get_metadata(factor_id)
                if metadata:
                    metadata_items.append(metadata)
            except Exception as exc:
                logger.warning("Skipping unreadable factor metadata %s: %s", filename, exc)
        return metadata_items

    def update_lifecycle_status(self, factor_id: str, lifecycle_status: str) -> FactorMetadata:
        metadata = self.get_metadata(factor_id)
        if not metadata:
            return None
        metadata.lifecycle_status = lifecycle_status
        self._save_meta(metadata)
        logger.info("Updated lifecycle status for %s to %s", factor_id, lifecycle_status)
        return metadata

    def load_factor_values(self, factor_id: str) -> Any:
        path = os.path.join(self.val_dir, f"{factor_id}.parquet")
        if os.path.exists(path):
            return pd.read_parquet(path)
        return None

    def get_all_logic_hashes(self) -> set[str]:
        hashes = set()
        for filename in os.listdir(self.meta_dir):
            if filename.endswith(".json"):
                path = os.path.join(self.meta_dir, filename)
                try:
                    with open(path, "r") as f:
                        data = json.load(f)
                        if "logic_hash" in data and data["logic_hash"]:
                            hashes.add(data["logic_hash"])
                except Exception as e:
                    logger.error(f"Failed to read logic_hash from {path}: {e}")
        return hashes


def get_global_storage() -> FactorStorageInterface:
    return LocalFactorStorage()
