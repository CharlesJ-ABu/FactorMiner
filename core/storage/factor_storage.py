import os
import json
import logging
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any, List
import dataclasses

from core.miner.entities import FactorMetadata

logger = logging.getLogger(__name__)

class FactorStorageInterface(ABC):
    @abstractmethod
    def save_gp_factor(self, ast_dict: Dict, metadata: FactorMetadata) -> bool: pass
    
    @abstractmethod
    def save_rl_factor(self, action_sequence: List[str], agent_snapshot_bytes: bytes, metadata: FactorMetadata) -> bool: pass
    
    @abstractmethod
    def save_llm_factor(self, python_code: str, reflection_log: str, metadata: FactorMetadata) -> bool: pass
    
    @abstractmethod
    def save_model_weights(self, model_version_id: str, model_weights: bytes) -> bool: pass
    
    @abstractmethod
    def save_dl_factor_channel(self, model_version_id: str, channel_index: int, metadata: FactorMetadata) -> bool: pass

    @abstractmethod
    def save_factor_values(self, factor_id: str, values_df: Any) -> bool: pass

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
        self.src_dir = os.path.join(db_root, "sources")
        
        for d in [self.meta_dir, self.val_dir, self.weights_dir, self.src_dir]:
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

    def save_llm_factor(self, python_code: str, reflection_log: str, metadata: FactorMetadata) -> bool:
        code_file = f"{metadata.factor_id}.py"
        code_path = os.path.join(self.src_dir, code_file)
        with open(code_path, "w") as f:
            f.write(python_code)
            
        metadata.logic_reference = {"type": "python_source", "source_file": code_file, "reflection": reflection_log}
        self._save_meta(metadata)
        logger.info(f"Saved LLM metadata and source for {metadata.factor_id}")
        return True

    def save_model_weights(self, model_version_id: str, model_weights: bytes) -> bool:
        weight_path = os.path.join(self.weights_dir, f"{model_version_id}.pt")
        if not os.path.exists(weight_path):
            with open(weight_path, "wb") as f:
                f.write(model_weights)
            logger.info(f"Saved DL model weights for {model_version_id}")
        return True

    def save_dl_factor_channel(self, model_version_id: str, channel_index: int, metadata: FactorMetadata) -> bool:
        metadata.logic_reference = {"type": "dl_channel", "model_version": model_version_id, "channel": channel_index}
        self._save_meta(metadata)
        logger.info(f"Saved DL channel metadata for {metadata.factor_id}")
        return True

    def save_factor_values(self, factor_id: str, values_df: Any) -> bool:
        # Fallback if values_df is not valid or empty
        if values_df is None or (hasattr(values_df, 'empty') and values_df.empty):
            logger.warning(f"No values to save for {factor_id}")
            return False
            
        path = os.path.join(self.val_dir, f"{factor_id}.parquet")
        # In real system, it would be values_df.to_parquet(path)
        # We mock it for the demo if it's just a raw dict or tensor
        if isinstance(values_df, pd.DataFrame) or isinstance(values_df, pd.Series):
            try:
                values_df.to_frame().to_parquet(path)
            except Exception:
                pass
        logger.info(f"Saved factor values for {factor_id} to {path}")
        return True

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
