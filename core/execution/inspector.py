import logging
from typing import Any
from core.storage.factor_storage import FactorStorageInterface

logger = logging.getLogger(__name__)

class FactorInspector:
    """因子白盒化审查与可视化面板"""
    def __init__(self, storage_client: FactorStorageInterface):
        self.storage = storage_client
        
    def show_tearsheet(self, factor_id: str):
        """生成因子的全身体检报告 (Tearsheet)"""
        logger.info(f"========== Tearsheet for {factor_id} ==========")
        
        # 1. 加载肉体 (Parquet 面值)
        factor_values = self.storage.load_factor_values(factor_id)
        if factor_values is not None:
            logger.info(f"Factor values loaded. Shape: {factor_values.shape if hasattr(factor_values, 'shape') else 'Unknown'}")
        else:
            logger.warning("No factor values found.")
            
        # 2. 还原灵魂 (可读性展示)
        metadata = self.storage.get_metadata(factor_id)
        if not metadata:
            logger.error("Metadata not found.")
            return
            
        miner_type = metadata.miner_type
        logger.info(f"Miner Type: {miner_type}")
        logger.info(f"Lifecycle Status: {metadata.lifecycle_status}")
        
        logic_ref = metadata.logic_reference
        if miner_type == "GP":
            ast = logic_ref.get("ast", {})
            print(f"GP Math Formula (AST): {ast}")
        elif miner_type == "LLM":
            src_file = logic_ref.get("source_file")
            print(f"LLM Generated Source File: {src_file}")
            print(f"LLM Reflection Log: {logic_ref.get('reflection')}")
        elif logic_ref.get("type") in {"nn_channel", "dl_channel"}:
            print(f"NN Extraction Channel: {logic_ref.get('channel')} from model {logic_ref.get('model_version')}")
        elif miner_type == "RL":
            print(f"RL Action Trajectory: {logic_ref.get('actions')}")
            
        logger.info("===============================================")
