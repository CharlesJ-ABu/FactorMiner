import logging
from typing import Any
from core.storage.factor_storage import FactorStorageInterface

logger = logging.getLogger(__name__)

class FactorCompiler:
    """
    因子上线编译器：将存储的因子逻辑转化为实盘极速推理模块
    """
    def __init__(self, storage_client: FactorStorageInterface):
        self.storage = storage_client

    def compile_for_live_trading(self, factor_id: str) -> Any:
        metadata = self.storage.get_metadata(factor_id)
        if not metadata:
            raise ValueError(f"Factor {factor_id} not found in storage.")
            
        miner_type = metadata.miner_type
        logic_ref = metadata.logic_reference
        
        if miner_type == "GP":
            ast_dict = logic_ref.get("ast", {})
            logger.info(f"Compiling GP AST for {factor_id} into NumExpr/C++...")
            return self._compile_ast_to_numexpr(ast_dict)
            
        elif miner_type == "LLM":
            code_str = ""
            src_file = logic_ref.get("source_file")
            # 在实际系统中，应该从存储加载 src_file 内容，此处仅演示逻辑
            logger.info(f"Compiling LLM python source for {factor_id} into bytecode...")
            try:
                # 假设源码可以读到
                # code_str = self.storage.read_file(src_file)
                pass
            except Exception:
                code_str = "factor = 0"
            return compile(code_str, f"<{factor_id}>", "exec")
            
        elif miner_type == "DL":
            model_version = logic_ref.get("model_version")
            logger.info(f"Exporting DL model {model_version} to ONNX/TensorRT...")
            return self._export_to_onnx(model_version)
            
        elif miner_type == "RL":
            actions = logic_ref.get("actions", [])
            logger.info(f"Compiling RL actions {actions} into sequence executor...")
            return actions
            
        else:
            raise ValueError(f"Unsupported miner type: {miner_type}")

    def _compile_ast_to_numexpr(self, ast_dict: dict):
        return "numexpr_engine_placeholder"
        
    def _export_to_onnx(self, model_version: str):
        return "onnx_engine_placeholder"

    def deploy_to_live_server(self, factor_id: str, server_target: str):
        compiled_engine = self.compile_for_live_trading(factor_id)
        
        # 统一包裹为 gRPC 接口
        wrapped_engine = self._wrap_with_grpc_interface(compiled_engine)
        
        logger.info(f"Deploying wrapped engine for {factor_id} to {server_target}...")
        return {"status": "success", "endpoint": server_target}

    def _wrap_with_grpc_interface(self, engine: Any):
        # 封装 C-Wrapper 或 gRPC 接口
        return f"GrpcWrapper({engine})"
