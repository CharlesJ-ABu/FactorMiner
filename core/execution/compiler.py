import logging
from typing import Any
from core.evaluation.code_sandbox import RestrictedSandbox
from core.miner.expressions import (
    FactorExpressionAST,
    FactorExpressionCode,
)
from core.storage.factor_storage import FactorStorageInterface

logger = logging.getLogger(__name__)

class FactorCompiler:
    """
    因子上线编译器：将存储的因子逻辑转化为实盘极速推理模块
    """
    def __init__(
        self,
        storage_client: FactorStorageInterface,
        sandbox: RestrictedSandbox = None,
    ):
        self.storage = storage_client
        self.sandbox = sandbox or RestrictedSandbox()

    def compile_for_live_trading(self, factor_id: str) -> Any:
        metadata = self.storage.get_metadata(factor_id)
        if not metadata:
            raise ValueError(f"Factor {factor_id} not found in storage.")
            
        miner_type = metadata.miner_type
        logic_ref = metadata.logic_reference

        logic_type = logic_ref.get("type")

        # Dispatch by the persisted logic contract, not a customizable miner name.
        if logic_type in {"nn_channel", "dl_channel"}:
            model_file = logic_ref.get("model_file")
            model_format = logic_ref.get("model_format")
            channel = logic_ref.get("channel")
            if model_file and model_format:
                from core.miner.nn import load_nn_model

                payload = self.storage.load_model_artifact(model_file)
                model = load_nn_model(model_format, payload)

                def compiled_nn_factor(data):
                    return model.predict_channel(data, channel)

                return compiled_nn_factor
            raise ValueError(
                f"NN factor {factor_id} uses a legacy artifact that cannot be "
                "reconstructed. Re-run mining to create a portable model bundle."
            )

        if logic_type in {"json_ast", "ast"}:
            ast_dict = logic_ref.get("ast")
            if not isinstance(ast_dict, dict):
                raise ValueError(f"Factor {factor_id} contains no valid AST.")
            expression = FactorExpressionAST(ast_dict=ast_dict)
            return expression.compute

        if logic_type == "python_source":
            source_file = logic_ref.get("source_file")
            if not source_file:
                raise ValueError(f"LLM factor {factor_id} contains no source file.")
            code_str = self.storage.load_llm_source(source_file)
            self.sandbox.validate_code(code_str)
            expression = FactorExpressionCode(
                code_str=code_str,
                sandbox=self.sandbox,
                reflection_history=logic_ref.get("reflection", ""),
                provenance=logic_ref.get("provenance", {}),
            )
            return expression.compute

        if logic_type == "rl_actions":
            raise NotImplementedError(
                "RL action-only artifacts cannot be reconstructed into a live "
                "factor. Persist the resulting AST expression instead."
            )
        
        if miner_type == "GP":
            ast_dict = logic_ref.get("ast", {})
            logger.info("Compiling legacy GP AST for %s.", factor_id)
            return self._compile_ast_to_numexpr(ast_dict)
            
        elif miner_type == "LLM":
            raise ValueError(
                f"Legacy LLM factor {factor_id} has no python_source contract."
            )
            
        elif miner_type == "DL":
            raise ValueError(
                f"Legacy DL factor {factor_id} has no portable NN model artifact."
            )
            
        elif miner_type == "RL":
            raise NotImplementedError(
                "Legacy RL actions are not an executable factor artifact."
            )
            
        else:
            raise ValueError(f"Unsupported miner type: {miner_type}")

    def _compile_ast_to_numexpr(self, ast_dict: dict):
        if not isinstance(ast_dict, dict):
            raise ValueError("GP AST must be a dictionary.")
        return FactorExpressionAST(ast_dict=ast_dict).compute

    def deploy_to_live_server(self, factor_id: str, server_target: str):
        # Validate that the artifact can be reconstructed, but never report a
        # deployment that this repository has not actually performed.
        self.compile_for_live_trading(factor_id)
        raise NotImplementedError(
            "Live server deployment transport is not implemented. "
            f"Factor {factor_id} was validated but was not deployed to {server_target}."
        )
