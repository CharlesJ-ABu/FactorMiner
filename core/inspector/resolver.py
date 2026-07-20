import ast
import json
import logging
from typing import Any, Dict, Union

from core.miner.expressions import FactorExpression, FactorExpressionAST, FactorExpressionCode
from core.storage.factor_storage import FactorStorageInterface, LocalFactorStorage

logger = logging.getLogger(__name__)


class FactorResolver:
    """
    因子解析器：将 Factor ID、AST 字典字符串或 Python 代码解析为统一的 FactorExpression 对象。
    """
    def __init__(self, storage: FactorStorageInterface = None):
        self.storage = storage or LocalFactorStorage()

    def resolve(
        self,
        factor_id: str = None,
        ast_str: str = None,
        code_str: str = None,
        ast_dict: Dict = None,
    ) -> FactorExpression:
        """
        根据不同的输入形式，解析并返回一个 FactorExpression 实例。
        """
        # 1. 优先从 Factor ID 从数据库/本地存储中解包
        if factor_id:
            logger.info("Resolving factor from Factor ID: %s", factor_id)
            meta = self.storage.load_factor_metadata(factor_id)
            if not meta:
                raise ValueError(f"Factor ID '{factor_id}' not found in storage.")
                
            ast_payload = meta.get("ast_dict") or meta.get("expression")
            if isinstance(ast_payload, dict):
                return FactorExpressionAST(ast_dict=ast_payload)
            elif isinstance(ast_payload, str):
                try:
                    parsed = json.loads(ast_payload)
                    return FactorExpressionAST(ast_dict=parsed)
                except Exception:
                    try:
                        parsed = ast.literal_eval(ast_payload)
                        return FactorExpressionAST(ast_dict=parsed)
                    except Exception as e:
                        raise ValueError(f"Failed to parse metadata AST for factor '{factor_id}': {e}")
            elif "code" in meta:
                return FactorExpressionCode(code_str=meta["code"])
            else:
                raise ValueError(f"Metadata for factor '{factor_id}' contains neither valid AST nor code.")

        # 2. 直接从传入的 ast_dict 或 ast_str 解析
        if ast_dict:
            return FactorExpressionAST(ast_dict=ast_dict)

        if ast_str:
            logger.info("Resolving factor from AST string...")
            try:
                parsed = json.loads(ast_str)
                return FactorExpressionAST(ast_dict=parsed)
            except Exception:
                try:
                    parsed = ast.literal_eval(ast_str)
                    return FactorExpressionAST(ast_dict=parsed)
                except Exception as e:
                    raise ValueError(f"Failed to parse AST string: {e}")

        # 3. 从 Python 代码解析
        if code_str:
            logger.info("Resolving factor from Python code string...")
            return FactorExpressionCode(code_str=code_str)

        raise ValueError("Must provide at least one of factor_id, ast_str, ast_dict, or code_str to resolve a factor.")
