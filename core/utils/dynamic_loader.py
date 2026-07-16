import os
import importlib.util
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class ModuleLoadReport:
    loaded_modules: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def load_user_modules(workspace_path: str = "user_workspace"):
    """
    动态扫描并加载用户工作区下的所有 Python 文件。
    类似于 freqtrade 扫描 user_data 文件夹的方式。
    一旦被 import，文件中的 @MinerRegistry.register 等装饰器就会自动触发，将用户的类和算子注入到系统的注册中心。
    """
    workspace_dir = Path(workspace_path).absolute()
    report = ModuleLoadReport()
    if not workspace_dir.exists():
        message = f"User workspace directory {workspace_dir} not found."
        logger.warning(message)
        report.errors.append(message)
        return report

    # 定义我们要扫描的核心用户子目录
    target_dirs = ["custom_miners", "custom_operators", "custom_fitness"]
    
    loaded_count = 0
    for target in target_dirs:
        target_path = workspace_dir / target
        if not target_path.exists() or not target_path.is_dir():
            continue
            
        for file_path in target_path.rglob("*.py"):
            # 忽略 __init__.py
            if file_path.name == "__init__.py":
                continue
                
            module_name = f"{target}.{file_path.stem}"
            try:
                spec = importlib.util.spec_from_file_location(module_name, str(file_path))
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    loaded_count += 1
                    report.loaded_modules.append(module_name)
                    logger.debug(f"Successfully loaded user module: {module_name} from {file_path}")
            except Exception as e:
                message = f"Failed to load user module {file_path}: {type(e).__name__}: {e}"
                logger.error(message)
                report.errors.append(message)
                
    logger.info(f"Dynamic loader successfully imported {loaded_count} user modules from {workspace_path}.")
    return report
