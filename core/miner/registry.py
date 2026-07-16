import inspect
from typing import Callable, Dict


class ExtensionRegistrationError(ValueError):
    """Raised when a user extension cannot satisfy its registry contract."""


def _validate_callable_arity(func: Callable, expected_arity: int, extension_type: str) -> None:
    if not callable(func):
        raise ExtensionRegistrationError(f"{extension_type} must be callable.")
    try:
        inspect.signature(func).bind(*([object()] * expected_arity))
    except TypeError as exc:
        name = getattr(func, "__name__", repr(func))
        raise ExtensionRegistrationError(
            f"{extension_type} '{name}' must accept at least {expected_arity} positional argument(s): {exc}"
        ) from exc

class OperatorRegistry:
    _registry = {}
    
    @classmethod
    def register(cls, arity: int = 1):
        if arity not in (1, 2):
            raise ExtensionRegistrationError(
                f"Custom operator arity must be 1 or 2, received {arity}."
            )

        def decorator(func: Callable):
            _validate_callable_arity(func, arity, "Custom operator")
            cls._registry[func.__name__] = {"func": func, "arity": arity}
            return func
        return decorator

class EvaluatorRegistry:
    _registry = {}
    
    @classmethod
    def register_fitness_hook(cls, hook_name: str):
        if not isinstance(hook_name, str) or not hook_name.strip():
            raise ExtensionRegistrationError("Fitness Hook name must be a non-empty string.")

        def decorator(func: Callable):
            _validate_callable_arity(func, 3, "Fitness Hook")
            cls._registry[hook_name] = func
            return func
        return decorator

class MinerRegistry:
    _registry = {}
    
    @classmethod
    def register(cls, paradigm_name: str):
        if not isinstance(paradigm_name, str) or not paradigm_name.strip():
            raise ExtensionRegistrationError("Miner name must be a non-empty string.")

        def decorator(miner_cls: type):
            from core.miner.paradigms.base import BaseFactorMiner

            if not inspect.isclass(miner_cls) or not issubclass(miner_cls, BaseFactorMiner):
                raise ExtensionRegistrationError(
                    f"Miner '{paradigm_name}' must inherit from BaseFactorMiner."
                )
            cls._registry[paradigm_name] = miner_cls
            return miner_cls
        return decorator
