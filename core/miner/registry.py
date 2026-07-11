from typing import Callable, Dict

class OperatorRegistry:
    _registry = {}
    
    @classmethod
    def register(cls, arity: int = 1):
        def decorator(func: Callable):
            cls._registry[func.__name__] = {"func": func, "arity": arity}
            return func
        return decorator

class EvaluatorRegistry:
    _registry = {}
    
    @classmethod
    def register_fitness_hook(cls, hook_name: str):
        def decorator(func: Callable):
            cls._registry[hook_name] = func
            return func
        return decorator

class MinerRegistry:
    _registry = {}
    
    @classmethod
    def register(cls, paradigm_name: str):
        def decorator(miner_cls: type):
            cls._registry[paradigm_name] = miner_cls
            return miner_cls
        return decorator
