"""Fail-closed execution boundary for LLM-generated factor code."""

from __future__ import annotations

import ast
import multiprocessing
import traceback
from dataclasses import dataclass
from typing import Any, Union

import pandas as pd


class SecurityError(Exception):
    """Generated code violates the static execution policy."""


class SandboxExecutionError(Exception):
    """Generated code failed inside the isolated worker."""


class SandboxTimeoutError(SandboxExecutionError):
    """Generated code exceeded its wall-clock budget."""


class FactorOutputError(ValueError):
    """Generated code returned an object that violates the factor contract."""


@dataclass(frozen=True)
class SandboxLimits:
    timeout_seconds: float = 5.0
    cpu_seconds: int = 3
    memory_mb: int = 1024


_SAFE_BUILTINS = {
    "abs",
    "min",
    "max",
    "len",
    "range",
    "list",
    "dict",
    "float",
    "int",
}

_SAFE_ATTRIBUTES = {
    # pandas Series/DataFrame transforms used by factor formulas.
    "abs",
    "add",
    "clip",
    "corr",
    "cummax",
    "cummin",
    "cumprod",
    "cumsum",
    "diff",
    "div",
    "divide",
    "dropna",
    "ewm",
    "expanding",
    "fillna",
    "index",
    "isna",
    "log",
    "max",
    "mean",
    "median",
    "min",
    "mul",
    "multiply",
    "notna",
    "pct_change",
    "pow",
    "quantile",
    "rank",
    "replace",
    "rolling",
    "shift",
    "sign",
    "std",
    "sub",
    "subtract",
    "sum",
    "var",
    "where",
    "winsorize",
    # Explicit NumPy/Pandas entry points.
    "Series",
    "DataFrame",
    "array",
    "exp",
    "log1p",
    "maximum",
    "minimum",
    "nan",
    "sqrt",
}

_ALLOWED_NODE_TYPES = (
    ast.Module,
    ast.Assign,
    ast.Expr,
    ast.Name,
    ast.Load,
    ast.Store,
    ast.Constant,
    ast.Subscript,
    ast.Slice,
    ast.Tuple,
    ast.List,
    ast.Dict,
    ast.Call,
    ast.Attribute,
    ast.keyword,
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.IfExp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    ast.Invert,
    ast.And,
    ast.Or,
    ast.BitAnd,
    ast.BitOr,
    ast.BitXor,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
)


class FactorCodePolicy(ast.NodeVisitor):
    """Small AST allowlist for vectorized pandas/numpy factor expressions."""

    def __init__(self) -> None:
        self.assigned_names: set[str] = set()
        self.factor_assigned = False

    def validate(self, code_str: str) -> ast.Module:
        try:
            tree = ast.parse(code_str, mode="exec")
        except SyntaxError as exc:
            raise SecurityError(f"Generated factor code has invalid syntax: {exc.msg}") from exc
        self.visit(tree)
        if not self.factor_assigned:
            raise FactorOutputError(
                "The generated code must assign the result to the variable 'factor'"
            )
        return tree

    def generic_visit(self, node: ast.AST) -> None:
        if not isinstance(node, _ALLOWED_NODE_TYPES):
            raise SecurityError(
                f"Generated factor code uses forbidden syntax: {type(node).__name__}"
            )
        super().generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if not isinstance(target, ast.Name):
                raise SecurityError("Assignments may target local variable names only")
            if target.id in {"df", "np", "pd"} or target.id in _SAFE_BUILTINS:
                raise SecurityError(f"Assignment to protected name '{target.id}' is forbidden")
            if target.id.startswith("_"):
                raise SecurityError("Private or dunder variable names are forbidden")
            self.assigned_names.add(target.id)
            if target.id == "factor":
                self.factor_assigned = True
        self.visit(node.value)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id.startswith("_"):
            raise SecurityError("Private or dunder names are forbidden")
        if isinstance(node.ctx, ast.Load):
            allowed = {"df", "np", "pd"} | _SAFE_BUILTINS | self.assigned_names
            if node.id not in allowed:
                raise SecurityError(f"Use of name '{node.id}' is not allowed")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("_") or node.attr not in _SAFE_ATTRIBUTES:
            raise SecurityError(f"Attribute '{node.attr}' is not allowed")
        self.visit(node.value)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            if node.func.id not in _SAFE_BUILTINS:
                raise SecurityError(f"Call to '{node.func.id}' is not allowed")
        elif not isinstance(node.func, ast.Attribute):
            raise SecurityError("Only approved builtin and attribute calls are allowed")
        self.visit(node.func)
        for argument in node.args:
            self.visit(argument)
        for keyword in node.keywords:
            if keyword.arg is None or keyword.arg.startswith("_"):
                raise SecurityError("Dynamic or private keyword arguments are forbidden")
            self.visit(keyword.value)


def _validate_finite_numeric(value: Union[pd.Series, pd.DataFrame]) -> None:
    dtypes = [value.dtype] if isinstance(value, pd.Series) else list(value.dtypes)
    if any(not pd.api.types.is_numeric_dtype(dtype) for dtype in dtypes):
        raise FactorOutputError("LLM factor output must contain numeric values only")
    if int(value.notna().to_numpy().sum()) == 0:
        raise FactorOutputError("LLM factor output must contain at least one finite value")
    numeric = value.to_numpy(dtype=float, na_value=float("nan"))
    import numpy as np

    if bool((~np.isnan(numeric) & ~np.isfinite(numeric)).any()):
        raise FactorOutputError("LLM factor output must not contain infinite values")


def validate_llm_factor_output(
    value: Any,
    data: Any,
) -> Union[pd.Series, pd.DataFrame]:
    """Enforce the code-expression contract before returning across the boundary."""
    if isinstance(data, pd.DataFrame):
        if not isinstance(value, pd.Series):
            raise FactorOutputError(
                "Sequential LLM factor code must return pandas.Series, "
                f"received {type(value).__name__}"
            )
        if len(value) != len(data.index):
            raise FactorOutputError(
                f"LLM factor length {len(value)} does not match input length "
                f"{len(data.index)}"
            )
        if not value.index.equals(data.index):
            raise FactorOutputError("LLM factor index must exactly match the input df index")
        if value.index.has_duplicates:
            raise FactorOutputError("LLM factor index must not contain duplicates")
        _validate_finite_numeric(value)
        return value

    if not isinstance(data, dict) or not data:
        raise FactorOutputError(
            "LLM factor input must be a DataFrame or a non-empty feature mapping"
        )
    feature_frames = list(data.values())
    if any(not isinstance(frame, pd.DataFrame) for frame in feature_frames):
        raise FactorOutputError(
            "Cross-asset LLM inputs must map every feature to pandas.DataFrame"
        )
    reference = feature_frames[0]
    for frame in feature_frames[1:]:
        if (
            not frame.index.equals(reference.index)
            or not frame.columns.equals(reference.columns)
        ):
            raise FactorOutputError(
                "Cross-asset LLM feature frames must have identical indexes and columns"
            )
    if not isinstance(value, pd.DataFrame):
        raise FactorOutputError(
            "Cross-asset LLM factor code must return pandas.DataFrame, "
            f"received {type(value).__name__}"
        )
    if (
        not value.index.equals(reference.index)
        or not value.columns.equals(reference.columns)
    ):
        raise FactorOutputError(
            "Cross-asset LLM factor index and columns must exactly match the inputs"
        )
    if value.index.has_duplicates or value.columns.has_duplicates:
        raise FactorOutputError(
            "Cross-asset LLM factor axes must not contain duplicates"
        )
    _validate_finite_numeric(value)
    return value


def _apply_resource_limits(limits: SandboxLimits) -> None:
    try:
        import resource

        cpu = max(1, int(limits.cpu_seconds))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
        memory = max(128, int(limits.memory_mb)) * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
        if hasattr(resource, "RLIMIT_NOFILE"):
            resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
    except (ImportError, OSError, ValueError):
        # Wall-clock timeout and process termination remain active on platforms
        # where one or more POSIX resource limits are unavailable.
        pass


def _sandbox_worker(connection, code_str: str, data: Any, limits: SandboxLimits) -> None:
    try:
        _apply_resource_limits(limits)
        import builtins
        import numpy as np

        safe_globals = {
            "__builtins__": {
                name: getattr(builtins, name)
                for name in _SAFE_BUILTINS
            },
            "np": np,
            "pd": pd,
        }
        local_vars = {"df": data}
        compiled = compile(code_str, "<llm-factor>", "exec")
        exec(compiled, safe_globals, local_vars)
        output = validate_llm_factor_output(local_vars.get("factor"), data)
        connection.send({"status": "success", "output": output})
    except Exception as exc:
        connection.send(
            {
                "status": "error",
                "error_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        connection.close()


class RestrictedSandbox:
    """Validate statically, then execute in a bounded short-lived process."""

    def __init__(
        self,
        timeout_seconds: float = 5.0,
        cpu_seconds: int = 3,
        memory_mb: int = 1024,
    ) -> None:
        self.limits = SandboxLimits(
            timeout_seconds=float(timeout_seconds),
            cpu_seconds=int(cpu_seconds),
            memory_mb=int(memory_mb),
        )

    def validate_code(self, code_str: str) -> None:
        FactorCodePolicy().validate(code_str)

    def execute_factor_code(
        self,
        code_str: str,
        data: Any,
    ) -> Union[pd.Series, pd.DataFrame]:
        self.validate_code(code_str)
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe(duplex=False)
        process = context.Process(
            target=_sandbox_worker,
            args=(child, code_str, data, self.limits),
            daemon=True,
        )
        process.start()
        child.close()
        try:
            if not parent.poll(self.limits.timeout_seconds):
                process.terminate()
                process.join(timeout=1)
                if process.is_alive() and hasattr(process, "kill"):
                    process.kill()
                    process.join(timeout=1)
                raise SandboxTimeoutError(
                    f"LLM factor execution exceeded {self.limits.timeout_seconds:.2f}s"
                )
            result = parent.recv()
        except EOFError as exc:
            raise SandboxExecutionError(
                f"LLM factor worker exited without a result (exit code {process.exitcode})"
            ) from exc
        finally:
            parent.close()
            process.join(timeout=1)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1)

        if result["status"] == "success":
            return result["output"]
        error_type = result.get("error_type", "SandboxExecutionError")
        message = result.get("message", "LLM factor execution failed")
        if error_type == "FactorOutputError":
            raise FactorOutputError(message)
        raise SandboxExecutionError(f"{error_type}: {message}")
