from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Dict, Mapping

import numpy as np


_ALLOWED_FUNCS = {
    "sqrt": "sqrt",
    "sin": "sin",
    "cos": "cos",
    "atan2": "atan2",
    "log": "log",
    "exp": "exp",
    "abs": "abs",
}

_ALLOWED_CONSTS = {"pi": np.pi}

_ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp, ast.UnaryOp,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow,
    ast.USub, ast.UAdd,
    ast.Compare,
    ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq,
    ast.BoolOp, ast.And, ast.Or,
    ast.Name, ast.Load,
    ast.Call,
    ast.Constant,
)


def normalize_name(name: str) -> str:
    return name.replace(".", "_")


def _assert_allowed(node: ast.AST) -> None:
    for n in ast.walk(node):
        if not isinstance(n, _ALLOWED_NODES):
            raise ValueError(f"Disallowed AST node: {type(n).__name__}")


@dataclass(frozen=True, slots=True)
class CompiledExpr:
    expr: str
    code: Any
    varmap: Dict[str, str]

    def __call__(self, values: Mapping[str, Any]) -> Any:
        any_tensor = any(hasattr(v, "type") for v in values.values())

        env: Dict[str, Any] = {"pi": np.pi}

        if any_tensor:
            import pytensor.tensor as pt
            env.update({
                "sqrt": pt.sqrt, "sin": pt.sin, "cos": pt.cos,
                "atan2": pt.arctan2, "log": pt.log, "exp": pt.exp, "abs": pt.abs,
            })
        else:
            env.update({
                "sqrt": np.sqrt, "sin": np.sin, "cos": np.cos,
                "atan2": np.arctan2, "log": np.log, "exp": np.exp, "abs": np.abs,
            })

        for norm, orig in self.varmap.items():
            env[norm] = values[orig]

        return eval(self.code, {"__builtins__": {}}, env)


def compile_expr(expr: str, available_names: Mapping[str, Any]) -> CompiledExpr:
    varmap: Dict[str, str] = {normalize_name(k): k for k in available_names.keys()}

    rewritten = expr
    for orig in sorted(available_names.keys(), key=len, reverse=True):
        rewritten = rewritten.replace(orig, normalize_name(orig))

    node = ast.parse(rewritten, mode="eval")
    _assert_allowed(node)

    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            if n.id in varmap:
                continue
            if n.id in _ALLOWED_FUNCS:
                continue
            if n.id in _ALLOWED_CONSTS:
                continue
            raise ValueError(f"Unknown identifier in expression: {n.id}")

    code = compile(node, "<constraint>", "eval")
    return CompiledExpr(expr=expr, code=code, varmap=varmap)
