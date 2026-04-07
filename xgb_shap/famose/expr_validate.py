"""Validate feature expressions: only Handler columns + funcs_methods names, safe AST."""

from __future__ import annotations

import ast
from typing import Iterable


def validate_feature_expr(expr: str, allowed_names: set[str]) -> str | None:
    expr = expr.strip()
    if not expr:
        return "empty expression"
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        return f"syntax error: {e}"

    errors: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id not in allowed_names:
                errors.append(f"unknown name: {node.id}")
        elif isinstance(node, ast.Attribute):
            errors.append("attribute access is not allowed")
        elif isinstance(node, ast.Subscript):
            errors.append("subscripting is not allowed")
        elif isinstance(
            node,
            (
                ast.Import,
                ast.ImportFrom,
                ast.Lambda,
                ast.ListComp,
                ast.DictComp,
                ast.SetComp,
                ast.GeneratorExp,
                ast.Await,
                ast.Yield,
                ast.YieldFrom,
            ),
        ):
            errors.append(f"disallowed syntax: {type(node).__name__}")
        elif isinstance(node, ast.Call):
            if any(kw.arg is None for kw in node.keywords):
                errors.append("star-args are not allowed in calls")
    return "; ".join(errors) if errors else None


def allowed_names_for_handler(handler_keys: Iterable[str], func_keys: Iterable[str]) -> set[str]:
    return set(handler_keys) | set(func_keys) | {"True", "False", "None"}
