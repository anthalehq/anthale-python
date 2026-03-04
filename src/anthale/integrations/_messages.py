"""
Module for handling message formatting and normalization for policy enforcement.
"""

from __future__ import annotations

from json import dumps
from typing import Any, Mapping, Iterable
from typing_extensions import Literal

__all__ = (
    "normalize_role",
    "stringify",
)

_ROLE_MAP: dict[str, Literal["system", "user", "assistant", "tool"]] = {
    "system": "system",
    "developer": "system",
    "machine": "system",
    "user": "user",
    "human": "user",
    "assistant": "assistant",
    "ai": "assistant",
    "tool": "tool",
    "function": "tool",
}


def normalize_role(*, value: str) -> Literal["system", "user", "assistant", "tool"]:
    """
    Normalize a role string to one of the allowed roles: system, user, assistant, tool. Maps common variants to these
    roles.

    Args:
        value (str): The role string to normalize.

    Returns:
        Literal["system", "user", "assistant", "tool"]: The normalized role.
    """
    return _ROLE_MAP.get(value.lower(), "user")


def stringify(*, value: Any) -> str:
    """
    Convert a value to a string for use in a message content. Handles common types like str, bytes, dict, list, etc.

    Args:
        value (Any): The value to convert to a string.

    Returns:
        str: The string representation of the value.
    """
    if isinstance(value, str):
        return value

    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")

    if isinstance(value, Mapping):
        try:
            return dumps(value, sort_keys=True, default=str)

        except Exception:
            return str(dict(value))  # type: ignore

    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
        try:
            return dumps(list(value), sort_keys=True, default=str)  # type: ignore

        except Exception:
            return str(list(value))  # type: ignore

    return str(value)
