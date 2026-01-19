# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import Dict, Union, Iterable
from typing_extensions import Literal, Required, TypedDict

__all__ = ["PolicyEnforceParams", "Message", "MessageContentBlock"]


class PolicyEnforceParams(TypedDict, total=False):
    direction: Required[Literal["input", "output"]]
    """Whether to evaluate input or output messages against the policy."""

    messages: Required[Iterable[Message]]
    """Ordered list of messages that compose the conversation to evaluate."""

    metadata: Dict[str, object]
    """Optional contextual metadata forwarded to guardrails."""


class MessageContentBlock(TypedDict, total=False):
    """Policy enforcer text message request schema."""

    text: Required[str]
    """Message text content."""

    type: Required[Literal["text", "document", "image", "audio", "video"]]
    """Content block type."""


class Message(TypedDict, total=False):
    """Policy enforcer message request schema."""

    content: Required[Union[str, Iterable[MessageContentBlock]]]
    """Raw text or a list of typed text blocks composing the message."""

    role: Required[Literal["system", "user", "assistant"]]
    """Message role within the conversation."""
