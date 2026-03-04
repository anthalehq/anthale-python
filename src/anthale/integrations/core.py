"""
Core Anthale policy enforcer classes.
"""

from __future__ import annotations

from typing import Any, Mapping, Iterable

from anthale import Anthale, AsyncAnthale
from anthale._exceptions import AnthaleError
from anthale.types.organizations.policy_enforce_params import Message
from anthale.types.organizations.policy_enforce_response import PolicyEnforceResponse

__all__ = (
    "AsyncPolicyEnforcer",
    "AnthalePolicyViolationError",
    "SyncPolicyEnforcer",
)


class AnthalePolicyViolationError(AnthaleError):
    """
    Raised when Anthale policy enforcement returns a blocked action.

    Example:
    ```python
    from anthale.integrations.core import AnthalePolicyViolationError

    try:
        ...
    except AnthalePolicyViolationError as error:
        print(error.enforcement_identifier)
    ```
    """

    enforcement_identifier: str

    def __init__(self, *, enforcement_identifier: str) -> None:
        """
        Initialize an AnthalePolicyViolationError.

        Args:
            enforcement_identifier (str): Identifier of the policy enforcement.
        """
        self.enforcement_identifier = enforcement_identifier

        message = f"Policy enforcement '{enforcement_identifier}' was blocked due to a policy violation."
        super().__init__(message)


class SyncPolicyEnforcer:
    """
    Synchronous Anthale policy enforcer.
    """

    _client: Any
    _policy_identifier: str
    _metadata: dict[str, Any]

    def __init__(self, *, client: Any, policy_identifier: str, metadata: Mapping[str, Any] | None = None) -> None:
        """
        Initialize a sync policy enforcer.

        Args:
            client (Any): Anthale sync client instance.
            policy_identifier (str): Identifier of the policy to enforce.
            metadata (Mapping[str, Any]): Optional metadata to include in the enforcement request.
        """
        self._client = client
        self._policy_identifier = policy_identifier
        self._metadata = {} if metadata is None else dict(metadata)

    def enforce(
        self,
        *,
        direction: str,
        messages: Iterable[Message],
        metadata: Mapping[str, object] | None = None,
    ) -> PolicyEnforceResponse:
        """
        Enforce Anthale policy for a message list.

        Args:
            direction (str): Either `input` or `output`.
            messages (Iterable[Message]): Messages to evaluate.
            metadata (Mapping[str, object] | None): Optional metadata to include in the enforcement request.

        Returns:
            PolicyEnforceResponse: Enforcement result.

        Raises:
            AnthalePolicyViolationError: If the policy enforcement response action is `block`.
        """
        response = self._client.organizations.policies.enforce(
            self._policy_identifier,
            direction=direction,
            messages=messages,
            include_evaluations=False,
            metadata={**self._metadata, **(metadata or {})},
        )

        if response.action == "block":
            raise AnthalePolicyViolationError(enforcement_identifier=response.enforcer_identifier)

        return response  # type: ignore[no-any-return]


class AsyncPolicyEnforcer:
    """
    Asynchronous Anthale policy enforcer.
    """

    _client: Any
    _policy_identifier: str
    _metadata: dict[str, Any]

    def __init__(self, *, client: Any, policy_identifier: str, metadata: Mapping[str, Any] | None = None) -> None:
        """
        Initialize an async policy enforcer.

        Args:
            client (Any): Anthale async client instance.
            policy_identifier (str): Identifier of the policy to enforce.
            metadata (Mapping[str, Any] | None, optional): Optional metadata to include in the enforcement request.
        """
        self._client = client
        self._policy_identifier = policy_identifier
        self._metadata = {} if metadata is None else dict(metadata)

    async def enforce(
        self,
        *,
        direction: str,
        messages: Iterable[Message],
        metadata: Mapping[str, object] | None = None,
    ) -> PolicyEnforceResponse:
        """
        Enforce Anthale policy for a message list.

        Args:
            direction (str): Either `input` or `output`.
            messages (Iterable[Message]): Messages to evaluate.
            metadata (Mapping[str, object] | None): Optional metadata to include in the enforcement request.

        Returns:
            PolicyEnforceResponse: Enforcement result.

        Raises:
            AnthalePolicyViolationError: If the policy enforcement response action is `block`.
        """
        response = await self._client.organizations.policies.enforce(
            self._policy_identifier,
            direction=direction,
            messages=messages,
            include_evaluations=False,
            metadata={**self._metadata, **(metadata or {})},
        )

        if response.action == "block":
            raise AnthalePolicyViolationError(enforcement_identifier=response.enforcer_identifier)

        return response  # type: ignore[no-any-return]


def build_enforcers(
    *,
    policy_id: str,
    api_key: str | None,
    client: Any | None,
    async_client: Any | None,
    metadata: Mapping[str, Any] | None,
) -> tuple[SyncPolicyEnforcer | None, AsyncPolicyEnforcer | None]:
    """
    Build sync and async policy enforcers based on provided parameters.

    Args:
        policy_id (str): Identifier of the policy to enforce.
        api_key (str | None, optional): Anthale API key, used when clients are not provided.
        client (Any | None, optional): Optional sync Anthale client instance.
        async_client (Any | None, optional): Optional async Anthale client instance.
        metadata (Mapping[str, Any] | None, optional): Metadata to include in enforcement requests.

    Returns:
        tuple[SyncPolicyEnforcer | None, AsyncPolicyEnforcer | None]: Tuple containing the sync and async policy
        enforcers. One or both may be None if corresponding clients were not provided.
    """
    if client is None and async_client is None:
        client = Anthale(api_key=api_key)
        async_client = AsyncAnthale(api_key=api_key)

    sync_enforcer = None
    if client is not None:
        sync_enforcer = SyncPolicyEnforcer(client=client, policy_identifier=policy_id, metadata=metadata)

    async_enforcer = None
    if async_client is not None:
        async_enforcer = AsyncPolicyEnforcer(client=async_client, policy_identifier=policy_id, metadata=metadata)

    return sync_enforcer, async_enforcer
