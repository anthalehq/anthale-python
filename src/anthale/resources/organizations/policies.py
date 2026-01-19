# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import Dict, Iterable
from typing_extensions import Literal

import httpx

from ..._types import Body, Omit, Query, Headers, NotGiven, omit, not_given
from ..._utils import maybe_transform, async_maybe_transform
from ..._compat import cached_property
from ..._resource import SyncAPIResource, AsyncAPIResource
from ..._response import (
    to_raw_response_wrapper,
    to_streamed_response_wrapper,
    async_to_raw_response_wrapper,
    async_to_streamed_response_wrapper,
)
from ..._base_client import make_request_options
from ...types.organizations import policy_enforce_params
from ...types.organizations.policy_enforce_response import PolicyEnforceResponse

__all__ = ["PoliciesResource", "AsyncPoliciesResource"]


class PoliciesResource(SyncAPIResource):
    @cached_property
    def with_raw_response(self) -> PoliciesResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://www.github.com/anthalehq/anthale-python#accessing-raw-response-data-eg-headers
        """
        return PoliciesResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> PoliciesResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://www.github.com/anthalehq/anthale-python#with_streaming_response
        """
        return PoliciesResourceWithStreamingResponse(self)

    def enforce(
        self,
        policy_identifier: str,
        *,
        direction: Literal["input", "output"],
        messages: Iterable[policy_enforce_params.Message],
        metadata: Dict[str, object] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> PolicyEnforceResponse:
        """
        Evaluates a set of messages against the specified policy and returns guardrail
        decisions.

        Args:
          policy_identifier: The policy identifier to enforce.

          direction: Whether to evaluate input or output messages against the policy.

          messages: Ordered list of messages that compose the conversation to evaluate.

          metadata: Optional contextual metadata forwarded to guardrails.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not policy_identifier:
            raise ValueError(f"Expected a non-empty value for `policy_identifier` but received {policy_identifier!r}")
        return self._post(
            f"/organizations/policies/{policy_identifier}/enforce",
            body=maybe_transform(
                {
                    "direction": direction,
                    "messages": messages,
                    "metadata": metadata,
                },
                policy_enforce_params.PolicyEnforceParams,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=PolicyEnforceResponse,
        )


class AsyncPoliciesResource(AsyncAPIResource):
    @cached_property
    def with_raw_response(self) -> AsyncPoliciesResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://www.github.com/anthalehq/anthale-python#accessing-raw-response-data-eg-headers
        """
        return AsyncPoliciesResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> AsyncPoliciesResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://www.github.com/anthalehq/anthale-python#with_streaming_response
        """
        return AsyncPoliciesResourceWithStreamingResponse(self)

    async def enforce(
        self,
        policy_identifier: str,
        *,
        direction: Literal["input", "output"],
        messages: Iterable[policy_enforce_params.Message],
        metadata: Dict[str, object] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> PolicyEnforceResponse:
        """
        Evaluates a set of messages against the specified policy and returns guardrail
        decisions.

        Args:
          policy_identifier: The policy identifier to enforce.

          direction: Whether to evaluate input or output messages against the policy.

          messages: Ordered list of messages that compose the conversation to evaluate.

          metadata: Optional contextual metadata forwarded to guardrails.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not policy_identifier:
            raise ValueError(f"Expected a non-empty value for `policy_identifier` but received {policy_identifier!r}")
        return await self._post(
            f"/organizations/policies/{policy_identifier}/enforce",
            body=await async_maybe_transform(
                {
                    "direction": direction,
                    "messages": messages,
                    "metadata": metadata,
                },
                policy_enforce_params.PolicyEnforceParams,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=PolicyEnforceResponse,
        )


class PoliciesResourceWithRawResponse:
    def __init__(self, policies: PoliciesResource) -> None:
        self._policies = policies

        self.enforce = to_raw_response_wrapper(
            policies.enforce,
        )


class AsyncPoliciesResourceWithRawResponse:
    def __init__(self, policies: AsyncPoliciesResource) -> None:
        self._policies = policies

        self.enforce = async_to_raw_response_wrapper(
            policies.enforce,
        )


class PoliciesResourceWithStreamingResponse:
    def __init__(self, policies: PoliciesResource) -> None:
        self._policies = policies

        self.enforce = to_streamed_response_wrapper(
            policies.enforce,
        )


class AsyncPoliciesResourceWithStreamingResponse:
    def __init__(self, policies: AsyncPoliciesResource) -> None:
        self._policies = policies

        self.enforce = async_to_streamed_response_wrapper(
            policies.enforce,
        )
