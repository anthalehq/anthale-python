# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

import os
from typing import Any, cast

import pytest

from anthale import Anthale, AsyncAnthale
from tests.utils import assert_matches_type
from anthale.types.organizations import PolicyEnforceResponse

base_url = os.environ.get("TEST_API_BASE_URL", "http://127.0.0.1:4010")


class TestPolicies:
    parametrize = pytest.mark.parametrize("client", [False, True], indirect=True, ids=["loose", "strict"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    def test_method_enforce(self, client: Anthale) -> None:
        policy = client.organizations.policies.enforce(
            policy_identifier="a90e34d6-41af-432f-a6ae-046598df4539",
            direction="input",
            messages=[
                {
                    "content": "Can you summarize the plot of Interstellar?",
                    "role": "user",
                }
            ],
        )
        assert_matches_type(PolicyEnforceResponse, policy, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    def test_method_enforce_with_all_params(self, client: Anthale) -> None:
        policy = client.organizations.policies.enforce(
            policy_identifier="a90e34d6-41af-432f-a6ae-046598df4539",
            direction="input",
            messages=[
                {
                    "content": "Can you summarize the plot of Interstellar?",
                    "role": "user",
                }
            ],
            metadata={"foo": "bar"},
        )
        assert_matches_type(PolicyEnforceResponse, policy, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    def test_raw_response_enforce(self, client: Anthale) -> None:
        response = client.organizations.policies.with_raw_response.enforce(
            policy_identifier="a90e34d6-41af-432f-a6ae-046598df4539",
            direction="input",
            messages=[
                {
                    "content": "Can you summarize the plot of Interstellar?",
                    "role": "user",
                }
            ],
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        policy = response.parse()
        assert_matches_type(PolicyEnforceResponse, policy, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    def test_streaming_response_enforce(self, client: Anthale) -> None:
        with client.organizations.policies.with_streaming_response.enforce(
            policy_identifier="a90e34d6-41af-432f-a6ae-046598df4539",
            direction="input",
            messages=[
                {
                    "content": "Can you summarize the plot of Interstellar?",
                    "role": "user",
                }
            ],
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            policy = response.parse()
            assert_matches_type(PolicyEnforceResponse, policy, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    def test_path_params_enforce(self, client: Anthale) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `policy_identifier` but received ''"):
            client.organizations.policies.with_raw_response.enforce(
                policy_identifier="",
                direction="input",
                messages=[
                    {
                        "content": "Can you summarize the plot of Interstellar?",
                        "role": "user",
                    }
                ],
            )


class TestAsyncPolicies:
    parametrize = pytest.mark.parametrize(
        "async_client", [False, True, {"http_client": "aiohttp"}], indirect=True, ids=["loose", "strict", "aiohttp"]
    )

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    async def test_method_enforce(self, async_client: AsyncAnthale) -> None:
        policy = await async_client.organizations.policies.enforce(
            policy_identifier="a90e34d6-41af-432f-a6ae-046598df4539",
            direction="input",
            messages=[
                {
                    "content": "Can you summarize the plot of Interstellar?",
                    "role": "user",
                }
            ],
        )
        assert_matches_type(PolicyEnforceResponse, policy, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    async def test_method_enforce_with_all_params(self, async_client: AsyncAnthale) -> None:
        policy = await async_client.organizations.policies.enforce(
            policy_identifier="a90e34d6-41af-432f-a6ae-046598df4539",
            direction="input",
            messages=[
                {
                    "content": "Can you summarize the plot of Interstellar?",
                    "role": "user",
                }
            ],
            metadata={"foo": "bar"},
        )
        assert_matches_type(PolicyEnforceResponse, policy, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    async def test_raw_response_enforce(self, async_client: AsyncAnthale) -> None:
        response = await async_client.organizations.policies.with_raw_response.enforce(
            policy_identifier="a90e34d6-41af-432f-a6ae-046598df4539",
            direction="input",
            messages=[
                {
                    "content": "Can you summarize the plot of Interstellar?",
                    "role": "user",
                }
            ],
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        policy = await response.parse()
        assert_matches_type(PolicyEnforceResponse, policy, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    async def test_streaming_response_enforce(self, async_client: AsyncAnthale) -> None:
        async with async_client.organizations.policies.with_streaming_response.enforce(
            policy_identifier="a90e34d6-41af-432f-a6ae-046598df4539",
            direction="input",
            messages=[
                {
                    "content": "Can you summarize the plot of Interstellar?",
                    "role": "user",
                }
            ],
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            policy = await response.parse()
            assert_matches_type(PolicyEnforceResponse, policy, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    async def test_path_params_enforce(self, async_client: AsyncAnthale) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `policy_identifier` but received ''"):
            await async_client.organizations.policies.with_raw_response.enforce(
                policy_identifier="",
                direction="input",
                messages=[
                    {
                        "content": "Can you summarize the plot of Interstellar?",
                        "role": "user",
                    }
                ],
            )
