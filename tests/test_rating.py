"""Tests for the batched GraphQL rating resolver.

This module exists because `fetch_ratings` had *no* direct coverage: it is
monkeypatched in every `tests/test_api.py` case and `tests/test_http.py` never
reaches it. A one-word variable rename inside it therefore shipped to
production, where it killed a 15-minute crawl with

    TypeError: 'FetchResult' object does not support item assignment

after the ranking crawl had already succeeded. Everything here drives the real
function against a stub session.
"""
from __future__ import annotations

import json

import pytest

from predictor.crawler.http import CrawlBlockedError
from predictor.crawler.user_rating import (
    DEFAULT_ATTENDED_COUNT,
    DEFAULT_RATING,
    fetch_ratings,
)
from tests.conftest import StubResponse

pytestmark = pytest.mark.asyncio


def _users(n: int, region: str = "US"):
    return [(f"user{i}", region) for i in range(n)]


def _ok(alias_count: int, rating=1600.0, attended=12):
    """A GraphQL reply with every requested alias resolved."""
    return StubResponse(
        200,
        {
            "data": {
                f"u{i}": {"rating": rating + i, "attendedContestsCount": attended}
                for i in range(alias_count)
            }
        },
    )


async def test_returns_mapping_keyed_by_rating_key(stub_settings, stub_session):
    """Regression: the shadowing bug wrote into a FetchResult instead of this."""
    stub_settings(rating_batch_size=40, rating_concurrency=4)
    stub_session(lambda url, n: _ok(40))

    result = await fetch_ratings(_users(40))

    assert isinstance(result, dict)
    assert set(result) == set(_users(40))
    rating, attended = result[("user0", "US")]
    assert (rating, attended) == (1600.0, 12)
    assert all(
        isinstance(v, tuple) and len(v) == 2 for v in result.values()
    ), "each value must be (rating, attended_count)"


async def test_resolves_across_multiple_batches(stub_settings, stub_session):
    """Every user must appear exactly once even when split over batches."""
    stub_settings(rating_batch_size=10, rating_concurrency=4)
    session = stub_session(lambda url, n: _ok(10))

    result = await fetch_ratings(_users(35))

    assert len(result) == 35
    assert set(result) == set(_users(35))
    assert len(session.calls) == 4  # ceil(35/10)


async def test_newcomers_get_defaults(stub_settings, stub_session):
    """A null node means no contest history — the documented (1500, 0) case."""
    stub_settings(rating_batch_size=4, rating_concurrency=2)
    stub_session(
        lambda url, n: StubResponse(
            200,
            {
                "data": {
                    "u0": {"rating": 1700.0, "attendedContestsCount": 5},
                    "u1": None,
                    "u2": {"rating": 1800.0, "attendedContestsCount": 9},
                    "u3": None,
                }
            },
        )
    )

    result = await fetch_ratings(_users(4))

    assert result[("user0", "US")] == (1700.0, 5)
    assert result[("user1", "US")] == (DEFAULT_RATING, DEFAULT_ATTENDED_COUNT)
    assert result[("user2", "US")] == (1800.0, 9)
    assert result[("user3", "US")] == (DEFAULT_RATING, DEFAULT_ATTENDED_COUNT)


async def test_a_single_failed_batch_defaults_its_members(
    stub_settings, stub_session
):
    """One straggler batch is tolerable — its users fall back to defaults."""
    stub_settings(
        rating_batch_size=10,
        rating_concurrency=4,
        http_retry=0,
        http_block_threshold=99,
        max_failed_rating_batches=0.5,
    )
    # 10 batches; only the very first request fails.
    stub_session(lambda url, n: StubResponse(500) if n == 1 else _ok(10))

    result = await fetch_ratings(_users(100))

    assert len(result) == 100
    defaulted = [k for k, v in result.items() if v == (DEFAULT_RATING, 0)]
    assert len(defaulted) == 10, "exactly the failed batch's members default"


async def test_too_many_failed_batches_refuses(stub_settings, stub_session):
    """Defaulting a large slice of the field would fabricate the Elo inputs."""
    stub_settings(
        rating_batch_size=10,
        rating_concurrency=4,
        http_retry=0,
        http_block_threshold=99,
        max_failed_rating_batches=0.02,
    )
    stub_session(lambda url, n: StubResponse(500))

    with pytest.raises(CrawlBlockedError) as exc:
        await fetch_ratings(_users(100))

    assert "rating batches failed" in str(exc.value)


async def test_cn_users_use_the_cn_endpoint(stub_settings, stub_session):
    """Region picks both the endpoint and the GraphQL argument name."""
    stub_settings(rating_batch_size=10, rating_concurrency=2)
    session = stub_session(lambda url, n: _ok(2))

    await fetch_ratings([("cn-user", "CN"), ("us-user", "US")])

    urls = " ".join(session.calls)
    assert "leetcode.cn" in urls and "leetcode.com" in urls
    queries = [json.dumps(p) for p in session.payloads if p]
    assert any("userSlug" in q for q in queries), "CN batch queries by userSlug"
    assert any("username" in q for q in queries), "US batch queries by username"


async def test_empty_input_is_a_no_op(stub_settings, stub_session):
    """No users means no requests — and no divide-by-zero in the guard."""
    stub_settings()
    session = stub_session(lambda url, n: _ok(0))

    result = await fetch_ratings([])

    assert result == {}
    assert session.calls == []
