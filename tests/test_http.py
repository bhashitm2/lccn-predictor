"""Crawler resilience tests: pacing, the circuit breaker, and crawl completeness.

These cover the failure mode that produced a *green* workflow run holding 350 of
~39,400 ranking rows: Cloudflare 403'd almost every page, the failures were
skipped silently, and the truncated field was persisted over the good one.

Everything here runs against a stub session — no network.
"""
from __future__ import annotations

import asyncio
import time

import pytest

from tests.conftest import StubResponse
from predictor.crawler.http import (
    CircuitBreaker,
    CrawlBlockedError,
    RateLimiter,
    fetch_all,
)
from predictor.crawler.ranking import IncompleteCrawlError, fetch_ranking

pytestmark = pytest.mark.asyncio


# --------------------------------------------------------------------------- #
# Rate limiter
# --------------------------------------------------------------------------- #


async def test_rate_limiter_bounds_aggregate_throughput():
    """Concurrency caps in-flight requests; only the limiter caps the rate."""
    limiter = RateLimiter(rate_per_second=20.0, burst=1)
    start = time.monotonic()
    await asyncio.gather(*(limiter.acquire() for _ in range(10)))
    elapsed = time.monotonic() - start
    # 10 tokens at 20/s from a burst of 1 => >= ~0.45s. Generous lower bound.
    assert elapsed >= 0.3


async def test_rate_limiter_applies_across_concurrent_tasks(
    stub_settings, stub_session
):
    rate = 40.0
    stub_settings(http_rate_limit_per_second=rate, ranking_concurrency=8)
    session = stub_session(lambda url, n: StubResponse(200, {"ok": True}))

    # The bucket allows one second of traffic as an opening burst, so ask for
    # appreciably more than that to observe the sustained rate.
    count = 60
    start = time.monotonic()
    results = await fetch_all({i: {"url": f"https://x/{i}"} for i in range(count)})
    elapsed = time.monotonic() - start

    assert all(results.values())
    # After the burst, the remainder is paced at `rate`/s no matter how wide the
    # semaphore is — the property that stops a 403 storm turning into 200 req/s.
    assert elapsed >= (count - rate) / rate * 0.7
    assert len(session.calls) == count


# --------------------------------------------------------------------------- #
# Circuit breaker
# --------------------------------------------------------------------------- #


async def test_breaker_aborts_instead_of_exhausting_every_retry(
    stub_settings, stub_session
):
    """The bug this replaces: ~1,600 pages x 16 attempts = ~25,000 requests."""
    stub_settings(
        http_retry=15, http_block_threshold=4, http_max_cooldowns=1
    )
    session = stub_session(lambda url, n: StubResponse(403))

    results = await fetch_all({i: {"url": f"https://x/{i}"} for i in range(200)})

    assert not any(results.values())
    assert any(r.blocked for r in results.values())
    # Without the breaker this would be 200 * 16 = 3,200 requests.
    assert len(session.calls) < 3200 / 2


async def test_breaker_recovers_when_requests_start_succeeding(
    stub_settings, stub_session
):
    stub_settings(http_retry=6, http_block_threshold=3, http_max_cooldowns=5)
    # Refuse the first handful, then let everything through.
    session = stub_session(
        lambda url, n: StubResponse(403) if n <= 6 else StubResponse(200, {"ok": 1})
    )

    results = await fetch_all({i: {"url": f"https://x/{i}"} for i in range(10)})

    assert all(results.values()), "every request should eventually succeed"


async def test_404_is_not_retried(stub_settings, stub_session):
    """A missing resource is not a transient condition."""
    stub_settings(http_retry=10)
    session = stub_session(lambda url, n: StubResponse(404))

    results = await fetch_all({"only": {"url": "https://x/missing"}})

    assert not results["only"]
    assert results["only"].status_code == 404
    assert not results["only"].blocked
    assert len(session.calls) == 1


async def test_retry_after_header_is_honoured(stub_settings, stub_session):
    stub_settings(http_retry=2, http_block_threshold=99)
    stub_session(
        lambda url, n: StubResponse(429, headers={"Retry-After": "0.05"})
        if n == 1
        else StubResponse(200, {"ok": 1})
    )

    start = time.monotonic()
    results = await fetch_all({"a": {"url": "https://x/a"}})
    elapsed = time.monotonic() - start

    assert results["a"]
    assert elapsed >= 0.05


async def test_blocked_result_is_distinguishable_from_missing(
    stub_settings, stub_session
):
    """`fetch_user_num` depends on this to avoid 'check the slug' on a 403."""
    stub_settings(http_retry=1, http_block_threshold=99)
    stub_session(lambda url, n: StubResponse(403))

    results = await fetch_all({"a": {"url": "https://x/a"}})

    assert results["a"].blocked
    assert results["a"].status_code == 403


async def test_circuit_breaker_gate_releases_waiters():
    breaker = CircuitBreaker(threshold=1, cooldown=0.01, max_cooldowns=5)
    await breaker.record_block()
    await asyncio.wait_for(breaker.wait(), timeout=1.0)


# --------------------------------------------------------------------------- #
# Crawl completeness
# --------------------------------------------------------------------------- #


def _page_payload(page: int, per_page: int = 25):
    return {
        "total_rank": [
            {
                "username": f"u{(page - 1) * per_page + i}",
                "user_slug": f"u{(page - 1) * per_page + i}",
                "data_region": "US",
                "rank": (page - 1) * per_page + i + 1,
                "score": 10,
                "finish_time": 1700000000,
            }
            for i in range(per_page)
        ]
    }


def _page_of(url: str) -> int:
    return int(url.split("pagination=")[1].split("&")[0])


async def test_partial_crawl_raises_instead_of_returning_truncated_rows(
    stub_settings, stub_session
):
    """The exact shape of the corrupting run: a handful of pages, the rest 403."""
    stub_settings(http_retry=1, http_block_threshold=3, http_max_cooldowns=1)
    # Only the first 2 of 40 pages are served.
    stub_session(
        lambda url, n: StubResponse(200, _page_payload(_page_of(url)))
        if _page_of(url) <= 2
        else StubResponse(403)
    )

    with pytest.raises(CrawlBlockedError) as exc:
        await fetch_ranking("weekly-contest-517", user_num=1000)

    assert "1000" in str(exc.value)


async def test_incomplete_crawl_without_blocking_raises_incomplete(
    stub_settings, stub_session
):
    stub_settings(http_retry=1, repair_passes=0)
    # Pages beyond 2 return 500s — failures, but not refusals.
    stub_session(
        lambda url, n: StubResponse(200, _page_payload(_page_of(url)))
        if _page_of(url) <= 2
        else StubResponse(500)
    )

    with pytest.raises(IncompleteCrawlError):
        await fetch_ranking("weekly-contest-517", user_num=1000)


async def test_repair_pass_recovers_dropped_pages(stub_settings, stub_session):
    """A few pages fail on the first pass and are re-fetched, not skipped."""
    stub_settings(http_retry=0, repair_passes=2, http_block_threshold=99)
    failed_once: set[int] = set()

    def handler(url, n):
        page = _page_of(url)
        # Page 3 fails exactly once, then succeeds on the repair pass.
        if page == 3 and page not in failed_once:
            failed_once.add(page)
            return StubResponse(500)
        return StubResponse(200, _page_payload(page))

    stub_session(handler)

    rows = await fetch_ranking("weekly-contest-517", user_num=250)

    assert len(rows) == 250
    assert [r.rank for r in rows] == list(range(1, 251))


async def test_complete_crawl_returns_all_rows(stub_settings, stub_session):
    stub_settings(http_retry=1)
    stub_session(lambda url, n: StubResponse(200, _page_payload(_page_of(url))))

    rows = await fetch_ranking("weekly-contest-517", user_num=500)

    assert len(rows) == 500
    assert rows[0].rank == 1
    assert rows[-1].rank == 500


async def test_a_few_missing_pages_stay_within_tolerance(stub_settings, stub_session):
    """Coverage is measured in pages; a handful short of a big crawl is fine."""
    stub_settings(http_retry=0, repair_passes=0, min_ranking_coverage=0.98)
    # 1000 pages, 10 of them permanently unavailable => 99% coverage.
    dead = {i for i in range(100, 110)}
    stub_session(
        lambda url, n: StubResponse(500)
        if _page_of(url) in dead
        else StubResponse(200, _page_payload(_page_of(url)))
    )

    rows = await fetch_ranking("weekly-contest-517", user_num=25000)

    assert len(rows) == (1000 - len(dead)) * 25


async def test_limit_narrows_the_expected_field(stub_settings, stub_session):
    """A top-N request must not be judged against the full contest size."""
    stub_settings(http_retry=1)
    stub_session(
        lambda url, n: StubResponse(200, _page_payload(_page_of(url)))
        if _page_of(url) <= 4
        else StubResponse(403)
    )

    rows = await fetch_ranking("weekly-contest-517", user_num=10000, limit=100)

    assert len(rows) == 100
