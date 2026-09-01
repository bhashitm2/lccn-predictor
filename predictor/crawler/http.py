"""Shared async HTTP layer for the crawler.

A bounded-concurrency, *rate-limited* request runner with retry, jittered
backoff and a shared circuit breaker. Callers get a :class:`FetchResult` per
request so they can tell "the server said no such thing" apart from "we were
blocked" — a distinction the prediction pipeline depends on.

**Why curl_cffi and not httpx/requests:** LeetCode's contest ranking REST API
(``/contest/api/ranking/...``) is behind Cloudflare, which fingerprints the TLS
handshake and returns 403 to plain Python TLS stacks. ``curl_cffi`` impersonates
a real Chrome handshake, so the requests pass. (GraphQL works either way, but we
use one client for both.)

**Why the pacing matters:** a full contest is ~1,600 ranking pages. Issuing them
as 1,600 tasks bounded only by a concurrency semaphore means that as soon as
Cloudflare starts refusing (a 403 comes back from the edge in ~20ms) the
effective request rate explodes into the hundreds per second, which earns a
harder block and turns a slow crawl into no crawl at all. So requests are paced
by a batch-wide token bucket, retries are jittered, and a run of consecutive
blocks trips a breaker that pauses everything rather than hammering on.
"""
from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, Hashable, Optional

from curl_cffi.requests import AsyncSession, Response
from loguru import logger

from predictor.config import get_settings

# Status codes that mean "Cloudflare/LeetCode is refusing us", as opposed to a
# genuine 404 or a transient 5xx.
BLOCKED_STATUSES = frozenset({403, 429})
# Retrying these is pointless — the resource really isn't there.
FATAL_STATUSES = frozenset({400, 404, 410})


class CrawlBlockedError(RuntimeError):
    """Raised when the remote is refusing us outright (Cloudflare 403/429).

    Distinct from a missing contest: the slug may be perfectly valid and the
    data may be there — we just can't reach it from this IP right now.
    """


@dataclass
class FetchResult:
    """Outcome of one request. Truthy only when a 200 response came back."""

    response: Optional[Response] = None
    status_code: Optional[int] = None
    attempts: int = 0
    blocked: bool = False
    error: Optional[str] = None

    def __bool__(self) -> bool:
        return self.response is not None

    def json(self) -> Any:
        if self.response is None:
            raise ValueError("no response to decode")
        return self.response.json()


class RateLimiter:
    """Token bucket shared by every task in a batch.

    The concurrency semaphore caps requests *in flight*; this caps requests
    *per second*, which is the thing Cloudflare actually counts.
    """

    def __init__(self, rate_per_second: float, burst: Optional[float] = None) -> None:
        self.rate = max(rate_per_second, 0.01)
        self.capacity = burst if burst is not None else max(self.rate, 1.0)
        self._tokens = self.capacity
        self._updated = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        while True:
            async with self._lock:
                now = time.monotonic()
                self._tokens = min(
                    self.capacity, self._tokens + (now - self._updated) * self.rate
                )
                self._updated = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                deficit = (1.0 - self._tokens) / self.rate
            await asyncio.sleep(deficit)


class CircuitBreaker:
    """Pauses the whole batch when the remote starts refusing us.

    Without this, a block turns every remaining request into an instant 403 and
    the retry loops issue tens of thousands of them in a couple of minutes —
    which is both useless and the reason the block gets worse. Instead: after
    ``threshold`` consecutive blocked responses with no success in between, all
    tasks wait out a cooldown. If ``max_cooldowns`` cooldowns pass without a
    single success, the batch is abandoned.
    """

    def __init__(self, threshold: int, cooldown: float, max_cooldowns: int) -> None:
        self.threshold = threshold
        self.cooldown = cooldown
        self.max_cooldowns = max_cooldowns
        self._consecutive_blocks = 0
        self._cooldowns_without_success = 0
        self._lock = asyncio.Lock()
        # Open (set) = traffic may flow. Cleared = everyone waits.
        self._gate = asyncio.Event()
        self._gate.set()
        self.tripped = False

    async def wait(self) -> None:
        """Block until the breaker lets traffic through."""
        await self._gate.wait()

    def record_success(self) -> None:
        self._consecutive_blocks = 0
        self._cooldowns_without_success = 0

    async def record_block(self) -> None:
        async with self._lock:
            self._consecutive_blocks += 1
            if self._consecutive_blocks < self.threshold or not self._gate.is_set():
                return
            # Trip: hold every task back while we wait the remote out.
            self._cooldowns_without_success += 1
            if self._cooldowns_without_success > self.max_cooldowns:
                self.tripped = True
                self._gate.set()  # release everyone so they can bail out
                return
            self._gate.clear()
            logger.warning(
                f"blocked {self._consecutive_blocks}x in a row — pausing all "
                f"requests for {self.cooldown:.0f}s "
                f"(cooldown {self._cooldowns_without_success}/{self.max_cooldowns})"
            )

        try:
            await asyncio.sleep(self.cooldown)
        finally:
            async with self._lock:
                self._consecutive_blocks = 0
                self._gate.set()


# --------------------------------------------------------------------------- #
# Session reuse
#
# Cloudflare hands out clearance cookies. Opening a fresh AsyncSession per
# fetch_all() call threw them away between the meta request and the ranking
# crawl, so every batch started cold. One session per process keeps them.
# --------------------------------------------------------------------------- #
_session: Optional[AsyncSession] = None


async def get_session() -> AsyncSession:
    """Return the process-wide session, creating it on first use.

    No lock: there is no await between the check and the assignment, so this is
    already atomic on an event loop. A module-level ``asyncio.Lock`` would also
    bind itself to whichever loop touched it first and then raise if the process
    ever ran a second loop.
    """
    global _session
    if _session is None:
        _session = AsyncSession(impersonate=get_settings().impersonate)
    return _session


async def close_http_session() -> None:
    """Close the shared session. Call once at process shutdown."""
    global _session
    session, _session = _session, None
    if session is not None:
        try:
            await session.close()
        except Exception as exc:  # noqa: BLE001 - shutdown must not raise
            logger.debug(f"error closing http session: {exc!r}")


def _retry_after_seconds(resp: Response) -> Optional[float]:
    """Honour a ``Retry-After`` header when the server sends one."""
    raw = resp.headers.get("Retry-After") if resp.headers else None
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return None  # HTTP-date form; the normal backoff covers it


def _backoff_seconds(attempt: int, base: float, cap: float) -> float:
    """Exponential backoff with full jitter.

    Jitter matters more than the curve here: ~1,600 tasks that back off by the
    same amount just re-form into a synchronised wave and hit the edge together.
    """
    ceiling = min(cap, base * (2 ** attempt))
    return random.uniform(ceiling / 2, ceiling)


async def _one_request(
    session: AsyncSession,
    semaphore: asyncio.Semaphore,
    limiter: RateLimiter,
    breaker: CircuitBreaker,
    request: Dict[str, Any],
    retry: int,
    timeout: float,
    backoff_base: float,
    backoff_cap: float,
) -> FetchResult:
    """Issue a single request with retry + jittered backoff."""
    method = request.get("method", "GET")
    url = request["url"]
    json_body = request.get("json")
    params = request.get("params")
    headers = request.get("headers")

    result = FetchResult()
    for attempt in range(retry + 1):
        if breaker.tripped:
            result.blocked = True
            result.error = "circuit breaker open"
            return result

        await breaker.wait()
        await limiter.acquire()
        result.attempts = attempt + 1
        retry_after: Optional[float] = None
        # Per-attempt, not cumulative: a 403 followed by a 500 must not report
        # the 500 to the breaker as another block.
        attempt_blocked = False

        async with semaphore:
            try:
                resp = await session.request(
                    method,
                    url,
                    json=json_body,
                    params=params,
                    headers=headers,
                    timeout=timeout,
                )
                result.status_code = resp.status_code
                if resp.status_code == 200:
                    breaker.record_success()
                    result.response = resp
                    result.blocked = False
                    return result
                if resp.status_code in FATAL_STATUSES:
                    # Not a transient condition — retrying just wastes budget.
                    logger.warning(f"{resp.status_code} for {url} — not retrying")
                    result.error = f"HTTP {resp.status_code}"
                    result.blocked = False
                    return result
                if resp.status_code in BLOCKED_STATUSES:
                    attempt_blocked = True
                    retry_after = _retry_after_seconds(resp)
                logger.warning(
                    f"non-200 {resp.status_code} for {url} "
                    f"(attempt {attempt + 1}/{retry + 1})"
                )
                result.error = f"HTTP {resp.status_code}"
            except Exception as exc:  # network error, timeout, etc.
                logger.warning(
                    f"request error {exc!r} for {url} "
                    f"(attempt {attempt + 1}/{retry + 1})"
                )
                result.error = repr(exc)

        result.blocked = attempt_blocked
        if attempt_blocked:
            await breaker.record_block()
        # backoff outside the semaphore so we free a slot while waiting
        if attempt < retry:
            await asyncio.sleep(
                retry_after
                if retry_after is not None
                else _backoff_seconds(attempt, backoff_base, backoff_cap)
            )

    logger.error(f"giving up after {result.attempts} attempts: {url}")
    return result


async def fetch_all(
    requests: Dict[Hashable, Dict[str, Any]],
    *,
    concurrency: Optional[int] = None,
    retry: Optional[int] = None,
    timeout: Optional[float] = None,
    rate_limit: Optional[float] = None,
) -> Dict[Hashable, FetchResult]:
    """Run a mapping of ``key -> request-spec`` concurrently.

    ``request-spec`` keys: ``url`` (required), ``method`` (default GET), optional
    ``json``, ``params`` and ``headers``. Returns a ``key -> FetchResult``
    mapping with the same keys; a result is truthy only if it holds a 200.

    Unset arguments fall back to ``Settings`` rather than to hardcoded values, so
    tuning via ``LCCN_*`` env vars applies to every caller.
    """
    settings = get_settings()
    concurrency = concurrency if concurrency is not None else settings.ranking_concurrency
    retry = retry if retry is not None else settings.http_retry
    timeout = timeout if timeout is not None else settings.http_timeout_seconds
    rate_limit = (
        rate_limit if rate_limit is not None else settings.http_rate_limit_per_second
    )

    semaphore = asyncio.Semaphore(concurrency)
    limiter = RateLimiter(rate_limit)
    breaker = CircuitBreaker(
        threshold=settings.http_block_threshold,
        cooldown=settings.http_block_cooldown_seconds,
        max_cooldowns=settings.http_max_cooldowns,
    )

    keys = list(requests.keys())
    session = await get_session()
    results = await asyncio.gather(
        *(
            _one_request(
                session,
                semaphore,
                limiter,
                breaker,
                requests[k],
                retry,
                timeout,
                settings.http_backoff_base_seconds,
                settings.http_backoff_cap_seconds,
            )
            for k in keys
        )
    )

    if breaker.tripped:
        logger.error(
            f"circuit breaker open after {settings.http_max_cooldowns} cooldowns "
            f"with no successful request"
        )
    return dict(zip(keys, results))
