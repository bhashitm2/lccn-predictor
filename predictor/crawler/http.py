"""Shared async HTTP layer for the crawler.

A bounded-concurrency request runner with automatic retry + backoff. Failed
requests resolve to ``None`` (callers must check), so a few unreachable users
never sink a whole contest prediction.

**Why curl_cffi and not httpx/requests:** LeetCode's contest ranking REST API
(``/contest/api/ranking/...``) is behind Cloudflare, which fingerprints the TLS
handshake and returns 403 to plain Python TLS stacks. ``curl_cffi`` impersonates
a real Chrome handshake, so the requests pass. (GraphQL works either way, but we
use one client for both.)
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, Hashable, Optional

from curl_cffi.requests import AsyncSession, Response
from loguru import logger

# Which browser TLS fingerprint to impersonate (curl_cffi target).
IMPERSONATE = "chrome"


async def _one_request(
    session: AsyncSession,
    semaphore: asyncio.Semaphore,
    request: Dict[str, Any],
    retry: int,
    timeout: float,
) -> Optional[Response]:
    """Issue a single request with retry + linear backoff. Returns None on give-up."""
    method = request.get("method", "GET")
    url = request["url"]
    json_body = request.get("json")
    params = request.get("params")

    for attempt in range(retry + 1):
        async with semaphore:
            try:
                resp = await session.request(
                    method, url, json=json_body, params=params, timeout=timeout
                )
                if resp.status_code == 200:
                    return resp
                logger.warning(
                    f"non-200 {resp.status_code} for {url} "
                    f"(attempt {attempt + 1}/{retry + 1})"
                )
            except Exception as exc:  # network error, timeout, etc.
                logger.warning(
                    f"request error {exc!r} for {url} "
                    f"(attempt {attempt + 1}/{retry + 1})"
                )
        # backoff outside the semaphore so we free a slot while waiting
        await asyncio.sleep(min(1.0 * (attempt + 1), 8.0))
    logger.error(f"giving up after {retry + 1} attempts: {url}")
    return None


async def fetch_all(
    requests: Dict[Hashable, Dict[str, Any]],
    *,
    concurrency: int = 5,
    retry: int = 10,
    timeout: float = 30.0,
) -> Dict[Hashable, Optional[Response]]:
    """Run a mapping of ``key -> request-spec`` concurrently.

    ``request-spec`` keys: ``url`` (required), ``method`` (default GET), optional
    ``json`` and ``params``. Returns a ``key -> Response | None`` mapping with the
    same keys.
    """
    semaphore = asyncio.Semaphore(concurrency)
    keys = list(requests.keys())
    async with AsyncSession(impersonate=IMPERSONATE) as session:
        tasks = [
            _one_request(session, semaphore, requests[k], retry, timeout)
            for k in keys
        ]
        responses = await asyncio.gather(*tasks)
    return dict(zip(keys, responses))
