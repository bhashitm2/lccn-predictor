"""Reusable FastAPI dependencies: API-key auth and a simple per-IP rate limiter."""
from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Deque, Dict

from fastapi import Header, HTTPException, Request, status
from loguru import logger

from predictor.config import get_settings


async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Guard crawl-triggering endpoints.

    If ``LCCN_API_KEY`` is configured, a matching ``X-API-Key`` header is required.
    If it is NOT configured we allow the call (local-dev convenience) but warn —
    production deployments must set the key.
    """
    settings = get_settings()
    if not settings.api_key:
        logger.warning(
            "admin/crawl endpoint hit with no LCCN_API_KEY set — open in dev mode"
        )
        return
    if x_api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid or missing API key"
        )


def _client_ip(request: Request) -> str:
    # Behind a proxy (Render/Railway/Caddy) the real IP is in X-Forwarded-For.
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# ip -> timestamps of recent requests (sliding 60s window)
_hits: Dict[str, Deque[float]] = defaultdict(deque)


async def rate_limit(request: Request) -> None:
    """Lightweight in-process sliding-window limiter for public reads.

    Single-instance only (free tier); for multiple replicas use a shared store.
    """
    settings = get_settings()
    limit = settings.rate_limit_per_minute
    if limit <= 0:
        return
    now = time.monotonic()
    window_start = now - 60.0
    ip = _client_ip(request)
    bucket = _hits[ip]
    while bucket and bucket[0] < window_start:
        bucket.popleft()
    if len(bucket) >= limit:
        retry = int(60 - (now - bucket[0])) + 1
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="rate limit exceeded",
            headers={"Retry-After": str(max(1, retry))},
        )
    bucket.append(now)
