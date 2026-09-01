"""Fetch a contest's full ranking from LeetCode's public API.

Endpoints (no auth required):
  * ``GET {base}/contest/api/ranking/{slug}/``                 -> meta (``user_num``)
  * ``GET {base}/contest/api/ranking/{slug}/?pagination=N&region=global``
        -> page N of 25 ranking rows (``total_rank``)
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from math import ceil
from typing import Callable, Dict, List, Optional

from loguru import logger

from predictor.config import get_settings
from predictor.crawler.http import CrawlBlockedError, FetchResult, fetch_all

PAGE_SIZE = 25


class IncompleteCrawlError(RuntimeError):
    """Raised when too few ranking pages came back to trust the result.

    Predicting from a truncated field is worse than not predicting: the Elo/FFT
    engine models the whole contest, and persisting the result deletes the
    previous (complete) prediction for that slug.
    """


@dataclass
class RankingRow:
    username: str
    user_slug: str
    data_region: str  # "US" or "CN"
    rank: int
    score: int
    finish_time: Optional[datetime]


def _base_url() -> str:
    return get_settings().leetcode_base_us


def _page_headers(slug: str) -> dict:
    """Browser-ish headers for the ranking API.

    Cloudflare scores more than the TLS handshake; an XHR-looking request to a
    contest API with no ``Referer`` from the contest page is a cheap tell.
    """
    return {
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": f"{_base_url()}/contest/{slug}/ranking/",
        "X-Requested-With": "XMLHttpRequest",
    }


async def fetch_user_num(slug: str) -> Optional[int]:
    """Return the number of participants.

    Returns ``None`` when the contest genuinely has no participants or doesn't
    exist. Raises :class:`CrawlBlockedError` when we were refused — the caller
    must not confuse "no such contest" with "we can't see it from here".
    """
    settings = get_settings()
    url = f"{_base_url()}/contest/api/ranking/{slug}/"
    result = (
        await fetch_all(
            {
                "meta": {
                    "url": url,
                    "method": "GET",
                    "headers": _page_headers(slug),
                }
            },
            concurrency=1,
            retry=settings.http_retry,
            timeout=settings.http_timeout_seconds,
        )
    )["meta"]
    if not result:
        if result.blocked:
            raise CrawlBlockedError(
                f"blocked by LeetCode/Cloudflare while reading '{slug}' "
                f"(HTTP {result.status_code} after {result.attempts} attempts) — "
                "the contest may well exist; this IP is being refused"
            )
        raise_for = result.error or "no response"
        logger.warning(f"could not read contest meta for {slug}: {raise_for}")
        return None
    return result.json().get("user_num")


def _parse_row(raw: dict) -> Optional[RankingRow]:
    username = raw.get("username") or raw.get("user_slug")
    user_slug = raw.get("user_slug") or raw.get("username")
    if not user_slug:
        return None
    ft = raw.get("finish_time")
    finish_time = (
        datetime.fromtimestamp(ft, tz=timezone.utc) if isinstance(ft, (int, float)) else None
    )
    return RankingRow(
        username=username,
        user_slug=user_slug,
        data_region=(raw.get("data_region") or "US").upper(),
        rank=int(raw.get("rank", 0)),
        score=int(raw.get("score", 0)),
        finish_time=finish_time,
    )


async def fetch_ranking(
    slug: str,
    user_num: int,
    *,
    limit: Optional[int] = None,
    progress_cb: Optional[Callable[[int], None]] = None,
) -> List[RankingRow]:
    """Fetch ranking rows, sorted by rank ascending.

    :param limit: if set, only fetch the top-N participants (fewer pages) — useful
        for a fast first prediction; the rest can be backfilled later.
    :param progress_cb: optional callback invoked with the running row count.
    """
    settings = get_settings()
    effective = min(limit, user_num) if limit else user_num
    page_max = ceil(effective / PAGE_SIZE)
    base = _base_url()
    headers = _page_headers(slug)

    def _spec(page: int) -> dict:
        return {
            "url": f"{base}/contest/api/ranking/{slug}/?pagination={page}&region=global",
            "method": "GET",
            "headers": headers,
        }

    logger.info(f"fetching {page_max} ranking pages for {slug} (effective={effective})")

    by_page: Dict[int, List[RankingRow]] = {}
    pending = list(range(1, page_max + 1))
    blocked = False

    # First pass, then progressively gentler repair passes over whatever failed.
    # A handful of dropped pages used to sail through as a "successful" crawl;
    # now we go back for them before deciding the crawl is incomplete.
    for attempt in range(settings.repair_passes + 1):
        if not pending:
            break
        if attempt > 0:
            logger.warning(
                f"repair pass {attempt}/{settings.repair_passes}: "
                f"re-fetching {len(pending)} failed page(s) for {slug}"
            )
            await asyncio.sleep(settings.http_block_cooldown_seconds)

        # Halve rate and concurrency on each repair pass — if the first pass was
        # refused, going back at the same speed just gets refused again.
        divisor = 2 ** attempt
        results = await fetch_all(
            {page: _spec(page) for page in pending},
            concurrency=max(1, settings.ranking_concurrency // divisor),
            retry=settings.http_retry,
            timeout=settings.http_timeout_seconds,
            rate_limit=max(0.5, settings.http_rate_limit_per_second / divisor),
        )

        failed: List[int] = []
        fetched_rows = sum(len(v) for v in by_page.values())
        for page in sorted(results):
            result: FetchResult = results[page]
            if not result:
                blocked = blocked or result.blocked
                failed.append(page)
                continue
            page_rows: List[RankingRow] = []
            for raw in result.json().get("total_rank", []):
                row = _parse_row(raw)
                if row is not None:
                    page_rows.append(row)
            by_page[page] = page_rows
            fetched_rows += len(page_rows)
            if progress_cb:
                progress_cb(fetched_rows)
        pending = failed

    rows: List[RankingRow] = [row for page in sorted(by_page) for row in by_page[page]]
    rows.sort(key=lambda r: r.rank)
    if limit:
        rows = rows[:limit]

    # A partial crawl must never be mistaken for a complete one.
    #
    # Coverage is measured in PAGES, not rows: pages are exactly what we asked
    # for, whereas the row count also depends on how LeetCode counts `user_num`
    # (which can drift from the number of ranked rows it actually serves), and a
    # row-based threshold would fail every crawl if that drift exceeded it.
    fetched_pages = len(by_page)
    minimum_pages = ceil(page_max * settings.min_ranking_coverage)
    if not rows or fetched_pages < minimum_pages:
        detail = (
            f"incomplete crawl for '{slug}': got {fetched_pages}/{page_max} "
            f"ranking pages ({len(rows)} rows, expected ~{effective}) — "
            f"need at least {minimum_pages} pages "
            f"({settings.min_ranking_coverage:.0%} coverage)"
        )
        if blocked:
            raise CrawlBlockedError(f"{detail}; requests were being refused (403/429)")
        raise IncompleteCrawlError(detail)

    if pending:
        logger.warning(
            f"{len(pending)} page(s) still missing for {slug}, but coverage "
            f"({fetched_pages}/{page_max} pages, {len(rows)} rows) is above "
            "the threshold"
        )
    logger.success(f"fetched {len(rows)} ranking rows for {slug}")
    return rows
