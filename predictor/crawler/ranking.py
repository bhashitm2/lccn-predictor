"""Fetch a contest's full ranking from LeetCode's public API.

Endpoints (no auth required):
  * ``GET {base}/contest/api/ranking/{slug}/``                 -> meta (``user_num``)
  * ``GET {base}/contest/api/ranking/{slug}/?pagination=N&region=global``
        -> page N of 25 ranking rows (``total_rank``)
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from math import ceil
from typing import Callable, List, Optional

from loguru import logger

from predictor.config import get_settings
from predictor.crawler.http import fetch_all

PAGE_SIZE = 25


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


async def fetch_user_num(slug: str) -> Optional[int]:
    """Return the number of participants, or None if the contest can't be read."""
    url = f"{_base_url()}/contest/api/ranking/{slug}/"
    resp = (await fetch_all({"meta": {"url": url, "method": "GET"}}, concurrency=1))[
        "meta"
    ]
    if resp is None:
        return None
    return resp.json().get("user_num")


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

    requests = {
        page: {
            "url": f"{base}/contest/api/ranking/{slug}/?pagination={page}&region=global",
            "method": "GET",
        }
        for page in range(1, page_max + 1)
    }
    logger.info(f"fetching {page_max} ranking pages for {slug} (effective={effective})")
    responses = await fetch_all(
        requests,
        concurrency=settings.ranking_concurrency,
        retry=settings.http_retry,
        timeout=settings.http_timeout_seconds,
    )

    rows: List[RankingRow] = []
    for page in sorted(responses):
        resp = responses[page]
        if resp is None:
            continue
        for raw in resp.json().get("total_rank", []):
            row = _parse_row(raw)
            if row is not None:
                rows.append(row)
        if progress_cb:
            progress_cb(len(rows))

    rows.sort(key=lambda r: r.rank)
    if limit:
        rows = rows[:limit]
    logger.success(f"fetched {len(rows)} ranking rows for {slug}")
    return rows
