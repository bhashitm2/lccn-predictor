"""Discover recent finished contests via LeetCode's ``pastContests`` GraphQL.

Used by the auto-predict scheduler / admin trigger so we don't have to hardcode
or date-compute contest slugs (LeetCode occasionally reschedules).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from loguru import logger

from predictor.config import get_settings
from predictor.crawler.http import CrawlBlockedError, fetch_all

_PAST_CONTESTS_QUERY = """
query pastContests($pageNo: Int) {
  pastContests(pageNo: $pageNo) {
    data { title titleSlug startTime }
  }
}
"""


async def fetch_past_contests(page_no: int = 1) -> List[Tuple[str, str, int]]:
    """Return ``[(title, titleSlug, startTime), ...]`` newest-first for a page."""
    settings = get_settings()
    req = {
        "main": {
            "url": f"{settings.leetcode_base_us}/graphql/",
            "method": "POST",
            "json": {"query": _PAST_CONTESTS_QUERY, "variables": {"pageNo": page_no}},
        }
    }
    result = (
        await fetch_all(
            req,
            concurrency=1,
            retry=settings.http_retry,
            timeout=settings.http_timeout_seconds,
        )
    )["main"]
    if not result:
        if result.blocked:
            raise CrawlBlockedError(
                "blocked by LeetCode/Cloudflare while listing past contests "
                f"(HTTP {result.status_code} after {result.attempts} attempts)"
            )
        return []
    data = ((result.json().get("data") or {}).get("pastContests") or {}).get("data") or []
    return [
        (c.get("title"), c.get("titleSlug"), c.get("startTime"))
        for c in data
        if c.get("titleSlug")
    ]


async def fetch_latest_contest_slug() -> Optional[str]:
    """Slug of the most recently finished contest, or None."""
    contests = await fetch_past_contests(1)
    if not contests:
        logger.warning("could not fetch past contests")
        return None
    return contests[0][1]
