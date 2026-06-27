"""Fetch each user's pre-contest rating + attended-contest count via GraphQL.

The ranking API does NOT include a participant's rating, so we query LeetCode's
GraphQL ``userContestRanking``. To make this fast for large contests we **batch
many users into one request using GraphQL aliases** (e.g. 40 users per HTTP
call), instead of one request per user — turning tens of thousands of requests
into a few hundred.

A ``null`` result means LeetCode has no contest history for that user — a
newcomer — so we apply the documented defaults (rating 1500, attended 0).

IMPORTANT (timing): ``userContestRanking.rating`` is the user's *current* rating.
It equals the pre-contest rating only until LeetCode publishes the new ratings
(a few hours after a contest ends). Crawl within that window for accurate input.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from loguru import logger

from predictor.config import get_settings
from predictor.crawler.http import fetch_all

# New-user defaults (see lccn_predictor app/constants.py)
DEFAULT_RATING = 1500.0
DEFAULT_ATTENDED_COUNT = 0

# key: (user_slug, data_region); value: (rating, attended_count)
RatingKey = Tuple[str, str]


def _chunks(seq: list, size: int):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _build_batch_request(chunk: List[RatingKey], region: str) -> dict:
    """Build one aliased GraphQL request resolving every user in ``chunk``."""
    settings = get_settings()
    n = len(chunk)
    arg = "userSlug" if region == "CN" else "username"
    var_defs = ", ".join(f"$u{i}: String!" for i in range(n))
    fields = "\n".join(
        f"  u{i}: userContestRanking({arg}: $u{i}) "
        "{ rating attendedContestsCount }"
        for i in range(n)
    )
    query = f"query batch({var_defs}) {{\n{fields}\n}}"
    variables = {f"u{i}": chunk[i][0] for i in range(n)}
    url = (
        f"{settings.leetcode_base_cn}/graphql/noj-go/"
        if region == "CN"
        else f"{settings.leetcode_base_us}/graphql/"
    )
    return {"url": url, "method": "POST", "json": {"query": query, "variables": variables}}


async def fetch_ratings(
    users: List[RatingKey],
) -> Dict[RatingKey, Tuple[float, int]]:
    """Resolve ratings for ``(user_slug, data_region)`` keys via batched GraphQL.

    Newcomers (and anyone unresolved after retries) get the (1500, 0) defaults so
    a prediction is never blocked by missing lookups.
    """
    settings = get_settings()
    batch_size = settings.rating_batch_size

    # Region matters: US uses `username`, CN uses `userSlug`, different endpoints.
    by_region: Dict[str, List[RatingKey]] = {"US": [], "CN": []}
    for key in users:
        by_region.setdefault(key[1], by_region["US"]).append(key)

    requests: Dict[int, dict] = {}
    batch_members: Dict[int, List[RatingKey]] = {}
    bidx = 0
    for region, group in by_region.items():
        for chunk in _chunks(group, batch_size):
            requests[bidx] = _build_batch_request(chunk, region)
            batch_members[bidx] = chunk
            bidx += 1

    logger.info(
        f"resolving {len(users)} ratings in {len(requests)} batched requests "
        f"(batch_size={batch_size}, concurrency={settings.rating_concurrency})"
    )
    responses = await fetch_all(
        requests,
        concurrency=settings.rating_concurrency,
        retry=settings.http_retry,
        timeout=settings.http_timeout_seconds,
    )

    result: Dict[RatingKey, Tuple[float, int]] = {}
    newcomers = 0
    failed_batches = 0
    for bid, members in batch_members.items():
        resp = responses.get(bid)
        data = {}
        if resp is not None:
            data = (resp.json().get("data") or {})
        else:
            failed_batches += 1
        for i, key in enumerate(members):
            node = data.get(f"u{i}")
            if node is not None:
                result[key] = (
                    float(node.get("rating", DEFAULT_RATING)),
                    int(node.get("attendedContestsCount", DEFAULT_ATTENDED_COUNT)),
                )
            else:
                result[key] = (DEFAULT_RATING, DEFAULT_ATTENDED_COUNT)
                newcomers += 1
    logger.info(
        f"resolved {len(result)} ratings "
        f"({newcomers} newcomer/default, {failed_batches} failed batches)"
    )
    return result
