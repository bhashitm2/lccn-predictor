"""Orchestrates a crawl-by-slug prediction.

Pipeline (``run_prediction``):
  1. mark Contest ``crawling`` and read ``user_num``
  2. fetch the ranking pages
  3. resolve each participant's pre-contest rating: reuse fresh entries from
     ``UserRatingCache``, fetch the rest via GraphQL, then refresh the cache
  4. mark ``predicting`` and run the Elo/FFT engine
  5. persist one ``ContestRecord`` per participant, mark Contest ``done``

Jobs run as background asyncio tasks; the API kicks one off and polls
``Contest.status``. An in-process guard prevents duplicate concurrent jobs for
the same slug.
"""
from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Dict, List, Optional, Set, Tuple

from loguru import logger

from predictor.config import get_settings
from predictor.core.engine import predict
from predictor.crawler.contest_list import fetch_latest_contest_slug
from predictor.crawler.ranking import RankingRow, fetch_ranking, fetch_user_num
from predictor.crawler.user_rating import fetch_ratings
from predictor.db.models import Contest, ContestRecord, UserRatingCache, utcnow

_running: Set[str] = set()
_lock = asyncio.Lock()

RatingKey = Tuple[str, str]  # (user_slug, data_region)


def _chunks(seq: list, size: int):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _pymongo_collection(model):
    """Raw async collection, compatible with Beanie 2.x (pymongo) and 1.x (motor)."""
    getter = getattr(model, "get_pymongo_collection", None) or getattr(
        model, "get_motor_collection"
    )
    return getter()


async def ensure_prediction_started(
    slug: str, *, limit: Optional[int] = None, force: bool = False
) -> Contest:
    """Idempotently ensure a prediction job exists/runs for ``slug``.

    Returns the current Contest doc (creating a ``pending`` one if needed). If a
    completed prediction exists and ``force`` is False, it is left untouched.
    """
    contest = await Contest.find_one(Contest.title_slug == slug)
    if contest is None:
        contest = Contest(title_slug=slug, status="pending")
        await contest.insert()

    if contest.status == "done" and not force:
        return contest

    async with _lock:
        if slug in _running:
            return contest
        _running.add(slug)

    # fire-and-forget; status is tracked in the DB
    asyncio.create_task(_run_guarded(slug, limit=limit, force=force))
    return contest


async def predict_latest(*, force: bool = False) -> Optional[str]:
    """Find the most recently finished contest and ensure it's predicted.

    Returns the slug it acted on (or None if it couldn't be resolved). Used by the
    scheduler and the admin 'predict latest' endpoint.
    """
    slug = await fetch_latest_contest_slug()
    if slug is None:
        return None
    contest = await Contest.find_one(Contest.title_slug == slug)
    if contest and contest.status == "done" and not force:
        logger.info(f"latest contest {slug} already predicted")
        return slug
    logger.info(f"auto-predicting latest contest: {slug}")
    await ensure_prediction_started(slug, force=force)
    return slug


async def _run_guarded(slug: str, *, limit: Optional[int], force: bool) -> None:
    try:
        await run_prediction(slug, limit=limit, force=force)
    finally:
        async with _lock:
            _running.discard(slug)


async def run_prediction(
    slug: str, *, limit: Optional[int] = None, force: bool = False
) -> Contest:
    """Run the full crawl+predict pipeline synchronously. Updates Contest status."""
    contest = await Contest.find_one(Contest.title_slug == slug)
    if contest is None:
        contest = Contest(title_slug=slug)
        await contest.insert()

    if contest.status == "done" and not force:
        return contest

    try:
        # 1. resolve user_num ------------------------------------------------
        contest.status = "crawling"
        contest.error = None
        contest.limit = limit
        contest.crawled_ranking = 0
        contest.resolved_ratings = 0
        contest.updated_at = utcnow()
        await contest.save()

        user_num = await fetch_user_num(slug)
        if not user_num:
            raise RuntimeError(
                f"contest '{slug}' not found or has no participants "
                "(check the slug, e.g. 'weekly-contest-450')"
            )
        contest.user_num = user_num
        await contest.save()

        # 2. crawl ranking ---------------------------------------------------
        # (ranking pages are fetched as one concurrent batch, so we record the
        # final count rather than racing per-page DB writes)
        rows = await fetch_ranking(slug, user_num, limit=limit)
        if not rows:
            raise RuntimeError(f"no ranking rows fetched for '{slug}'")
        contest.crawled_ranking = len(rows)
        contest.total_records = len(rows)
        await contest.save()

        # 3. resolve ratings (cache + batched GraphQL) -----------------------
        contest.status = "resolving"
        await contest.save()
        ratings_map = await _resolve_ratings(rows)
        contest.resolved_ratings = len(ratings_map)
        contest.status = "predicting"
        await contest.save()

        # 4. predict ---------------------------------------------------------
        rows.sort(key=lambda r: r.rank)
        ranks = [r.rank for r in rows]
        old_ratings = [ratings_map[(r.user_slug, r.data_region)][0] for r in rows]
        attended = [ratings_map[(r.user_slug, r.data_region)][1] for r in rows]
        # CPU-bound (FFT + per-user binary search) -> off the event loop
        deltas, new_ratings = await asyncio.to_thread(
            predict, ranks, old_ratings, attended
        )

        # 5. persist ---------------------------------------------------------
        await _persist_records(slug, rows, old_ratings, attended, deltas, new_ratings)
        contest.status = "done"
        contest.predicted_at = utcnow()
        contest.updated_at = utcnow()
        await contest.save()
        logger.success(f"prediction done for {slug}: {len(rows)} records")
        return contest

    except Exception as exc:  # noqa: BLE001 - record any failure for the API
        logger.exception(f"prediction failed for {slug}")
        contest.status = "error"
        contest.error = str(exc)
        contest.updated_at = utcnow()
        await contest.save()
        return contest


async def _resolve_ratings(
    rows: List[RankingRow],
) -> Dict[RatingKey, Tuple[float, int]]:
    """Return ``(slug, region) -> (rating, attended)`` using the cache where fresh."""
    settings = get_settings()
    ttl = timedelta(hours=settings.rating_cache_ttl_hours)
    fresh_after = utcnow() - ttl

    keys: List[RatingKey] = list({(r.user_slug, r.data_region) for r in rows})
    result: Dict[RatingKey, Tuple[float, int]] = {}

    # read cache in chunks (avoid an enormous $in)
    slugs = [k[0] for k in keys]
    cache_map: Dict[RatingKey, UserRatingCache] = {}
    for chunk in _chunks(slugs, 5000):
        docs = await UserRatingCache.find(
            {"user_slug": {"$in": chunk}}
        ).to_list()
        for d in docs:
            cache_map[(d.user_slug, d.data_region)] = d

    stale: List[RatingKey] = []
    for key in keys:
        doc = cache_map.get(key)
        if doc is not None and _aware(doc.updated_at) >= fresh_after:
            result[key] = (doc.rating, doc.attended_count)
        else:
            stale.append(key)

    logger.info(
        f"ratings: {len(result)} from cache, {len(stale)} to fetch via GraphQL"
    )

    if stale:
        fetched = await fetch_ratings(stale)
        result.update(fetched)
        await _upsert_cache(rows, fetched)

    return result


def _aware(dt):
    """Mongo may return naive datetimes; treat them as UTC for comparison."""
    from datetime import timezone

    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


async def _upsert_cache(
    rows: List[RankingRow], fetched: Dict[RatingKey, Tuple[float, int]]
) -> None:
    # map slug+region -> a representative row (for username)
    row_by_key = {(r.user_slug, r.data_region): r for r in rows}
    coll = _pymongo_collection(UserRatingCache)
    now = utcnow()

    async def upsert_one(key: RatingKey) -> None:
        slug, region = key
        rating, attended = fetched[key]
        row = row_by_key.get(key)
        await coll.update_one(
            {"user_slug": slug, "data_region": region},
            {
                "$set": {
                    "username": row.username if row else slug,
                    "user_slug": slug,
                    "data_region": region,
                    "rating": rating,
                    "attended_count": attended,
                    "updated_at": now,
                }
            },
            upsert=True,
        )

    # Run upserts concurrently but in bounded batches (avoid exhausting the pool).
    keys = list(fetched.keys())
    for batch in _chunks(keys, 500):
        await asyncio.gather(*(upsert_one(k) for k in batch))


async def _persist_records(
    slug: str,
    rows: List[RankingRow],
    old_ratings,
    attended,
    deltas,
    new_ratings,
) -> None:
    # replace any previous prediction for this slug
    await ContestRecord.find(ContestRecord.contest_slug == slug).delete()
    # Dedupe by user_slug (the unique handle) keeping the best rank, in case the
    # live ranking shifted during crawl and a user appeared on two pages.
    seen: set = set()
    docs = []
    for i, r in enumerate(rows):
        if r.user_slug in seen:
            continue
        seen.add(r.user_slug)
        docs.append(
            ContestRecord(
                contest_slug=slug,
                username=r.username,
                user_slug=r.user_slug,
                data_region=r.data_region,
                rank=r.rank,
                score=r.score,
                finish_time=r.finish_time,
                old_rating=float(old_ratings[i]),
                attended_count=int(attended[i]),
                delta_rating=round(float(deltas[i]), 2),
                new_rating=round(float(new_ratings[i]), 2),
            )
        )
    for chunk in _chunks(docs, 1000):
        await ContestRecord.insert_many(chunk)
