"""Optional background warm-up of the user-rating cache.

Disabled by default (``LCCN_SCHEDULER_ENABLED=false``). When enabled it
periodically refreshes the most-stale cached ratings via GraphQL, so the next
contest's cold-start crawl resolves more users from cache instead of the network.
"""
from __future__ import annotations

from typing import Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from loguru import logger

from predictor.config import get_settings
from predictor.crawler.user_rating import fetch_ratings
from predictor.db.models import UserRatingCache, utcnow

_scheduler: Optional[AsyncIOScheduler] = None

# how many of the stalest cache rows to refresh per tick
_BATCH = 2000


async def warm_up_rating_cache() -> None:
    docs = (
        await UserRatingCache.find()
        .sort("updated_at")  # oldest first
        .limit(_BATCH)
        .to_list()
    )
    if not docs:
        logger.info("rating-cache warm-up: nothing to refresh")
        return
    keys = [(d.user_slug, d.data_region) for d in docs]
    fetched = await fetch_ratings(keys)
    now = utcnow()
    for d in docs:
        rating, attended = fetched[(d.user_slug, d.data_region)]
        d.rating, d.attended_count, d.updated_at = rating, attended, now
        await d.save()
    logger.success(f"rating-cache warm-up: refreshed {len(docs)} users")


async def auto_predict_latest() -> None:
    """Predict the most recently finished contest if it isn't done yet.

    Best for always-on/paid hosting. On free tiers (which sleep when idle) use the
    GitHub Actions cron hitting POST /api/v1/admin/predict-latest instead.
    """
    # imported here to avoid a circular import at module load
    from predictor.service.predict_service import predict_latest

    try:
        slug = await predict_latest()
        logger.info(f"auto-predict tick: latest={slug}")
    except Exception:
        logger.exception("auto-predict tick failed")


def start_scheduler() -> None:
    global _scheduler
    settings = get_settings()
    if not (settings.scheduler_enabled or settings.auto_predict_enabled):
        logger.info(
            "scheduler disabled (LCCN_SCHEDULER_ENABLED / "
            "LCCN_AUTO_PREDICT_ENABLED both false)"
        )
        return
    _scheduler = AsyncIOScheduler(timezone="UTC")
    if settings.scheduler_enabled:
        _scheduler.add_job(
            warm_up_rating_cache,
            "interval",
            minutes=settings.scheduler_interval_minutes,
            id="rating_cache_warm_up",
            max_instances=1,
            coalesce=True,
        )
        logger.info(
            f"scheduled rating-cache warm-up every "
            f"{settings.scheduler_interval_minutes} min"
        )
    if settings.auto_predict_enabled:
        _scheduler.add_job(
            auto_predict_latest,
            "interval",
            minutes=settings.auto_predict_interval_minutes,
            id="auto_predict_latest",
            max_instances=1,
            coalesce=True,
        )
        logger.info(
            f"scheduled auto-predict every "
            f"{settings.auto_predict_interval_minutes} min"
        )
    _scheduler.start()


def stop_scheduler() -> None:
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
