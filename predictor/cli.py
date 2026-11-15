"""Command-line entrypoint for running a prediction to completion.

Used by the GitHub Actions crawler (``.github/workflows/crawl-cron.yml``) which
runs the crawl from an Actions runner — whose IP can reach LeetCode's ranking
API via curl_cffi — and writes results to MongoDB Atlas. The deployed (Render)
API then serves those cached results.

Usage:
    python -m predictor.cli predict-recent [--refresh weekly|biweekly] [--limit N]
    python -m predictor.cli predict-latest [--force] [--limit N]
    python -m predictor.cli predict <slug> [--force] [--limit N]
    python -m predictor.cli backfill [--count N] [--force] [--limit N]
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timedelta, timezone

from loguru import logger

from predictor.crawler.contest_list import (
    fetch_latest_contest_slug,
    fetch_past_contests,
)
from predictor.config import get_settings
from predictor.crawler.http import CrawlBlockedError, close_http_session
from predictor.db.models import Contest, utcnow
from predictor.db.mongodb import close_db, init_db
from predictor.service.predict_service import run_prediction

# Exit codes. 75 is EX_TEMPFAIL: "we were refused, try again later" — distinct
# from a real failure, so the workflow can retry it and label it correctly.
EXIT_OK = 0
EXIT_FAILED = 1
EXIT_UNRESOLVED = 2
EXIT_BLOCKED = 75


async def _predict_one(slug: str, force: bool, limit: int | None) -> int:
    """Run one prediction. Returns an exit code."""
    logger.info(f"predicting {slug} (force={force}, limit={limit})")
    contest = await run_prediction(slug, limit=limit, force=force)
    logger.info(
        f"  -> status={contest.status} records={contest.total_records} "
        f"error={contest.error}"
    )
    # With force=False a "done" status can only mean success — either the run
    # completed or it was a no-op on an already-predicted contest. With
    # force=True the run always executes, and a failed refresh rolls the doc
    # back to "done" while recording why, so `error` is what distinguishes it.
    if contest.status == "done" and (not force or contest.error is None):
        return EXIT_OK
    if contest.error and contest.error.startswith(CrawlBlockedError.__name__):
        return EXIT_BLOCKED
    return EXIT_FAILED


async def _run(slug: str | None, force: bool, limit: int | None) -> int:
    await init_db()
    try:
        if slug in (None, "latest"):
            try:
                slug = await fetch_latest_contest_slug()
            except CrawlBlockedError as exc:
                logger.error(f"blocked while resolving the latest contest: {exc}")
                return EXIT_BLOCKED
            if not slug:
                logger.error("could not resolve the latest contest slug")
                return EXIT_UNRESOLVED
        return await _predict_one(slug, force, limit)
    finally:
        await close_http_session()
        await close_db()


async def _predict_recent(refresh: str | None, limit: int | None) -> int:
    """Predict every recently finished contest that still needs it.

    This is what the scheduled slots run, so each slot also catches up on
    whatever an earlier slot was blocked from — "latest" alone gave a biweekly
    contest exactly one chance, because by the next slot the latest finished
    contest was already the weekly.

    A contest is predicted if it isn't ``done`` yet. With ``refresh`` set to
    ``weekly`` or ``biweekly``, a done contest of that kind is re-predicted
    (post-purge refresh) when its prediction is older than
    ``refresh_min_age_hours`` — so a retry of this command on a fresh runner
    skips a refresh an earlier attempt already completed.
    """
    settings = get_settings()
    try:
        contests = await fetch_past_contests(1)
    except CrawlBlockedError as exc:
        logger.error(f"blocked while listing recent contests: {exc}")
        return EXIT_BLOCKED
    if not contests:
        logger.error("could not fetch past contests")
        return EXIT_UNRESOLVED

    now = utcnow()
    oldest = (now - timedelta(hours=settings.recent_contest_max_age_hours)).timestamp()
    recent = sorted(
        (c for c in contests if c[2] and c[2] >= oldest), key=lambda c: c[2]
    )
    if not recent:
        logger.info(
            f"no contest started within {settings.recent_contest_max_age_hours:g}h; "
            "nothing to do"
        )
        return EXIT_OK

    refresh_before = now - timedelta(hours=settings.refresh_min_age_hours)
    codes: list[int] = []
    for _title, slug, _start in recent:
        contest = await Contest.find_one(Contest.title_slug == slug)
        if contest is None or contest.status != "done":
            codes.append(await _predict_one(slug, False, limit))
        elif (
            refresh
            and slug.startswith(f"{refresh}-contest-")
            and _predicted_before(contest, refresh_before)
        ):
            codes.append(await _predict_one(slug, True, limit))
        else:
            logger.info(f"{slug} already predicted; skipping")

    if EXIT_BLOCKED in codes:
        return EXIT_BLOCKED
    if any(code != EXIT_OK for code in codes):
        return EXIT_FAILED
    return EXIT_OK


def _predicted_before(contest: Contest, when: datetime) -> bool:
    predicted = contest.predicted_at
    if predicted is None:
        return True
    # Mongo hands back naive datetimes; they are UTC.
    if predicted.tzinfo is None:
        predicted = predicted.replace(tzinfo=timezone.utc)
    return predicted < when


async def _run_recent(refresh: str | None, limit: int | None) -> int:
    await init_db()
    try:
        return await _predict_recent(refresh, limit)
    finally:
        await close_http_session()
        await close_db()


async def _run_backfill(count: int, force: bool, limit: int | None) -> int:
    """Predict the ``count`` most recently finished contests (newest first)."""
    await init_db()
    try:
        contests: list = []
        page = 1
        try:
            while len(contests) < count and page <= 10:
                batch = await fetch_past_contests(page)
                if not batch:
                    break
                contests.extend(batch)
                page += 1
        except CrawlBlockedError as exc:
            logger.error(f"blocked while listing past contests: {exc}")
            return EXIT_BLOCKED
        slugs = [c[1] for c in contests[:count]]
        if not slugs:
            logger.error("could not fetch past contests")
            return EXIT_UNRESOLVED
        logger.info(f"backfilling {len(slugs)} contests: {slugs}")
        ok = 0
        blocked = False
        for i, slug in enumerate(slugs, 1):
            logger.info(f"=== [{i}/{len(slugs)}] {slug} ===")
            code = await _predict_one(slug, force, limit)
            if code == EXIT_OK:
                ok += 1
            elif code == EXIT_BLOCKED:
                blocked = True
        logger.info(f"backfill complete: {ok}/{len(slugs)} succeeded")
        if ok == len(slugs):
            return EXIT_OK
        return EXIT_BLOCKED if blocked else EXIT_FAILED
    finally:
        await close_http_session()
        await close_db()


def main() -> None:
    parser = argparse.ArgumentParser(prog="predictor.cli")
    parser.add_argument(
        "command", choices=["predict-recent", "predict-latest", "predict", "backfill"]
    )
    parser.add_argument("slug", nargs="?", default=None, help="contest slug")
    parser.add_argument("--force", action="store_true", help="re-predict if done")
    parser.add_argument("--limit", type=int, default=None, help="top-N only")
    parser.add_argument(
        "--count", type=int, default=10, help="backfill: number of contests"
    )
    parser.add_argument(
        "--refresh",
        choices=["weekly", "biweekly"],
        default=None,
        help="predict-recent: also re-predict done contests of this kind",
    )
    args = parser.parse_args()

    if args.command == "predict-recent":
        sys.exit(asyncio.run(_run_recent(args.refresh, args.limit)))

    if args.command == "backfill":
        sys.exit(asyncio.run(_run_backfill(args.count, args.force, args.limit)))

    slug = "latest" if args.command == "predict-latest" else args.slug
    if args.command == "predict" and not slug:
        parser.error("predict requires a <slug>")
    sys.exit(asyncio.run(_run(slug, args.force, args.limit)))


if __name__ == "__main__":
    main()
