"""Command-line entrypoint for running a prediction to completion.

Used by the GitHub Actions crawler (``.github/workflows/crawl-cron.yml``) which
runs the crawl from an Actions runner — whose IP can reach LeetCode's ranking
API via curl_cffi — and writes results to MongoDB Atlas. The deployed (Render)
API then serves those cached results.

Usage:
    python -m predictor.cli predict-latest [--force] [--limit N]
    python -m predictor.cli predict <slug> [--force] [--limit N]
    python -m predictor.cli backfill [--count N] [--force] [--limit N]
"""
from __future__ import annotations

import argparse
import asyncio
import sys

from loguru import logger

from predictor.crawler.contest_list import (
    fetch_latest_contest_slug,
    fetch_past_contests,
)
from predictor.db.mongodb import close_db, init_db
from predictor.service.predict_service import run_prediction


async def _predict_one(slug: str, force: bool, limit: int | None) -> bool:
    logger.info(f"predicting {slug} (force={force}, limit={limit})")
    contest = await run_prediction(slug, limit=limit, force=force)
    logger.info(
        f"  -> status={contest.status} records={contest.total_records} "
        f"error={contest.error}"
    )
    return contest.status == "done"


async def _run(slug: str | None, force: bool, limit: int | None) -> int:
    await init_db()
    try:
        if slug in (None, "latest"):
            slug = await fetch_latest_contest_slug()
            if not slug:
                logger.error("could not resolve the latest contest slug")
                return 2
        return 0 if await _predict_one(slug, force, limit) else 1
    finally:
        await close_db()


async def _run_backfill(count: int, force: bool, limit: int | None) -> int:
    """Predict the ``count`` most recently finished contests (newest first)."""
    await init_db()
    try:
        contests: list = []
        page = 1
        while len(contests) < count and page <= 10:
            batch = await fetch_past_contests(page)
            if not batch:
                break
            contests.extend(batch)
            page += 1
        slugs = [c[1] for c in contests[:count]]
        if not slugs:
            logger.error("could not fetch past contests")
            return 2
        logger.info(f"backfilling {len(slugs)} contests: {slugs}")
        ok = 0
        for i, slug in enumerate(slugs, 1):
            logger.info(f"=== [{i}/{len(slugs)}] {slug} ===")
            if await _predict_one(slug, force, limit):
                ok += 1
        logger.info(f"backfill complete: {ok}/{len(slugs)} succeeded")
        return 0 if ok == len(slugs) else 1
    finally:
        await close_db()


def main() -> None:
    parser = argparse.ArgumentParser(prog="predictor.cli")
    parser.add_argument(
        "command", choices=["predict-latest", "predict", "backfill"]
    )
    parser.add_argument("slug", nargs="?", default=None, help="contest slug")
    parser.add_argument("--force", action="store_true", help="re-predict if done")
    parser.add_argument("--limit", type=int, default=None, help="top-N only")
    parser.add_argument(
        "--count", type=int, default=10, help="backfill: number of contests"
    )
    args = parser.parse_args()

    if args.command == "backfill":
        sys.exit(asyncio.run(_run_backfill(args.count, args.force, args.limit)))

    slug = "latest" if args.command == "predict-latest" else args.slug
    if args.command == "predict" and not slug:
        parser.error("predict requires a <slug>")
    sys.exit(asyncio.run(_run(slug, args.force, args.limit)))


if __name__ == "__main__":
    main()
