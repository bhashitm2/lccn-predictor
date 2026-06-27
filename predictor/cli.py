"""Command-line entrypoint for running a prediction to completion.

Used by the GitHub Actions crawler (``.github/workflows/crawl-cron.yml``) which
runs the crawl from an Actions runner — whose IP can reach LeetCode's ranking
API via curl_cffi — and writes results to MongoDB Atlas. The deployed (Render)
API then serves those cached results.

Usage:
    python -m predictor.cli predict-latest [--force] [--limit N]
    python -m predictor.cli predict <slug> [--force] [--limit N]
"""
from __future__ import annotations

import argparse
import asyncio
import sys

from loguru import logger

from predictor.crawler.contest_list import fetch_latest_contest_slug
from predictor.db.mongodb import close_db, init_db
from predictor.service.predict_service import run_prediction


async def _run(slug: str | None, force: bool, limit: int | None) -> int:
    await init_db()
    try:
        if slug in (None, "latest"):
            slug = await fetch_latest_contest_slug()
            if not slug:
                logger.error("could not resolve the latest contest slug")
                return 2
        logger.info(f"predicting {slug} (force={force}, limit={limit})")
        contest = await run_prediction(slug, limit=limit, force=force)
        logger.info(
            f"result: status={contest.status} "
            f"records={contest.total_records} error={contest.error}"
        )
        return 0 if contest.status == "done" else 1
    finally:
        await close_db()


def main() -> None:
    parser = argparse.ArgumentParser(prog="predictor.cli")
    parser.add_argument("command", choices=["predict-latest", "predict"])
    parser.add_argument("slug", nargs="?", default=None, help="contest slug")
    parser.add_argument("--force", action="store_true", help="re-predict if done")
    parser.add_argument("--limit", type=int, default=None, help="top-N only")
    args = parser.parse_args()

    slug = "latest" if args.command == "predict-latest" else args.slug
    if args.command == "predict" and not slug:
        parser.error("predict requires a <slug>")
    sys.exit(asyncio.run(_run(slug, args.force, args.limit)))


if __name__ == "__main__":
    main()
