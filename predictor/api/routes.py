"""REST API.

Public (rate-limited, read-only): your website calls
``GET /api/v1/contest/{slug}/predict``. By default these serve **cached** results
only and never start a live crawl.

Admin (API-key gated): the scheduler / GitHub-Actions cron calls these to trigger
crawling+prediction.
"""
from datetime import timedelta
from typing import List, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from pymongo import ASCENDING, DESCENDING

from predictor.api.deps import rate_limit, require_api_key
from predictor.api.schemas import (
    ContestStatus,
    RecentContest,
    ResolvedContest,
    PredictionPage,
    PredictionRecord,
)
from predictor.config import get_settings
from predictor.core.contests import (
    CONTEST_DURATION,
    contest_title,
    contests_started_between,
    estimated_start,
    parse_contest_query,
)
from predictor.db.models import Contest, ContestRecord, utcnow
from predictor.service.predict_service import (
    ensure_prediction_started,
    predict_latest,
)

router = APIRouter(prefix="/api/v1", tags=["prediction"])
admin = APIRouter(
    prefix="/api/v1/admin", tags=["admin"], dependencies=[Depends(require_api_key)]
)


def _to_status(contest: Contest) -> ContestStatus:
    return ContestStatus(
        slug=contest.title_slug,
        status=contest.status,
        user_num=contest.user_num,
        crawled_ranking=contest.crawled_ranking,
        resolved_ratings=contest.resolved_ratings,
        total_records=contest.total_records,
        limit=contest.limit,
        error=contest.error,
        predicted_at=contest.predicted_at,
    )


def _to_record(doc: ContestRecord) -> PredictionRecord:
    return PredictionRecord(
        rank=doc.rank,
        username=doc.username,
        user_slug=doc.user_slug,
        data_region=doc.data_region,
        old_rating=doc.old_rating,
        delta_rating=doc.delta_rating,
        new_rating=doc.new_rating,
        attended_count=doc.attended_count,
        score=doc.score,
        finish_time=doc.finish_time,
    )


async def resolve_contest(text: str) -> Tuple[Optional[str], List[str], Optional[Contest]]:
    """Map free-form input to the slug it most likely means.

    Returns ``(slug, candidates, contest_doc)``. Among the candidates (a bare
    number means weekly *or* biweekly) the one with a finished prediction wins,
    then the one with any prediction state, then the most recent; with nothing
    in the DB the first candidate is returned so callers can report on it.
    """
    candidates = parse_contest_query(text)
    if not candidates:
        return None, [], None
    found = [(slug, await Contest.find_one(Contest.title_slug == slug)) for slug in candidates]
    existing = [(slug, doc) for slug, doc in found if doc is not None]
    if not existing:
        return candidates[0], candidates, None

    def preference(item):
        slug, doc = item
        start = estimated_start(slug)
        return (doc.status != "done", -(start.timestamp() if start else 0))

    slug, doc = min(existing, key=preference)
    return slug, candidates, doc


async def _canonical(slug: str) -> str:
    resolved, _, _ = await resolve_contest(slug)
    return resolved or slug


# --------------------------------------------------------------------------- #
# Public, read-only endpoints
# --------------------------------------------------------------------------- #
@router.get(
    "/contests/resolve",
    response_model=ResolvedContest,
    dependencies=[Depends(rate_limit)],
)
async def resolve(q: str = Query(..., min_length=1, max_length=200)):
    """Turn what a person typed ("Weekly Contest 528", "w528", "bw 170", "528",
    a leetcode.com contest URL, ...) into the canonical contest slug."""
    slug, candidates, doc = await resolve_contest(q)
    if slug is None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Couldn't recognise '{q}' as a contest. Try e.g. "
                "'weekly 528', 'biweekly 195' or 'weekly-contest-528'."
            ),
        )
    return ResolvedContest(
        query=q,
        slug=slug,
        title=contest_title(slug),
        status=doc.status if doc else "pending",
        candidates=candidates,
    )


@router.get(
    "/contests/recent",
    response_model=List[RecentContest],
    dependencies=[Depends(rate_limit)],
)
async def recent_contests(n: int = Query(3, ge=1, le=10)):
    """The ``n`` most recently finished weekly/biweekly contests, newest first,
    each with its prediction status (``done`` means results are ready)."""
    now = utcnow()
    # Weekly + biweekly is 1.5 contests a week, so 8 weeks covers n<=10.
    slugs = contests_started_between(now - timedelta(weeks=8), now - CONTEST_DURATION)
    out = []
    for slug in reversed(slugs[-n:]):
        doc = await Contest.find_one(Contest.title_slug == slug)
        out.append(
            RecentContest(
                slug=slug,
                title=contest_title(slug),
                start_time=estimated_start(slug),
                status=doc.status if doc else "pending",
                total_records=doc.total_records if doc else 0,
                predicted_at=doc.predicted_at if doc else None,
            )
        )
    return out


@router.get(
    "/contest/{slug}/predict",
    response_model=PredictionPage,
    dependencies=[Depends(rate_limit)],
)
async def predict_contest(
    slug: str,
    response: Response,
    page: int = Query(1, ge=1),
    size: int = Query(50, ge=1, le=500),
    sort: str = Query("rank", pattern="^(rank|delta|new_rating)$"),
    limit: Optional[int] = Query(None, ge=1),
):
    """Predicted rating changes for a contest (paginated).

    Serves **cached** results. If a prediction is in progress, returns HTTP 202
    with the current status; if the contest was never queued, returns 404. By default it does NOT start a crawl — that is the
    scheduler/admin's job. (Set ``LCCN_PUBLIC_CRAWL_ENABLED=true`` to let this
    endpoint kick off crawls on demand, e.g. for local/testing.)
    """
    settings = get_settings()
    slug = await _canonical(slug)
    contest = await Contest.find_one(Contest.title_slug == slug)

    if contest is None or contest.status != "done":
        if settings.public_crawl_enabled:
            contest = await ensure_prediction_started(slug, limit=limit)
        if contest is None:
            # Never crawled and nothing will start a crawl from here, so a 202
            # would leave clients polling forever. Say so explicitly instead.
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No prediction available for '{slug}'. Predictions are "
                    "generated automatically for recent weekly/biweekly contests "
                    "only — check the slug, or try a more recent contest."
                ),
            )
        if contest.status != "done":
            response.status_code = 202
            return PredictionPage(
                slug=slug,
                status=contest.status,
                total=0,
                page=page,
                size=size,
                records=[],
            )

    sort_field, direction = {
        "rank": ("rank", ASCENDING),
        "delta": ("delta_rating", DESCENDING),
        "new_rating": ("new_rating", DESCENDING),
    }[sort]

    query = ContestRecord.find(ContestRecord.contest_slug == slug)
    total = await query.count()
    docs = (
        await query.sort((sort_field, direction))
        .skip((page - 1) * size)
        .limit(size)
        .to_list()
    )
    return PredictionPage(
        slug=slug,
        status=contest.status,
        total=total,
        page=page,
        size=size,
        records=[_to_record(d) for d in docs],
    )


@router.get(
    "/contest/{slug}/user/{username}",
    response_model=PredictionRecord,
    dependencies=[Depends(rate_limit)],
)
async def predict_user(slug: str, username: str):
    """A single participant's prediction (by username or user_slug)."""
    slug = await _canonical(slug)
    contest = await Contest.find_one(Contest.title_slug == slug)
    if contest is None or contest.status != "done":
        raise HTTPException(status_code=404, detail="prediction not ready for this slug")
    doc = await ContestRecord.find_one(
        ContestRecord.contest_slug == slug,
        {"$or": [{"username": username}, {"user_slug": username}]},
    )
    if doc is None:
        raise HTTPException(status_code=404, detail="user not found in this contest")
    return _to_record(doc)


@router.get(
    "/contest/{slug}/status",
    response_model=ContestStatus,
    dependencies=[Depends(rate_limit)],
)
async def contest_status(slug: str):
    slug = await _canonical(slug)
    contest = await Contest.find_one(Contest.title_slug == slug)
    if contest is None:
        return ContestStatus(slug=slug, status="pending")
    return _to_status(contest)


# --------------------------------------------------------------------------- #
# Admin, API-key gated (crawl triggers)
# --------------------------------------------------------------------------- #
@admin.post("/contest/{slug}/predict", response_model=ContestStatus)
async def admin_predict(
    slug: str,
    force: bool = Query(False),
    limit: Optional[int] = Query(None, ge=1),
):
    """Trigger a crawl + prediction for ``slug`` (scheduler / cron use this)."""
    contest = await ensure_prediction_started(slug, limit=limit, force=force)
    return _to_status(contest)


@admin.post("/predict-latest")
async def admin_predict_latest(force: bool = Query(False)):
    """Find the most recently finished contest and predict it. The free-tier cron
    calls this twice a week."""
    slug = await predict_latest(force=force)
    if slug is None:
        raise HTTPException(status_code=502, detail="could not resolve latest contest")
    return {"triggered": slug}
