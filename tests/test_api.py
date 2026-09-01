"""API + orchestration tests with the crawler mocked and an in-memory Mongo.

We drive the prediction pipeline deterministically by calling ``run_prediction``
directly (instead of the fire-and-forget background task) and then exercise the
route handlers against the persisted results.
"""
from datetime import datetime, timezone

import pytest

from predictor.api import routes
from predictor.crawler.http import CrawlBlockedError
from predictor.crawler.ranking import IncompleteCrawlError, RankingRow
from predictor.db.models import Contest, ContestRecord
from predictor.service import predict_service


def _fake_rows(n: int):
    """n participants, ranks 1..n, two CN users to exercise both regions."""
    rows = []
    for i in range(1, n + 1):
        rows.append(
            RankingRow(
                username=f"user{i}",
                user_slug=f"user{i}",
                data_region="CN" if i % 50 == 0 else "US",
                rank=i,
                score=max(0, 30 - i),
                finish_time=datetime(2026, 6, 1, tzinfo=timezone.utc),
            )
        )
    return rows


@pytest.fixture
def mock_crawler(monkeypatch):
    n = 120
    rows = _fake_rows(n)

    async def fake_user_num(slug):
        return n

    async def fake_fetch_ranking(slug, user_num, *, limit=None, progress_cb=None):
        out = rows[:limit] if limit else rows
        return list(out)

    async def fake_fetch_ratings(keys):
        # deterministic spread of ratings + attended counts
        result = {}
        for idx, (slug, region) in enumerate(keys):
            result[(slug, region)] = (1400.0 + (idx % 40) * 20, idx % 30)
        return result

    monkeypatch.setattr(predict_service, "fetch_user_num", fake_user_num)
    monkeypatch.setattr(predict_service, "fetch_ranking", fake_fetch_ranking)
    monkeypatch.setattr(predict_service, "fetch_ratings", fake_fetch_ratings)
    return {"n": n}


async def test_full_pipeline_persists(db, mock_crawler):
    contest = await predict_service.run_prediction("weekly-contest-test")
    assert contest.status == "done"
    assert contest.total_records == mock_crawler["n"]

    count = await ContestRecord.find(
        ContestRecord.contest_slug == "weekly-contest-test"
    ).count()
    assert count == mock_crawler["n"]

    # every record has a delta and a new rating
    docs = await ContestRecord.find(
        ContestRecord.contest_slug == "weekly-contest-test"
    ).to_list()
    assert all(d.delta_rating is not None and d.new_rating is not None for d in docs)


async def test_predict_endpoint_paginates(db, mock_crawler):
    await predict_service.run_prediction("weekly-contest-test")

    class FakeResp:
        status_code = 200

    page = await routes.predict_contest(
        "weekly-contest-test", FakeResp(), page=1, size=10, sort="rank", limit=None,
    )
    assert page.status == "done"
    assert page.total == mock_crawler["n"]
    assert len(page.records) == 10
    # default sort is by rank ascending
    assert [r.rank for r in page.records] == list(range(1, 11))


async def test_sort_by_delta(db, mock_crawler):
    await predict_service.run_prediction("weekly-contest-test")

    class FakeResp:
        status_code = 200

    page = await routes.predict_contest(
        "weekly-contest-test", FakeResp(), page=1, size=5, sort="delta", limit=None,
    )
    deltas = [r.delta_rating for r in page.records]
    assert deltas == sorted(deltas, reverse=True)


async def test_single_user_lookup(db, mock_crawler):
    await predict_service.run_prediction("weekly-contest-test")
    rec = await routes.predict_user("weekly-contest-test", "user1")
    assert rec.username == "user1"
    assert rec.rank == 1


async def test_user_lookup_404_when_not_ready(db):
    with pytest.raises(Exception):
        await routes.predict_user("nonexistent-contest", "user1")


async def test_status_pending_for_unknown(db):
    status = await routes.contest_status("never-seen")
    assert status.status == "pending"


async def test_error_status_on_missing_contest(db, monkeypatch):
    async def no_users(slug):
        return None

    monkeypatch.setattr(predict_service, "fetch_user_num", no_users)
    contest = await predict_service.run_prediction("bad-slug")
    assert contest.status == "error"
    assert contest.error


async def test_failed_forced_recrawl_keeps_previous_prediction(
    db, mock_crawler, monkeypatch
):
    """A blocked re-crawl must not take a good prediction offline.

    The API serves 404 for anything not `done`, so flipping status to "error"
    on a failed refresh loses a perfectly good prediction until some later run
    happens to succeed.
    """
    slug = "weekly-contest-517"
    contest = await predict_service.run_prediction(slug)
    assert contest.status == "done"
    assert contest.total_records == mock_crawler["n"]

    async def blocked(_slug):
        raise CrawlBlockedError("403 from Cloudflare")

    monkeypatch.setattr(predict_service, "fetch_user_num", blocked)
    contest = await predict_service.run_prediction(slug, force=True)

    assert contest.status == "done", "previous prediction must stay served"
    assert contest.total_records == mock_crawler["n"]
    assert contest.error and "CrawlBlockedError" in contest.error

    # ...and the records themselves are untouched.
    count = await ContestRecord.find(ContestRecord.contest_slug == slug).count()
    assert count == mock_crawler["n"]

    # The route still serves the cached prediction.
    class FakeResp:
        status_code = 200

    page = await routes.predict_contest(
        slug, FakeResp(), page=1, size=10, sort="rank", limit=None
    )
    assert page.status == "done" and page.total == mock_crawler["n"]


async def test_failed_first_crawl_still_marks_error(db, monkeypatch):
    """With no prior prediction to protect, a failure is a plain error."""

    async def blocked(_slug):
        raise CrawlBlockedError("403 from Cloudflare")

    monkeypatch.setattr(predict_service, "fetch_user_num", blocked)
    contest = await predict_service.run_prediction("weekly-contest-999", force=True)

    assert contest.status == "error"
    assert "CrawlBlockedError" in contest.error


async def test_partial_crawl_does_not_replace_good_records(
    db, mock_crawler, monkeypatch
):
    """The corrupting scenario: a truncated field must never be persisted."""
    slug = "weekly-contest-517"
    await predict_service.run_prediction(slug)

    async def truncated(_slug, user_num, *, limit=None, progress_cb=None):
        raise IncompleteCrawlError(
            f"only 350/{user_num} ranking rows for '{_slug}'"
        )

    monkeypatch.setattr(predict_service, "fetch_ranking", truncated)
    contest = await predict_service.run_prediction(slug, force=True)

    assert contest.status == "done"
    count = await ContestRecord.find(ContestRecord.contest_slug == slug).count()
    assert count == mock_crawler["n"], "good records must survive a partial crawl"


# --------------------------------------------------------------------------- #
# Hardening: auth, serve-only, rate limit, predict-latest
# --------------------------------------------------------------------------- #
from types import SimpleNamespace  # noqa: E402

from fastapi import HTTPException  # noqa: E402

from predictor.api import deps  # noqa: E402
from predictor.config import Settings  # noqa: E402


def _patch_settings(monkeypatch, **overrides):
    base = dict(public_crawl_enabled=False, api_key="", rate_limit_per_minute=0)
    base.update(overrides)
    s = Settings(**base)
    monkeypatch.setattr(routes, "get_settings", lambda: s)
    monkeypatch.setattr(deps, "get_settings", lambda: s)
    return s


class _Resp:
    status_code = 200


async def test_serve_only_does_not_crawl(db, monkeypatch):
    """With public crawling disabled, an unknown slug returns 202 and creates no
    contest doc (no crawl is started)."""
    _patch_settings(monkeypatch, public_crawl_enabled=False)
    resp = _Resp()
    page = await routes.predict_contest(
        "never-crawled", resp, page=1, size=10, sort="rank", limit=None
    )
    assert resp.status_code == 202
    assert page.status == "pending" and page.total == 0
    assert await Contest.find_one(Contest.title_slug == "never-crawled") is None


async def test_serve_only_serves_cached_after_admin(db, mock_crawler, monkeypatch):
    _patch_settings(monkeypatch, public_crawl_enabled=False)
    # admin/scheduler path produced the prediction
    await predict_service.run_prediction("weekly-contest-test")
    resp = _Resp()
    page = await routes.predict_contest(
        "weekly-contest-test", resp, page=1, size=10, sort="rank", limit=None
    )
    assert resp.status_code == 200
    assert page.status == "done" and page.total == mock_crawler["n"]


async def test_require_api_key(monkeypatch):
    _patch_settings(monkeypatch, api_key="secret")
    with pytest.raises(HTTPException) as e:
        await deps.require_api_key(x_api_key=None)
    assert e.value.status_code == 401
    with pytest.raises(HTTPException):
        await deps.require_api_key(x_api_key="wrong")
    await deps.require_api_key(x_api_key="secret")  # correct -> no raise

    # when no key is configured, the dependency allows (dev mode)
    _patch_settings(monkeypatch, api_key="")
    await deps.require_api_key(x_api_key=None)


async def test_rate_limit(monkeypatch):
    _patch_settings(monkeypatch, rate_limit_per_minute=3)
    deps._hits.clear()
    req = SimpleNamespace(headers={}, client=SimpleNamespace(host="9.9.9.9"))
    for _ in range(3):
        await deps.rate_limit(req)  # within limit
    with pytest.raises(HTTPException) as e:
        await deps.rate_limit(req)
    assert e.value.status_code == 429


async def test_predict_latest_triggers(db, monkeypatch):
    called = {}

    async def fake_latest():
        return "weekly-contest-test"

    async def fake_ensure(slug, **kw):
        called["slug"] = slug
        return Contest(title_slug=slug, status="pending")

    monkeypatch.setattr(predict_service, "fetch_latest_contest_slug", fake_latest)
    monkeypatch.setattr(predict_service, "ensure_prediction_started", fake_ensure)
    slug = await predict_service.predict_latest()
    assert slug == "weekly-contest-test"
    assert called["slug"] == "weekly-contest-test"


# --------------------------------------------------------------------------- #
# CLI exit codes — the workflow's retry loop keys off these.
# --------------------------------------------------------------------------- #


async def test_cli_exit_code_blocked_is_retryable(db, monkeypatch):
    """Exit 75 (EX_TEMPFAIL) tells the workflow to try again rather than fail."""
    from predictor import cli

    async def blocked(_slug):
        raise CrawlBlockedError("403 from Cloudflare")

    monkeypatch.setattr(predict_service, "fetch_user_num", blocked)
    code = await cli._predict_one("weekly-contest-517", force=False, limit=None)
    assert code == cli.EXIT_BLOCKED


async def test_cli_exit_code_missing_contest_is_hard_failure(db, monkeypatch):
    """A genuinely absent contest must not be retried as if it were a block."""
    from predictor import cli

    async def no_users(_slug):
        return None

    monkeypatch.setattr(predict_service, "fetch_user_num", no_users)
    code = await cli._predict_one("nope-contest", force=False, limit=None)
    assert code == cli.EXIT_FAILED


async def test_cli_exit_code_ok(db, mock_crawler):
    from predictor import cli

    code = await cli._predict_one("weekly-contest-517", force=False, limit=None)
    assert code == cli.EXIT_OK


async def test_cli_reports_failure_when_forced_recrawl_is_blocked(
    db, mock_crawler, monkeypatch
):
    """Data stays live, but the run still reports that the refresh failed."""
    from predictor import cli

    slug = "weekly-contest-517"
    assert await cli._predict_one(slug, force=False, limit=None) == cli.EXIT_OK

    async def blocked(_slug):
        raise CrawlBlockedError("403 from Cloudflare")

    monkeypatch.setattr(predict_service, "fetch_user_num", blocked)
    assert await cli._predict_one(slug, force=True, limit=None) == cli.EXIT_BLOCKED

    # A later no-op run on the still-good data reports success again.
    assert await cli._predict_one(slug, force=False, limit=None) == cli.EXIT_OK


async def test_total_records_reflects_what_was_persisted(db, mock_crawler, monkeypatch):
    """`total_records` must not claim work that never reached the database."""
    slug = "weekly-contest-517"

    # A run that dies after the crawl but before persistence must not leave a
    # record count behind — this is what left the doc claiming 39,401 records
    # while only 350 existed.
    async def boom(_rows):
        raise CrawlBlockedError("403 from Cloudflare")

    monkeypatch.setattr(predict_service, "_resolve_ratings", boom)
    contest = await predict_service.run_prediction(slug)

    assert contest.status == "error"
    assert contest.crawled_ranking == mock_crawler["n"], "rows were crawled"
    assert contest.total_records == 0, "but none were persisted"
    assert await ContestRecord.find(ContestRecord.contest_slug == slug).count() == 0
