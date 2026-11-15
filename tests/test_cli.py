"""``predict-recent``: which contests each scheduled slot acts on.

Every slot must catch up on contests an earlier slot was blocked from, refresh
only the contest kind it is scheduled for, and never redo a refresh that an
earlier fresh-runner attempt of the same run already completed.
"""
from __future__ import annotations

from datetime import timedelta

import pytest

from predictor import cli
from predictor.crawler.http import CrawlBlockedError
from predictor.db.models import Contest, utcnow

pytestmark = pytest.mark.asyncio


def _ago(hours: float) -> int:
    return int((utcnow() - timedelta(hours=hours)).timestamp())


@pytest.fixture
def recent(monkeypatch):
    """Install a fake contest listing and record every _predict_one call."""
    calls: list[tuple[str, bool]] = []
    codes: dict[str, int] = {}

    def _install(contests):
        async def fake_list(_page):
            return contests

        async def fake_predict(slug, force, limit):
            calls.append((slug, force))
            return codes.get(slug, cli.EXIT_OK)

        monkeypatch.setattr(cli, "fetch_past_contests", fake_list)
        monkeypatch.setattr(cli, "_predict_one", fake_predict)
        return calls, codes

    return _install


async def _done(slug: str, predicted_hours_ago: float) -> None:
    await Contest(
        title_slug=slug,
        status="done",
        total_records=100,
        predicted_at=utcnow() - timedelta(hours=predicted_hours_ago),
    ).insert()


async def test_pending_contests_are_predicted_oldest_first(db, recent):
    calls, _ = recent(
        [
            ("Weekly 523", "weekly-contest-523", _ago(3)),
            ("Biweekly 193", "biweekly-contest-193", _ago(15)),
            ("Weekly 522", "weekly-contest-522", _ago(7 * 24)),  # too old
        ]
    )

    code = await cli._predict_recent(refresh=None, limit=None)

    assert code == cli.EXIT_OK
    assert calls == [("biweekly-contest-193", False), ("weekly-contest-523", False)]


async def test_done_contests_are_skipped_without_refresh(db, recent):
    await _done("weekly-contest-523", predicted_hours_ago=20)
    calls, _ = recent([("Weekly 523", "weekly-contest-523", _ago(26))])

    code = await cli._predict_recent(refresh=None, limit=None)

    assert code == cli.EXIT_OK
    assert calls == []


async def test_errored_contest_is_retried(db, recent):
    await Contest(title_slug="biweekly-contest-193", status="error").insert()
    calls, _ = recent([("Biweekly 193", "biweekly-contest-193", _ago(12))])

    await cli._predict_recent(refresh=None, limit=None)

    assert calls == [("biweekly-contest-193", False)]


async def test_refresh_targets_only_its_contest_kind(db, recent):
    """The biweekly refresh slot runs after a newer weekly; it must still pick
    the biweekly, not whatever contest finished last."""
    await _done("biweekly-contest-193", predicted_hours_ago=22)
    await _done("weekly-contest-523", predicted_hours_ago=10)
    calls, _ = recent(
        [
            ("Weekly 523", "weekly-contest-523", _ago(13)),
            ("Biweekly 193", "biweekly-contest-193", _ago(25)),
        ]
    )

    await cli._predict_recent(refresh="biweekly", limit=None)

    assert calls == [("biweekly-contest-193", True)]


async def test_fresh_prediction_is_not_refreshed_again(db, recent):
    """A retry on a new runner must not redo a refresh the last attempt did."""
    await _done("weekly-contest-523", predicted_hours_ago=0.2)
    calls, _ = recent([("Weekly 523", "weekly-contest-523", _ago(26))])

    code = await cli._predict_recent(refresh="weekly", limit=None)

    assert code == cli.EXIT_OK
    assert calls == []


async def test_any_blocked_contest_makes_the_run_retryable(db, recent):
    calls, codes = recent(
        [
            ("Weekly 523", "weekly-contest-523", _ago(3)),
            ("Biweekly 193", "biweekly-contest-193", _ago(15)),
        ]
    )
    codes["biweekly-contest-193"] = cli.EXIT_FAILED
    codes["weekly-contest-523"] = cli.EXIT_BLOCKED

    assert await cli._predict_recent(refresh=None, limit=None) == cli.EXIT_BLOCKED
    assert len(calls) == 2


async def test_hard_failure_without_block_fails(db, recent):
    _, codes = recent([("Weekly 523", "weekly-contest-523", _ago(3))])
    codes["weekly-contest-523"] = cli.EXIT_FAILED

    assert await cli._predict_recent(refresh=None, limit=None) == cli.EXIT_FAILED


async def test_blocked_contest_listing_is_retryable(db, monkeypatch):
    async def blocked(_page):
        raise CrawlBlockedError("403 from Cloudflare")

    monkeypatch.setattr(cli, "fetch_past_contests", blocked)

    assert await cli._predict_recent(refresh=None, limit=None) == cli.EXIT_BLOCKED
