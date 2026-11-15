"""Free-form contest input, the computed contest schedule, and the endpoints and
CLI check built on them."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi import HTTPException

from predictor import cli
from predictor.api import routes
from predictor.config import Settings
from predictor.core.contests import (
    contest_title,
    contests_started_between,
    estimated_start,
    parse_contest_query,
)
from predictor.db.models import Contest

UTC = timezone.utc


@pytest.mark.parametrize(
    "text, expected",
    [
        ("weekly-contest-528", ["weekly-contest-528"]),
        ("Weekly Contest 528", ["weekly-contest-528"]),
        ("  WEEKLY   contest #528 ", ["weekly-contest-528"]),
        ("weekly528", ["weekly-contest-528"]),
        ("weekly_contest_528", ["weekly-contest-528"]),
        ("w528", ["weekly-contest-528"]),
        ("wc 528", ["weekly-contest-528"]),
        ("LC weekly 528", ["weekly-contest-528"]),
        ("Biweekly Contest 195", ["biweekly-contest-195"]),
        ("bi-weekly 195", ["biweekly-contest-195"]),
        ("bi weekly contest 195", ["biweekly-contest-195"]),
        ("bw195", ["biweekly-contest-195"]),
        ("b 195", ["biweekly-contest-195"]),
        ("528", ["weekly-contest-528", "biweekly-contest-528"]),
        ("#195", ["weekly-contest-195", "biweekly-contest-195"]),
        (
            "https://leetcode.com/contest/weekly-contest-528/ranking/",
            ["weekly-contest-528"],
        ),
        ("leetcode.com/contest/biweekly-contest-195", ["biweekly-contest-195"]),
        ("Some Special Contest", ["some-special-contest"]),
        ("", []),
        ("!!!", []),
    ],
)
def test_parse_contest_query(text, expected):
    assert parse_contest_query(text) == expected


def test_estimated_start_matches_leetcode():
    # Real start times from LeetCode's pastContests list.
    assert estimated_start("weekly-contest-528") == datetime(2026, 11, 15, 2, 30, tzinfo=UTC)
    assert estimated_start("weekly-contest-481") == datetime(2025, 12, 21, 2, 30, tzinfo=UTC)
    assert estimated_start("biweekly-contest-172") == datetime(
        2025, 12, 20, 14, 30, tzinfo=UTC
    )
    assert estimated_start("biweekly-contest-189") == datetime(
        2026, 8, 15, 14, 30, tzinfo=UTC
    )
    assert estimated_start("some-special-contest") is None


def test_contest_title():
    assert contest_title("weekly-contest-528") == "Weekly Contest 528"
    assert contest_title("biweekly-contest-195") == "Biweekly Contest 195"


def test_contests_started_between():
    got = contests_started_between(
        datetime(2026, 11, 7, 0, 0, tzinfo=UTC), datetime(2026, 11, 15, 2, 30, tzinfo=UTC)
    )
    assert got == ["biweekly-contest-195", "weekly-contest-527", "weekly-contest-528"]


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
async def test_resolve_prefers_the_contest_with_a_prediction(db):
    await Contest(title_slug="biweekly-contest-195", status="done", total_records=5).insert()
    res = await routes.resolve(q="195")
    assert res.slug == "biweekly-contest-195" and res.status == "done"
    assert res.title == "Biweekly Contest 195"


async def test_resolve_unknown_number_defaults_to_weekly(db):
    res = await routes.resolve(q="w999")
    assert res.slug == "weekly-contest-999" and res.status == "pending"


async def test_resolve_rejects_gibberish(db):
    with pytest.raises(HTTPException) as e:
        await routes.resolve(q="???")
    assert e.value.status_code == 400


async def test_predict_accepts_free_form_slug(db, monkeypatch):
    monkeypatch.setattr(
        routes, "get_settings", lambda: Settings(public_crawl_enabled=False)
    )
    await Contest(title_slug="weekly-contest-528", status="crawling").insert()

    class Resp:
        status_code = 200

    page = await routes.predict_contest(
        "Weekly Contest 528", Resp(), page=1, size=10, sort="rank", limit=None
    )
    assert page.slug == "weekly-contest-528" and page.status == "crawling"


async def test_recent_contests_newest_first_with_status(db, monkeypatch):
    # Monday after weekly-contest-528.
    monkeypatch.setattr(
        routes, "utcnow", lambda: datetime(2026, 11, 16, 12, 0, tzinfo=UTC)
    )
    await Contest(
        title_slug="weekly-contest-527", status="done", total_records=23048
    ).insert()
    await Contest(title_slug="weekly-contest-528", status="error").insert()
    got = await routes.recent_contests(n=3)
    assert [c.slug for c in got] == [
        "weekly-contest-528",
        "weekly-contest-527",
        "biweekly-contest-195",
    ]
    assert [c.status for c in got] == ["error", "done", "pending"]
    assert got[1].total_records == 23048


async def test_recent_contests_skip_one_still_running(db, monkeypatch):
    # 30 minutes into weekly-contest-529: it hasn't finished, so it isn't listed.
    monkeypatch.setattr(
        routes, "utcnow", lambda: datetime(2026, 11, 22, 3, 0, tzinfo=UTC)
    )
    got = await routes.recent_contests(n=2)
    assert [c.slug for c in got] == ["biweekly-contest-196", "weekly-contest-528"]


# --------------------------------------------------------------------------- #
# CLI: the overdue check that decides whether a fully blocked slot alerts
# --------------------------------------------------------------------------- #
async def test_overdue_lists_missing_contests_past_the_grace_period(db):
    now = datetime(2026, 11, 16, 16, 0, tzinfo=UTC)  # weekly-528 started 37.5h ago
    assert await cli._overdue_slugs(36, now) == ["weekly-contest-528"]
    await Contest(
        title_slug="weekly-contest-528", status="done", total_records=1
    ).insert()
    assert await cli._overdue_slugs(36, now) == []


async def test_overdue_not_raised_within_grace_period(db):
    now = datetime(2026, 11, 15, 12, 0, tzinfo=UTC)  # 9.5h after weekly-528 started
    assert await cli._overdue_slugs(36, now) == []


async def test_overdue_ignores_contests_past_the_catch_up_window(db):
    now = datetime(2026, 11, 18, 12, 0, tzinfo=UTC)  # weekly-528 is 81.5h old
    assert await cli._overdue_slugs(36, now) == []
