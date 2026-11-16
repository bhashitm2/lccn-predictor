"""Finding a participant anywhere in a contest, not just on the visible page."""
from __future__ import annotations

import pytest
from fastapi import HTTPException

from predictor.api import routes
from predictor.config import Settings
from predictor.core.users import normalize_user_query
from predictor.db.models import Contest, ContestRecord

SLUG = "weekly-contest-528"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("alice", "alice"),
        ("  @Alice  ", "Alice"),
        ("https://leetcode.com/u/lakshyakeshwani/", "lakshyakeshwani"),
        ("leetcode.com/u/neal_wu", "neal_wu"),
        ("https://leetcode.com/lakshyakeshwani/", "lakshyakeshwani"),
        ("https://leetcode.cn/u/some-cn-user/", "some-cn-user"),
        ("https://leetcode.com/contest/weekly-contest-528/", "https://leetcode.com/contest/weekly-contest-528/"),
        ("", ""),
    ],
)
def test_normalize_user_query(text, expected):
    assert normalize_user_query(text) == expected


@pytest.fixture
async def contest(db, monkeypatch):
    """120 participants; ranks 1..120 except that #7 and #8 tie at rank 7, and
    two different people share the display name "Alex"."""
    monkeypatch.setattr(routes, "get_settings", lambda: Settings(public_crawl_enabled=False))
    await Contest(title_slug=SLUG, status="done", total_records=120).insert()
    docs = []
    for i in range(1, 121):
        username, user_slug = f"User{i}", f"user{i}"
        if i == 60:
            username, user_slug = "Alex", "alex-a"
        if i == 90:
            username, user_slug = "Alex", "alex-b"
        if i == 100:
            username, user_slug = "Lakshya Keshwani", "lakshyakeshwani"
        docs.append(
            ContestRecord(
                contest_slug=SLUG,
                username=username,
                user_slug=user_slug,
                data_region="US",
                rank=7 if i == 8 else i,
                old_rating=1500.0,
                delta_rating=1.0,
                new_rating=1501.0,
            )
        )
    await ContestRecord.insert_many(docs)


class _Resp:
    status_code = 200


async def _page_slugs(page: int, size: int) -> list[str]:
    data = await routes.predict_contest(
        SLUG, _Resp(), page=page, size=size, sort="rank", limit=None
    )
    return [r.user_slug for r in data.records]


async def test_user_far_down_the_ranking_is_found_with_its_page(contest):
    m = await routes.predict_user(SLUG, "lakshyakeshwani", size=50)
    assert m.user_slug == "lakshyakeshwani" and m.rank == 100
    assert m.position == 100 and m.page == 2
    assert "lakshyakeshwani" in await _page_slugs(m.page, 50)


@pytest.mark.parametrize(
    "typed",
    [
        "LakshyaKeshwani",
        "lakshya keshwani",  # display name, any case
        "@lakshyakeshwani",
        "https://leetcode.com/u/lakshyakeshwani/",
    ],
)
async def test_lookup_is_forgiving(contest, typed):
    m = await routes.predict_user(SLUG, typed, size=50)
    assert m.user_slug == "lakshyakeshwani"


async def test_computed_page_matches_predict_ordering_for_every_user(contest):
    # Including the tie at rank 7 and a page boundary between the tied users.
    for size in (7, 50):
        for slug in ("user7", "user8", "user1", "alex-a", "user120"):
            m = await routes.predict_user(SLUG, slug, size=size)
            assert slug in await _page_slugs(m.page, size), (slug, size, m.page)


async def test_unknown_user_is_404(contest):
    with pytest.raises(HTTPException) as e:
        await routes.predict_user(SLUG, "nobody-here", size=50)
    assert e.value.status_code == 404


async def test_search_returns_everyone_sharing_a_display_name(contest):
    res = await routes.search_users(SLUG, q="alex", size=50, limit=10)
    assert res.exact is True
    assert [m.user_slug for m in res.matches] == ["alex-a", "alex-b"]
    assert [m.page for m in res.matches] == [2, 2]


async def test_search_prefers_the_exact_handle(contest):
    res = await routes.search_users(SLUG, q="alex-b", size=50, limit=10)
    assert res.exact is True and res.matches[0].user_slug == "alex-b"


async def test_search_suggests_partial_matches(contest):
    res = await routes.search_users(SLUG, q="keshw", size=50, limit=10)
    assert res.exact is False
    assert [m.user_slug for m in res.matches] == ["lakshyakeshwani"]


async def test_search_treats_regex_characters_literally(contest):
    res = await routes.search_users(SLUG, q=".*", size=50, limit=10)
    assert res.matches == []


async def test_search_on_contest_without_prediction_is_404(db):
    with pytest.raises(HTTPException) as e:
        await routes.search_users("weekly-contest-999", q="x", size=50, limit=10)
    assert e.value.status_code == 404
