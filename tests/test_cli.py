import predictor.cli as cli


async def test_predict_latest_falls_back_to_recent_slug(monkeypatch):
    calls = []

    async def fake_init_db():
        return None

    async def fake_close_db():
        return None

    async def fake_latest():
        return "weekly-contest-514"

    async def fake_past_contests(page_no=1):
        assert page_no == 1
        return [
            ("Weekly Contest 514", "weekly-contest-514", 0),
            ("Biweekly Contest 164", "biweekly-contest-164", 0),
        ]

    async def fake_predict_one(slug, force, limit):
        calls.append((slug, force, limit))
        return slug == "biweekly-contest-164"

    monkeypatch.setattr(cli, "init_db", fake_init_db)
    monkeypatch.setattr(cli, "close_db", fake_close_db)
    monkeypatch.setattr(cli, "fetch_latest_contest_slug", fake_latest)
    monkeypatch.setattr(cli, "fetch_past_contests", fake_past_contests)
    monkeypatch.setattr(cli, "_predict_one", fake_predict_one)

    rc = await cli._run("latest", force=True, limit=100)

    assert rc == 0
    assert calls == [
        ("weekly-contest-514", True, 100),
        ("biweekly-contest-164", False, 100),
    ]
