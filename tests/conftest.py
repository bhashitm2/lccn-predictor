"""Shared fixtures: an in-memory MongoDB (mongomock) and a stubbed HTTP session."""
import pytest
import pytest_asyncio
from beanie import init_beanie
from mongomock_motor import AsyncMongoMockClient

from predictor.config import Settings
from predictor.crawler import http as http_mod
from predictor.crawler import ranking as ranking_mod
from predictor.crawler import user_rating as user_rating_mod
from predictor.db.models import ALL_DOCUMENT_MODELS


@pytest_asyncio.fixture
async def db():
    """Fresh in-memory database per test."""
    client = AsyncMongoMockClient()
    await init_beanie(
        database=client["test_lccn"], document_models=ALL_DOCUMENT_MODELS
    )
    yield client


class StubResponse:
    def __init__(self, status_code: int, payload=None, headers=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.headers = headers or {}

    def json(self):
        return self._payload


class StubSession:
    """Records every request and replies from a caller-supplied handler."""

    def __init__(self, handler):
        self.handler = handler
        self.calls: list[str] = []
        self.payloads: list[dict] = []

    async def request(self, method, url, **kwargs):
        self.calls.append(url)
        self.payloads.append(kwargs.get("json"))
        return self.handler(url, len(self.calls))

    async def close(self):
        return None


@pytest.fixture
def stub_settings(monkeypatch):
    """Fast, deterministic settings so tests don't sit in real cooldowns."""

    def _apply(**overrides):
        defaults = dict(
            http_retry=3,
            http_rate_limit_per_second=1000.0,
            http_backoff_base_seconds=0.001,
            http_backoff_cap_seconds=0.002,
            http_block_threshold=5,
            http_block_cooldown_seconds=0.01,
            http_max_cooldowns=2,
            repair_passes=1,
            # Production defaults for these are minutes long — a test that
            # forgets to shrink them silently sleeps instead of failing.
            repair_pass_cooldown_seconds=0.01,
            ranking_concurrency=4,
            min_ranking_coverage=0.98,
        )
        defaults.update(overrides)
        settings = Settings(**defaults)
        for mod in (http_mod, ranking_mod, user_rating_mod):
            monkeypatch.setattr(mod, "get_settings", lambda: settings)
        return settings

    return _apply


@pytest.fixture
def stub_session(monkeypatch):
    def _install(handler):
        session = StubSession(handler)

        async def _get_session():
            return session

        monkeypatch.setattr(http_mod, "get_session", _get_session)
        return session

    return _install
