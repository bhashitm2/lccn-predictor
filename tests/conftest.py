"""Shared fixtures: an in-memory MongoDB (mongomock) wired into Beanie."""
import pytest_asyncio
from beanie import init_beanie
from mongomock_motor import AsyncMongoMockClient

from predictor.db.models import ALL_DOCUMENT_MODELS


@pytest_asyncio.fixture
async def db():
    """Fresh in-memory database per test."""
    client = AsyncMongoMockClient()
    await init_beanie(
        database=client["test_lccn"], document_models=ALL_DOCUMENT_MODELS
    )
    yield client
