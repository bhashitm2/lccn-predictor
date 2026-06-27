"""MongoDB connection + Beanie initialisation."""
from motor.motor_asyncio import AsyncIOMotorClient

from predictor.config import get_settings
from predictor.db.models import ALL_DOCUMENT_MODELS

_client: AsyncIOMotorClient | None = None


async def init_db() -> None:
    """Connect to MongoDB and initialise Beanie ODM. Idempotent."""
    global _client
    from beanie import init_beanie

    settings = get_settings()
    _client = AsyncIOMotorClient(settings.mongodb_uri)
    db = _client[settings.db_name]
    await _drop_stale_indexes(db)
    await init_beanie(database=db, document_models=ALL_DOCUMENT_MODELS)


async def _drop_stale_indexes(db) -> None:
    """Drop indexes from earlier schema versions that are now wrong.

    The old unique index on (contest_slug, username) is invalid — display
    usernames aren't unique. Must be removed before the new (contest_slug,
    user_slug) index is created, or inserts fail with E11000 on shared names.
    """
    from loguru import logger

    try:
        existing = await db["contest_record"].index_information()
        if "contest_slug_1_username_1" in existing:
            await db["contest_record"].drop_index("contest_slug_1_username_1")
            logger.info("dropped stale index contest_slug_1_username_1")
    except Exception as exc:  # collection may not exist yet — that's fine
        logger.debug(f"stale-index cleanup skipped: {exc!r}")


async def close_db() -> None:
    global _client
    if _client is not None:
        _client.close()
        _client = None
