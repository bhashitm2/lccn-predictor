"""Beanie (MongoDB) document models.

Three collections:

* ``Contest`` — one doc per contest slug, tracks crawl/predict status.
* ``ContestRecord`` — one doc per participant per contest, holds the prediction.
* ``UserRatingCache`` — latest known rating per user; the cache that makes
  crawl-by-slug practical so we don't re-query every user via GraphQL each time.
"""
from datetime import datetime, timezone
from typing import List, Literal, Optional

from beanie import Document
from pydantic import Field
from pymongo import ASCENDING, IndexModel

DATA_REGION = Literal["US", "CN"]
PredictStatus = Literal[
    "pending", "crawling", "resolving", "predicting", "done", "error"
]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Contest(Document):
    title_slug: str
    title: Optional[str] = None
    user_num: Optional[int] = None
    status: PredictStatus = "pending"
    error: Optional[str] = None
    # progress counters (so /status can report % crawled)
    crawled_ranking: int = 0
    resolved_ratings: int = 0
    total_records: int = 0
    limit: Optional[int] = None  # if a top-N prediction was requested
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    predicted_at: Optional[datetime] = None

    class Settings:
        name = "contest"
        indexes = [
            IndexModel("title_slug", unique=True),
            "status",
        ]


class ContestRecord(Document):
    contest_slug: str
    username: str
    user_slug: str
    data_region: DATA_REGION
    rank: int
    score: int = 0
    finish_time: Optional[datetime] = None
    # inputs to the prediction
    old_rating: Optional[float] = None
    attended_count: Optional[int] = None
    # outputs
    delta_rating: Optional[float] = None
    new_rating: Optional[float] = None

    class Settings:
        name = "contest_record"
        indexes = [
            # user_slug is the unique LeetCode handle; display `username` is NOT
            # unique (two different users can share a display name, e.g. "Alex").
            IndexModel(
                [("contest_slug", ASCENDING), ("user_slug", ASCENDING)], unique=True
            ),
            IndexModel([("contest_slug", ASCENDING), ("rank", ASCENDING)]),
            "username",  # non-unique, for lookups by display name
        ]


class UserRatingCache(Document):
    username: str
    user_slug: str
    data_region: DATA_REGION
    rating: float
    attended_count: int
    updated_at: datetime = Field(default_factory=utcnow)

    class Settings:
        name = "user_rating_cache"
        indexes = [
            IndexModel(
                [("user_slug", ASCENDING), ("data_region", ASCENDING)], unique=True
            ),
            "username",
        ]


ALL_DOCUMENT_MODELS: List[type[Document]] = [Contest, ContestRecord, UserRatingCache]
