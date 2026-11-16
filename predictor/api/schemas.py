"""API response models (what your other website consumes)."""
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel


class PredictionRecord(BaseModel):
    rank: int
    username: str
    user_slug: str
    data_region: str
    old_rating: Optional[float]
    delta_rating: Optional[float]
    new_rating: Optional[float]
    attended_count: Optional[int]
    score: int
    finish_time: Optional[datetime]


class ContestStatus(BaseModel):
    slug: str
    status: str  # pending | crawling | predicting | done | error
    user_num: Optional[int] = None
    crawled_ranking: int = 0
    resolved_ratings: int = 0
    total_records: int = 0
    limit: Optional[int] = None
    error: Optional[str] = None
    predicted_at: Optional[datetime] = None


class PredictionPage(BaseModel):
    slug: str
    status: str
    total: int
    page: int
    size: int
    records: List[PredictionRecord]


class ResolvedContest(BaseModel):
    query: str
    slug: str
    title: str
    status: str  # status of the resolved slug: pending (never queued) | ... | done
    candidates: List[str]


class RecentContest(BaseModel):
    slug: str
    title: str
    start_time: Optional[datetime] = None
    status: str
    total_records: int = 0
    predicted_at: Optional[datetime] = None


class UserMatch(PredictionRecord):
    position: int  # 1-based position in rank order
    page: int  # the /predict page (rank order, for the requested size) holding this user


class UserSearch(BaseModel):
    slug: str
    query: str
    exact: bool  # True: matches are exact (case-insensitive); False: suggestions
    size: int
    matches: List[UserMatch]
