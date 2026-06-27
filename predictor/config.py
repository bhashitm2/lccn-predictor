"""Centralised settings, loaded from environment variables (prefix ``LCCN_``).

Example: ``LCCN_MONGODB_URI`` populates ``Settings.mongodb_uri``.
A local ``.env`` file is read automatically (see ``.env.example``).
"""
from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LCCN_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Database
    mongodb_uri: str = "mongodb://localhost:27017"
    db_name: str = "lccn_predictor"

    # LeetCode endpoints
    leetcode_base_us: str = "https://leetcode.com"
    leetcode_base_cn: str = "https://leetcode.cn"

    # Crawler tuning
    ranking_concurrency: int = 8
    rating_concurrency: int = 8
    # How many users to resolve per GraphQL request (aliased batch). Each request
    # asks for many users at once, cutting tens of thousands of requests to a few
    # hundred. ~40 is safe; higher risks GraphQL query-complexity limits.
    rating_batch_size: int = 40
    http_retry: int = 10
    http_timeout_seconds: float = 30.0
    rating_cache_ttl_hours: int = 12

    # API
    cors_origins: str = "*"
    # If set, crawl-triggering endpoints (admin predict, refresh) require this key
    # via the `X-API-Key` header. Leave empty only for local dev.
    api_key: str = ""
    # When False (default), the public /predict endpoint serves cached results
    # only and never starts a live LeetCode crawl (prevents abuse / IP bans).
    # Crawls are then triggered solely by the scheduler or the admin endpoint.
    public_crawl_enabled: bool = False
    # Simple per-IP rate limit for public reads (requests per minute). 0 disables.
    rate_limit_per_minute: int = 120

    # Scheduler (rating-cache warm-up + optional auto-predict of latest contest)
    scheduler_enabled: bool = False
    scheduler_interval_minutes: int = 360
    # When True, the scheduler also auto-predicts the most recently finished
    # contest (best for always-on/paid hosting; on free tiers use the GitHub
    # Actions cron + admin endpoint instead).
    auto_predict_enabled: bool = False
    auto_predict_interval_minutes: int = 30

    @property
    def cors_origin_list(self) -> List[str]:
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
