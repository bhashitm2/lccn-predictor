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
    http_retry: int = 6
    http_timeout_seconds: float = 30.0
    rating_cache_ttl_hours: int = 12

    # Which browser TLS fingerprint curl_cffi impersonates. "chrome" tracks the
    # newest supported target and is what full crawls have been succeeding with,
    # so it stays the default; this is exposed only as an escape hatch (e.g.
    # LCCN_IMPERSONATE=chrome136) if that target ever starts collecting 403s.
    # Note the observed blocks correlate with request *volume*, not fingerprint:
    # the same target completes full crawls when the crawl is paced.
    impersonate: str = "chrome"

    # --- Politeness / anti-block -------------------------------------------
    # Requests per second across the WHOLE batch. Concurrency alone doesn't
    # bound this: when Cloudflare 403s from the edge in ~20ms, 8 concurrent
    # slots turn into hundreds of requests a second, which is what gets the
    # runner IP blocked in the first place.
    http_rate_limit_per_second: float = 4.0
    http_backoff_base_seconds: float = 1.0
    http_backoff_cap_seconds: float = 30.0
    # Consecutive blocked (403/429) responses before the breaker pauses the
    # batch, how long it pauses, and how many fruitless pauses before we give up.
    # A Cloudflare IP-reputation block does not clear in a minute; short
    # cooldowns just spend the breaker's budget without waiting anything out.
    http_block_threshold: int = 12
    http_block_cooldown_seconds: float = 300.0
    http_max_cooldowns: int = 3

    # --- Crawl completeness -------------------------------------------------
    # Extra slow passes over just the pages that failed, before we call the
    # crawl incomplete. These resume — each pass keeps every page already
    # fetched — which is why patience belongs here rather than in a workflow
    # retry that would restart the whole crawl from page 1.
    repair_passes: int = 2
    # Wait between repair passes. Distinct from the breaker's cooldown: that
    # one paces requests inside a pass, this one waits out a block between them.
    repair_pass_cooldown_seconds: float = 600.0
    # Minimum fraction of expected ranking rows required to accept a crawl.
    # Below this the prediction is refused rather than persisted: the Elo/FFT
    # engine needs the whole field, and persisting overwrites the previous
    # (good) prediction for the contest.
    min_ranking_coverage: float = 0.98
    # Max fraction of GraphQL rating batches allowed to fail before we refuse
    # the prediction. Unresolved users fall back to (1500, 0), which is fine for
    # a few stragglers and fabricated input for a large slice of the field.
    max_failed_rating_batches: float = 0.02

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
