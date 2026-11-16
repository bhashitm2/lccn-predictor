"""Normalising what a person types when looking themselves up (pure, no I/O)."""
from __future__ import annotations

import re

# leetcode.com/u/<slug>/, leetcode.com/<slug>/, leetcode.cn/u/<slug>/
_PROFILE_URL = re.compile(
    r"leetcode\.(?:com|cn)/(?:u/|profile/)?([A-Za-z0-9_\-.]+)", re.IGNORECASE
)
# Paths on leetcode.com that are not profiles.
_NOT_PROFILES = {"contest", "problems", "discuss", "problemset", "u", "profile"}


def normalize_user_query(text: str) -> str:
    """``"  @Alice "`` -> ``"Alice"``; a LeetCode profile URL -> its user slug."""
    q = (text or "").strip()
    m = _PROFILE_URL.search(q)
    if m and m.group(1).lower() not in _NOT_PROFILES:
        q = m.group(1)
    return q.lstrip("@").strip()


def exact_pattern(q: str) -> dict:
    """Case-insensitive whole-string match."""
    return {"$regex": f"^{re.escape(q)}$", "$options": "i"}


def contains_pattern(q: str) -> dict:
    """Case-insensitive substring match."""
    return {"$regex": re.escape(q), "$options": "i"}
