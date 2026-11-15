"""Contest naming and schedule helpers (pure functions, no I/O).

* ``parse_contest_query`` turns whatever a person types — "Weekly Contest 528",
  "w528", "bw 170", "528", a leetcode.com contest URL — into candidate slugs.
* ``estimated_start`` gives a weekly/biweekly contest's start time from its
  number. LeetCode has held both on a fixed UTC cadence for years (weekly every
  Sunday 02:30, biweekly every other Saturday 14:30), so this needs no request
  to LeetCode — useful because the web host can't reach LeetCode at all.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

# Verified against LeetCode's pastContests list (2025-12 .. 2026-11).
_ANCHORS = {
    "weekly": (528, datetime(2026, 11, 15, 2, 30, tzinfo=timezone.utc), 7),
    "biweekly": (195, datetime(2026, 11, 7, 14, 30, tzinfo=timezone.utc), 14),
}
CONTEST_DURATION = timedelta(minutes=90)

_SLUG_RE = re.compile(r"^(weekly|biweekly)-contest-(\d+)$")
_VALID_SLUG = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_URL_RE = re.compile(r"leetcode\.(?:com|cn)/contest/([a-z0-9-]+)")

# Applied after lower-casing and collapsing spaces/underscores/hyphens to " ".
_PREFIX = r"^(?:leetcode |lc )?"
_BIWEEKLY = re.compile(_PREFIX + r"(?:bi ?weekly|bwc|bw|b)(?: ?contest)? ?(?:no\.? ?|#)?(\d+)$")
_WEEKLY = re.compile(_PREFIX + r"(?:weekly|wc|w)(?: ?contest)? ?(?:no\.? ?|#)?(\d+)$")
_NUMBER = re.compile(_PREFIX + r"(?:contest )?(?:no\.? ?|#)?(\d+)$")


def parse_slug(slug: str) -> Optional[Tuple[str, int]]:
    """``"weekly-contest-528"`` -> ``("weekly", 528)``; None for other slugs."""
    m = _SLUG_RE.match(slug)
    return (m.group(1), int(m.group(2))) if m else None


def parse_contest_query(text: str) -> List[str]:
    """Candidate slugs for free-form input, most likely first.

    A bare number is ambiguous, so it yields both the weekly and the biweekly
    slug; the caller picks whichever actually has a prediction.
    """
    q = (text or "").strip().lower()
    if not q:
        return []
    url = _URL_RE.search(q)
    if url:
        q = url.group(1)
    if _SLUG_RE.match(q):
        return [q]
    norm = re.sub(r"[\s_\-]+", " ", q).strip()
    # "#528" / "no. 528" written with no space before the digits
    norm = re.sub(r"([a-z])#", r"\1 #", norm)
    for pattern, kinds in (
        (_BIWEEKLY, ("biweekly",)),
        (_WEEKLY, ("weekly",)),
        (_NUMBER, ("weekly", "biweekly")),
    ):
        m = pattern.match(norm)
        if m:
            n = int(m.group(1))
            return [f"{kind}-contest-{n}" for kind in kinds]
    # Anything else (e.g. a special contest's slug): accept it in slug form.
    slug = re.sub(r"[^a-z0-9]+", "-", q).strip("-")
    return [slug] if _VALID_SLUG.match(slug) else []


def estimated_start(slug: str) -> Optional[datetime]:
    """Start time (UTC) of a weekly/biweekly contest, or None for other slugs."""
    parsed = parse_slug(slug)
    if parsed is None:
        return None
    kind, n = parsed
    anchor_n, anchor_at, every_days = _ANCHORS[kind]
    return anchor_at + timedelta(days=every_days * (n - anchor_n))


def contest_title(slug: str) -> str:
    parsed = parse_slug(slug)
    if parsed is None:
        return slug.replace("-", " ").title()
    kind, n = parsed
    return f"{kind.capitalize()} Contest {n}"


def contests_started_between(start: datetime, end: datetime) -> List[str]:
    """Slugs of weekly/biweekly contests whose start falls in ``[start, end]``."""
    out: List[Tuple[datetime, str]] = []
    for kind, (anchor_n, anchor_at, every_days) in _ANCHORS.items():
        period = timedelta(days=every_days)
        # first contest number at or after `start`
        n = anchor_n + -((anchor_at - start) // period)
        while True:
            at = anchor_at + period * (n - anchor_n)
            if at > end:
                break
            if at >= start:
                out.append((at, f"{kind}-contest-{n}"))
            n += 1
    return [slug for _, slug in sorted(out)]
