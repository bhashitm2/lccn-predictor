"""Diagnose a Cloudflare/LeetCode 403: is it this IP, or this contest?

Sends exactly one request per (slug, endpoint) from one process, a few seconds
apart, with the crawler's own session and headers — no retries, no database.
For each it prints the status and the response headers/body that tell a
Cloudflare edge refusal apart from an answer from LeetCode itself.

Reading the output:
  * every slug 403s                      -> this IP (or range) is refused
  * weekly 200, biweekly 403             -> the contest is refused, not the IP
  * ``cf-mitigated: challenge`` / HTML   -> Cloudflare's bot check
  * JSON body with a 403                 -> LeetCode's own application answer

Usage: ``python -m predictor.probe weekly-contest-523 biweekly-contest-193``
"""
from __future__ import annotations

import argparse
import asyncio
import re

from predictor.crawler.http import close_http_session, get_session
from predictor.crawler.ranking import _base_url, _page_headers

# Response headers that say who answered and why.
_TELL_HEADERS = ("server", "cf-ray", "cf-mitigated", "content-type", "retry-after")
_BODY_CHARS = 300


def _endpoints(slug: str) -> list[tuple[str, str]]:
    base = _base_url()
    return [
        ("info", f"{base}/contest/api/info/{slug}/"),
        ("ranking meta", f"{base}/contest/api/ranking/{slug}/"),
        ("ranking p1", f"{base}/contest/api/ranking/{slug}/?pagination=1&region=global"),
    ]


def summarize_body(text: str, limit: int = _BODY_CHARS) -> str:
    """Collapse a response body to one short line: the <title> for HTML, else the text."""
    title = re.search(r"<title[^>]*>(.*?)</title>", text, re.IGNORECASE | re.DOTALL)
    if title:
        return f"HTML <title>{' '.join(title.group(1).split())}</title>"
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[:limit] + "…"


async def probe(slugs: list[str], gap: float) -> int:
    session = await get_session()
    refused = 0
    first = True
    try:
        for slug in slugs:
            for name, url in _endpoints(slug):
                if not first:
                    await asyncio.sleep(gap)
                first = False
                try:
                    resp = await session.get(url, headers=_page_headers(slug), timeout=20)
                except Exception as exc:  # noqa: BLE001 — report and carry on
                    print(f"{slug:<24} {name:<13} ERROR {type(exc).__name__}: {exc}")
                    continue
                if resp.status_code == 403:
                    refused += 1
                tells = ", ".join(
                    f"{h}={resp.headers.get(h)}" for h in _TELL_HEADERS if resp.headers.get(h)
                )
                print(f"{slug:<24} {name:<13} HTTP {resp.status_code}  [{tells}]")
                print(f"{'':<38} body: {summarize_body(resp.text or '')}")
    finally:
        await close_http_session()
    print(f"\n{refused} refused (403) responses")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("slugs", nargs="+", help="contest slugs, probed in this order")
    parser.add_argument("--gap", type=float, default=3.0, help="seconds between requests")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(probe(args.slugs, args.gap)))


if __name__ == "__main__":
    main()
