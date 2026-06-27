# LeetCode Contest Rating Predictor

A self-hosted microservice that predicts **LeetCode weekly/biweekly contest rating
changes** from a contest slug, and exposes them over a simple REST API your other
website can call.

It is a modern, standalone re-build of the prediction engine from
[`baoliay2008/lccn_predictor`](https://github.com/baoliay2008/lccn_predictor) (now
retired). That project's successor, **EntrantHub**, is only a contest *aggregator*
and contains no reusable predictor — so the algorithm here is ported faithfully from
lccn_predictor's `app/core/elo.py` + `app/core/fft.py`.

## How it works

Given a contest slug (e.g. `weekly-contest-450`) the service:

1. Crawls the public ranking API (`/contest/api/ranking/{slug}/`, 25 rows/page).
2. Resolves each participant's **pre-contest rating + attended-contest count** —
   reusing a MongoDB **rating cache** where fresh, else LeetCode's GraphQL
   `userContestRanking` (newcomers default to rating `1500`, attended `0`).
3. Runs LeetCode's Elo algorithm (FFT-accelerated, `O(n log n)`, <0.25 s for ~30k
   players) to compute each participant's **delta** and **new rating**.
4. Persists the results and serves them via REST.

### The algorithm

* Expected win rate of A vs B: `1 / (1 + 10^((R_B − R_A) / 400))`
* Expected rank of *i*: `0.5 + Σ_j E(R_j beats R_i)`
* Mean rank: `√(expected_rank × actual_rank)`
* Expected rating: binary-search the rating whose expected rank == mean rank
* Delta: `(expected_rating − rating) × coef(k)`, where `coef = 1 / (1 + Σ_{i=1..k} (5/7)^i)`
  shrinks toward `2/9` for veterans (they move less per contest).

The fast path (`predictor/core/fft.py`) bins ratings into a histogram and convolves
with a logistic kernel via SciPy `fftconvolve`. `tests/test_elo.py` proves it matches
the naive `O(n²)` oracle (`predictor/core/elo.py`) within `0.05` per participant.

## Quick start (Docker — recommended)

```bash
cp .env.example .env          # optional: tweak settings
docker compose up --build     # starts MongoDB + the API on :8000
```

* **Demo UI:** http://localhost:8000/
* **Swagger docs:** http://localhost:8000/docs
* **Predict:** http://localhost:8000/api/v1/contest/weekly-contest-450/predict

## Quick start (local, without Docker)

Requires a running MongoDB (set `LCCN_MONGODB_URI`).

```bash
python -m venv .venv && . .venv/Scripts/activate   # (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
uvicorn predictor.main:app --reload
```

## API

**Public (read-only, rate-limited):**

| Method | Path | Purpose |
| ------ | ---- | ------- |
| GET  | `/api/v1/contest/{slug}/predict` | **Main endpoint.** Paginated predicted records. Serves cached results; `202` if not ready yet. |
| GET  | `/api/v1/contest/{slug}/user/{username}` | One participant's prediction. |
| GET  | `/api/v1/contest/{slug}/status`  | Crawl/predict progress. |
| GET  | `/healthz` | Liveness. |

**Admin (require `X-API-Key: <LCCN_API_KEY>`):** trigger crawls — used by the scheduler / cron.

| Method | Path | Purpose |
| ------ | ---- | ------- |
| POST | `/api/v1/admin/contest/{slug}/predict` | Crawl + predict a specific slug (`?force=`, `?limit=`). |
| POST | `/api/v1/admin/predict-latest` | Resolve the most recently finished contest and predict it. |

**Query params for `/predict`:** `page` (default 1), `size` (1–500, default 50),
`sort` (`rank`\|`delta`\|`new_rating`), `limit` (top-N only).

> By default (`LCCN_PUBLIC_CRAWL_ENABLED=false`) the public `/predict` only serves
> cached data and never starts a crawl — crawling is triggered by the scheduler or
> the admin endpoint. Set it `true` for local/testing convenience.

### Calling it from your website

```js
async function getPrediction(slug) {
  // first call may return 202 (job started) — poll status, then re-fetch
  let res = await fetch(`https://your-host/api/v1/contest/${slug}/predict?size=100`);
  while (res.status === 202) {
    await new Promise(r => setTimeout(r, 2000));
    res = await fetch(`https://your-host/api/v1/contest/${slug}/predict?size=100`);
  }
  return res.json(); // { slug, status, total, page, size, records: [...] }
}
```

A record looks like:

```json
{
  "rank": 1, "username": "neal_wu", "user_slug": "neal_wu", "data_region": "US",
  "old_rating": 3600.1, "delta_rating": 24.7, "new_rating": 3624.8,
  "attended_count": 80, "score": 18, "finish_time": "2026-06-21T04:12:33Z"
}
```

CORS is enabled (`LCCN_CORS_ORIGINS`, `*` by default) so a browser frontend on a
different origin can call the API directly.

## Important notes

* **Timing.** `userContestRanking.rating` is the user's *current* rating, which equals
  the pre-contest rating only **until LeetCode publishes the new ratings** (a few hours
  after a contest). Predict within that window. For already-rated past contests the true
  pre-contest ratings are no longer available from the API.
* **Performance / cold start.** Ratings are resolved in **batched GraphQL requests**
  (`LCCN_RATING_BATCH_SIZE` users per request via aliases), so a full ~35k-user contest
  takes **~2–4 min** (mostly the ranking crawl) instead of ~30+. The `UserRatingCache`
  then makes re-runs near-instant (ratings served from cache in seconds), and the optional
  warm-up scheduler (`LCCN_SCHEDULER_ENABLED=true`) pre-warms it between contests. Use
  `?limit=` to predict the top-N in a few seconds. Tune `LCCN_RANKING_CONCURRENCY` /
  `LCCN_RATING_CONCURRENCY` if you need it faster (watch for rate limits).
* **Respect LeetCode.** Conservative concurrency and a realistic User-Agent are used by
  default. This is intended for personal/educational use.

## Deploy (free tier → paid later)

A free-tier setup that costs nothing and scales to paid when you grow:

1. **Database — MongoDB Atlas (free M0).** Create a cluster, allow network access,
   copy the connection string.
2. **API — Render (free web service).** Push this repo to GitHub, then in Render:
   *New → Blueprint* and pick the repo ([`render.yaml`](render.yaml) is included).
   Set these in the dashboard:
   - `LCCN_MONGODB_URI` = your Atlas string
   - `LCCN_API_KEY` = a secret (`openssl rand -hex 24`)
   - `LCCN_CORS_ORIGINS` = your website's origin, e.g. `https://your-site.com`
3. **Scheduler — GitHub Actions cron (free).** Free hosts sleep when idle, so an
   in-process scheduler won't fire reliably. Instead
   [`.github/workflows/predict-cron.yml`](.github/workflows/predict-cron.yml) runs
   after each contest, wakes the service, and calls the admin endpoint. Add repo
   secrets `PREDICTOR_URL` (your Render URL) and `PREDICTOR_API_KEY` (= `LCCN_API_KEY`).

That's it — your website calls `GET /api/v1/contest/{slug}/predict` and gets cached
predictions; the cron keeps them fresh.

**Moving to paid / always-on** (Render paid, Fly.io, a VPS, etc.): set
`LCCN_AUTO_PREDICT_ENABLED=true` to use the built-in scheduler instead of (or
alongside) the GitHub cron, and optionally `LCCN_SCHEDULER_ENABLED=true` to keep the
rating cache warm. Put a load balancer in front and run 2–3 replicas for read traffic.

### Will it scale?

* **Serving predictions** (your site's users reading results) scales horizontally —
  results are cached in MongoDB; add replicas behind a load balancer.
* **Generating predictions** is bounded by **LeetCode**, not your infra: only ~2
  contests/week, each predicted once. Keep crawl concurrency conservative to avoid
  Cloudflare/IP blocks. Don't expose the crawl trigger publicly (hence the API key).
* **Caveat at multiple replicas:** the "don't run the same job twice" guard and the
  rate limiter are per-process. For >1 replica that *generate* predictions, run a
  single dedicated worker, or move to a job queue (Arq/Celery + Redis) and a shared
  rate-limit store. Read-only replicas are unaffected.

## Configuration

All env vars are prefixed `LCCN_` — see [`.env.example`](.env.example). Key ones:

| Var | Purpose |
| --- | --- |
| `LCCN_MONGODB_URI`, `LCCN_DB_NAME` | Database connection |
| `LCCN_CORS_ORIGINS` | Allowed browser origin(s); set to your site in prod |
| `LCCN_API_KEY` | Secret gating the admin/crawl endpoints (`X-API-Key`) |
| `LCCN_PUBLIC_CRAWL_ENABLED` | If false (prod default), public `/predict` serves cached only |
| `LCCN_RATE_LIMIT_PER_MINUTE` | Per-IP rate limit on public reads (0 = off) |
| `LCCN_RATING_BATCH_SIZE` | Users per batched GraphQL request (perf) |
| `LCCN_RANKING_CONCURRENCY`, `LCCN_RATING_CONCURRENCY` | Crawl parallelism |
| `LCCN_RATING_CACHE_TTL_HOURS` | How long a cached rating stays fresh |
| `LCCN_SCHEDULER_ENABLED` | In-process rating-cache warm-up |
| `LCCN_AUTO_PREDICT_ENABLED` | In-process auto-predict of the latest contest (always-on hosting) |

## Tests

```bash
pytest            # algorithm (FFT vs naive) + API/service (in-memory Mongo, mocked crawler)
```

## Project layout

```
predictor/
  core/      elo.py (oracle) · fft.py (fast path) · engine.py (predict)
  crawler/   http.py · ranking.py · user_rating.py (GraphQL)
  db/        models.py · mongodb.py
  service/   predict_service.py (crawl -> cache -> engine -> persist)
  api/       routes.py · schemas.py
  scheduler.py · config.py · main.py
static/index.html   minimal demo UI
tests/              test_elo.py · test_api.py
```

## Credits

Prediction algorithm reverse-engineered and originally implemented by
[lccn_predictor](https://github.com/baoliay2008/lccn_predictor) (AGPL-3.0). This rebuild
ports that core faithfully.
