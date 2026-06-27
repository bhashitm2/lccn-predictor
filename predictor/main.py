"""FastAPI application entry point.

Run: ``uvicorn predictor.main:app --reload``
Docs: ``/docs``  ·  Demo: ``/``  ·  API: ``/api/v1/contest/{slug}/predict``
"""
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from loguru import logger
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from predictor import __version__
from predictor.api.routes import admin as admin_router
from predictor.api.routes import router as api_router
from predictor.config import get_settings
from predictor.db.mongodb import close_db, init_db
from predictor.scheduler import start_scheduler, stop_scheduler

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


# Server logs: keep tracebacks but don't dump local variable values (which can
# include usernames) into logs.
logger.remove()
logger.add(sys.stderr, level="INFO", backtrace=True, diagnose=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    start_scheduler()
    try:
        yield
    finally:
        stop_scheduler()
        await close_db()


app = FastAPI(
    title="LeetCode Contest Rating Predictor",
    version=__version__,
    description=(
        "Predicts LeetCode weekly/biweekly contest rating changes from a contest "
        "slug, using a faithful port of LeetCode's Elo algorithm (FFT-accelerated). "
        "Call GET /api/v1/contest/{slug}/predict from your website."
    ),
    lifespan=lifespan,
)

_settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.cors_origin_list,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)
app.include_router(admin_router)


@app.get("/healthz", tags=["meta"])
async def healthz():
    return JSONResponse({"status": "ok", "version": __version__})


@app.get("/", include_in_schema=False)
async def demo():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    return JSONResponse({"status": "ok", "docs": "/docs"})


# Serve the rest of the demo assets (if any) under /static
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
