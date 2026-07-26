"""FastAPI application entry point.

    uvicorn app.main:app --reload --port 8000
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from . import config, db
from .api import (
    admin,
    attributes,
    chat,
    describe,
    leakage,
    qa,
    qa_run,
    samples,
    search,
    stats,
    tags,
    views,
)
from .api import eval as eval_api
from .api import map as map_api

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    with db.get_db() as conn:
        db.init_db(conn)
    yield


app = FastAPI(title="CV Dataset Explorer", version="1.0.0", lifespan=lifespan)

# Frontend dev server origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

for router in (samples.router, search.router, stats.router, map_api.router,
               tags.router, qa.router, qa_run.router, attributes.router,
               describe.router, leakage.router,
               eval_api.router, admin.router, chat.router,
               views.router):
    app.include_router(router, prefix="/api")

# Local image/thumbnail serving.
config.ensure_dirs()
app.mount("/media/images", StaticFiles(directory=config.IMAGES_DIR), name="images")
app.mount("/media/thumbs", StaticFiles(directory=config.THUMBS_DIR), name="thumbs")
# QA screenshots and decks. Build artifacts, gitignored, served read-only like
# the thumbnails so a status report's evidence is viewable in the browser that
# asked for it. `/api/qa/artifact/...` serves the same files for callers that
# would rather not depend on a mount.
app.mount("/media/qa", StaticFiles(directory=config.QA_DIR), name="qa")


@app.get("/api/health")
def health():
    from .ml.index import get_index

    with db.get_db() as conn:
        n = conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
    return {"status": "ok", "samples": n, "semantic_search": get_index() is not None}
