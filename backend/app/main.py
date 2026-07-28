"""FastAPI application entry point.

    uvicorn app.main:app --reload --port 8000
"""
import logging
import math
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from . import config, db
from .api import (
    activity,
    admin,
    albums,
    annotations,
    attributes,
    chat,
    describe,
    detect,
    leakage,
    mcp,
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


def _finite(o):
    """Non-finite floats replaced by their repr, recursively. Only used on
    echoed *invalid* input — a NaN got this far precisely because it is not
    representable in JSON, so the repr is the honest rendering of it."""
    if isinstance(o, float) and not math.isfinite(o):
        return repr(o)
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_finite(v) for v in o]
    return o


@app.exception_handler(RequestValidationError)
async def validation_error_that_serializes(_request, exc: RequestValidationError):
    """The default 422, minus a measured 500: FastAPI echoes the offending
    input in the error detail, and starlette's JSONResponse (allow_nan=False)
    then refuses to render a body that contained NaN — so the one input the
    validator correctly rejected crashed the rejection. Same shape as the
    default handler otherwise."""
    return JSONResponse(status_code=422,
                        content={"detail": _finite(jsonable_encoder(exc.errors()))})

# Frontend dev server origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

for router in (samples.router, search.router, stats.router, map_api.router,
               tags.router, qa.router, qa_run.router, attributes.router,
               describe.router, detect.router, leakage.router,
               eval_api.router, admin.router, chat.router,
               views.router, albums.router, activity.router,
               annotations.router):
    app.include_router(router, prefix="/api")

# The MCP endpoint lives at /mcp, unprefixed: MCP clients configure a root-ish
# URL, and the protocol is a peer of the REST surface, not part of it.
app.include_router(mcp.router)

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
