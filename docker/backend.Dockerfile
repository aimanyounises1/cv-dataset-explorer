# CV Dataset Explorer API.
#
# Runs the same command the README documents for a local machine
# (uvicorn app.main:app --port 8000), with the state directory moved to /data
# so it can be a mounted volume. Build context is the repository root; see
# docker/backend.Dockerfile.dockerignore for what is excluded from it.
#
# Two stages: several wheels want a compiler at install time, and none of them
# want one afterwards, so the toolchain stays in the builder and only the
# finished virtualenv is copied forward.

FROM python:3.13-slim AS builder

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH
RUN python -m venv "$VIRTUAL_ENV"

# torch comes from the +cpu index on every architecture, not from PyPI.
# PyPI's default wheel is the CUDA build on linux/amd64 *and* on linux/arm64
# (measured: `torch 2.13.0+cu130` with `torch.cuda.is_available() == False`),
# and it drags in ~2.9 GB of nvidia libraries plus ~650 MB of triton that a
# container with no GPU passthrough can never load. The +cpu wheel is the same
# torch without them. Installing it first means requirements.txt's `torch>=2.2`
# below is already satisfied rather than resolved a second time.
RUN pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu "torch>=2.2"

COPY backend/requirements.txt backend/requirements-agent.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-agent.txt


FROM python:3.13-slim

# curl is here for the HEALTHCHECK below and nothing else.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CVDE_DATA_DIR=/data \
    HF_HOME=/opt/hf

WORKDIR /srv
COPY backend/pyproject.toml ./
COPY backend/app ./app
# The suite is small and text-only, and shipping it means the image can be
# checked on the machine that will run it:
#   docker compose exec backend python -m pytest -q
# tests/conftest.py repoints CVDE_DATA_DIR before importing app, so this never
# touches the mounted /data. Tests that genuinely construct an optional Qwen
# model skip when requirements-qwen.txt is absent; provider contracts still run.
COPY backend/tests ./tests

# Both are mount points in compose. Creating them keeps an unmounted
# `docker run` working instead of failing on the first write.
RUN mkdir -p /data /opt/hf

EXPOSE 8000

# /api/tags is one indexed SQLite read. It proves uvicorn is serving and the
# database opened, without the side effect /api/health has of pulling the
# embedding matrix into memory. start-period covers first-boot schema setup.
# 127.0.0.1 rather than localhost: uvicorn binds 0.0.0.0 (IPv4), and an
# explicit address keeps the probe independent of how the container resolves
# localhost. See the same note in frontend.Dockerfile.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=5 \
  CMD curl -f http://127.0.0.1:8000/api/tags || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
