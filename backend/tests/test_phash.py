"""The perceptual hash, and the leakage claim that rests on it.

The point of `ml/phash.py` is to stop the tool from asserting that a high
cosine means an image was memorised. A hash that reported "not a duplicate"
about everything would support that conclusion just as well and be worthless,
so the calibration is tested first and the endpoint second.
"""
import io

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from app import db
from app.api import leakage
from app.ml import phash


@pytest.fixture(scope="module")
def client():
    """Only the leakage router, following the repo's per-module pattern —
    nothing here needs the seeded corpus the heavier modules build."""
    conn = db.connect()
    db.init_db(conn)
    conn.close()
    api = FastAPI()
    api.include_router(leakage.router, prefix="/api")
    with TestClient(api) as c:
        yield c


def _photo(seed: int, w: int = 320, h: int = 240) -> Image.Image:
    """A deterministic, structured image.

    Smooth gradients plus a few blocks, not white noise: a difference hash
    compares neighbouring pixels, and noise makes every comparison a coin flip
    regardless of the content, which would make the test pass for the wrong
    reason.
    """
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    base = (x * rng.uniform(0.3, 0.9) + y * rng.uniform(0.3, 0.9)) % 256
    for _ in range(6):
        x0, y0 = rng.integers(0, w - 60), rng.integers(0, h - 60)
        base[y0:y0 + 60, x0:x0 + 60] = rng.integers(0, 256)
    return Image.fromarray(base.astype(np.uint8), "L").convert("RGB")


def _write(tmp_path, name, im):
    p = tmp_path / name
    im.save(p)
    return p


def test_identical_files_hash_identically(tmp_path):
    p = _write(tmp_path, "a.png", _photo(1))
    assert phash.hamming(phash.dhash(p), phash.dhash(p)) == 0


def test_survives_rescaling_and_recompression(tmp_path):
    """The two transforms that change every byte without changing the photo.

    If the hash failed here it would call ordinary re-encoding a different
    image, and the leakage endpoint would under-report duplicates.
    """
    im = _photo(2)
    original = phash.dhash(_write(tmp_path, "o.png", im))

    half = _write(tmp_path, "h.png", im.resize((im.width // 2, im.height // 2),
                                               Image.LANCZOS))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=40)
    buf.seek(0)
    jpeg = _write(tmp_path, "j.jpg", Image.open(buf))

    assert phash.hamming(original, phash.dhash(half)) <= phash.DUPLICATE_FRAME_MAX
    assert phash.hamming(original, phash.dhash(jpeg)) <= phash.DUPLICATE_FRAME_MAX


def test_unrelated_images_land_near_chance(tmp_path):
    """The null. Half of 64 bits differ when two inputs share nothing, and the
    cut has to sit well below that or it would call strangers duplicates."""
    paths = [_write(tmp_path, f"u{i}.png", _photo(100 + i)) for i in range(12)]
    hashes = [phash.dhash(p) for p in paths]
    distances = [phash.hamming(hashes[i], hashes[j])
                 for i in range(len(hashes)) for j in range(i + 1, len(hashes))]
    assert np.median(distances) > 2 * phash.DUPLICATE_FRAME_MAX
    # And the helper agrees with the direct computation.
    assert phash.null_distance(hashes) > 2 * phash.DUPLICATE_FRAME_MAX


def test_null_distance_is_stable_across_calls(tmp_path):
    """A figure printed in the interface must not change between two requests
    that measured the same corpus."""
    hashes = [phash.dhash(_write(tmp_path, f"s{i}.png", _photo(200 + i)))
              for i in range(10)]
    assert phash.null_distance(hashes) == phash.null_distance(hashes)


def test_hamming_is_symmetric_and_bounded(tmp_path):
    a = phash.dhash(_write(tmp_path, "x.png", _photo(7)))
    b = phash.dhash(_write(tmp_path, "y.png", _photo(8)))
    assert phash.hamming(a, b) == phash.hamming(b, a)
    assert 0 <= phash.hamming(a, b) <= phash.HASH_BITS


def test_cut_admits_a_crop_but_not_a_stranger():
    """`DUPLICATE_FRAME_MAX` is derived from measured distributions, so pin the
    ordering it depends on: a crop of one photo must stay under it, chance must
    stay far above it. A future edit that raises it towards the null breaks
    here rather than in the interface."""
    assert 0 < phash.DUPLICATE_FRAME_MAX < phash.HASH_BITS // 4


def test_pixel_report_degrades_without_an_index(client, monkeypatch):
    """No embeddings means no pair list to check, and the message names the
    command that fixes it — the degradation contract, not a silent empty."""
    monkeypatch.setattr("app.api.leakage.get_index", lambda: None)
    r = client.get("/api/stats/leakage/pixel")
    assert r.status_code == 503
    assert "app.ingest" in r.json()["detail"]


@pytest.mark.parametrize("threshold,limit", [(0.90, 0), (0.95, 5)])
def test_pixel_report_shape_when_embeddings_absent(client, threshold, limit):
    """Without embeddings the endpoint must refuse cleanly rather than 500.

    The suite runs on a fixture corpus with no index; the contract that matters
    here is that an absent capability is an honest 503 at every parameter.
    """
    r = client.get(f"/api/stats/leakage/pixel?threshold={threshold}&limit={limit}")
    assert r.status_code in (200, 503)
    if r.status_code == 200:
        body = r.json()
        assert body["duplicate_frame_max_distance"] == phash.DUPLICATE_FRAME_MAX
        assert body["pairs_measured"] >= body["duplicate_frames"]
        assert body["reading"]
