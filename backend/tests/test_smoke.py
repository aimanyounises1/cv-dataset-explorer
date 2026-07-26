"""Smoke tests: API works end-to-end on a seeded temp database, without the
embedding stack (exercises the graceful-degradation path).
Data-dir isolation happens in conftest.py, before any `app` import.

    cd backend && pytest
"""
import json

import pytest
from fastapi.testclient import TestClient

from app import db
from app.main import app


@pytest.fixture(scope="module")
def client():
    conn = db.connect()
    db.init_db(conn)
    for i, (split, caption) in enumerate([
        ("train", "a black dog runs across the wet grass"),
        ("train", "two children play soccer in a park"),
        ("test", "a man rides a red mountain bike on a trail"),
    ]):
        cur = conn.execute(
            "INSERT INTO samples(dataset, filename, split, width, height, filesize) "
            "VALUES ('flickr8k', ?, ?, 400, 300, 1000)", (f"img_{i}.jpg", split))
        sid = cur.lastrowid
        ccur = conn.execute(
            "INSERT INTO captions(sample_id, idx, text) VALUES (?, 0, ?)", (sid, caption))
        conn.execute("INSERT INTO captions_fts(rowid, text) VALUES (?, ?)",
                     (ccur.lastrowid, caption))
    # Distinct axis scores per sample so range filters and sorts are testable
    # without running the analysis pass (which needs images and embeddings).
    for filename, (leg, rar, diff, clut) in {
        "img_0.jpg": (1, 8, 2, 3),
        "img_1.jpg": (5, 5, 5, 5),
        "img_2.jpg": (9, 1, 9, 7),
    }.items():
        conn.execute(
            "UPDATE samples SET legibility=?, rarity=?, difficulty=?, clutter=?, "
            "axis_detail=? WHERE filename=?",
            (leg, rar, diff, clut,
             '{"difficulty": {"agreement": 0.11, "consistency": 0.42}}', filename))
    conn.commit()
    conn.close()
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["samples"] == 3


def test_list_and_filter(client):
    r = client.get("/api/samples")
    assert r.status_code == 200
    assert r.json()["total"] == 3
    r = client.get("/api/samples", params={"split": "test"})
    assert r.json()["total"] == 1


def test_detail(client):
    sid = client.get("/api/samples").json()["items"][0]["id"]
    r = client.get(f"/api/samples/{sid}")
    assert r.status_code == 200
    assert len(r.json()["captions"]) == 1


def test_keyword_search(client):
    r = client.get("/api/search", params={"q": "dog grass", "mode": "keyword"})
    assert r.status_code == 200
    items = r.json()["items"]
    assert len(items) == 1
    assert "dog" in items[0]["caption"]


def test_hybrid_degrades_without_embeddings(client):
    r = client.get("/api/search", params={"q": "dog", "mode": "hybrid"})
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is True
    assert body["mode_used"] == "keyword"
    assert len(body["items"]) == 1


def test_boosted_falls_all_the_way_back_to_keyword(client):
    """Two degradations in one request, on the install a reviewer starts from.

    With no embeddings there is no PRISM model and no semantic path either, so
    the response has to describe where the ranking ended up rather than where
    it was asked to go -- and publish no score basis, because BM25 exposes no
    number. The message is the only thing telling the user what to run.
    """
    body = client.get("/api/search",
                      params={"q": "dog", "mode": "boosted"}).json()
    assert body["degraded"] is True
    assert body["mode_used"] == "keyword"
    assert "keyword search" in (body["message"] or "")
    assert body["score_basis"] is None
    assert len(body["items"]) == 1


def test_tags_roundtrip(client):
    sid = client.get("/api/samples").json()["items"][0]["id"]
    assert client.post(f"/api/samples/{sid}/tags", json={"name": "Edge-Case"}).status_code == 200
    detail = client.get(f"/api/samples/{sid}").json()
    assert "edge-case" in detail["tags"]
    r = client.get("/api/samples", params={"tag": "edge-case"})
    assert r.json()["total"] == 1
    assert client.delete(f"/api/samples/{sid}/tags/edge-case").status_code == 200


def test_stats(client):
    r = client.get("/api/stats/overview")
    assert r.status_code == 200
    assert r.json()["embeddings_available"] is False
    r = client.get("/api/stats/captions")
    assert r.status_code == 200
    assert len(r.json()["top_words"]) > 0


def test_keyword_search_respects_filters_in_sql(client):
    # "a man rides..." lives in the test split; filtering to train must
    # exclude it even though it's the top keyword hit.
    r = client.get("/api/search", params={"q": "rides", "mode": "keyword"})
    assert len(r.json()["items"]) == 1
    r = client.get("/api/search",
                   params={"q": "rides", "mode": "keyword", "split": "train"})
    assert r.json()["items"] == []


def test_porter_stemming(client):
    # Caption says "runs"; the stemmed FTS index should match query "running".
    r = client.get("/api/search", params={"q": "running", "mode": "keyword"})
    items = r.json()["items"]
    assert len(items) == 1 and "runs" in items[0]["match_caption"]


def test_match_explanation_fields(client):
    r = client.get("/api/search", params={"q": "soccer", "mode": "keyword"})
    item = r.json()["items"][0]
    assert "soccer" in item["match_caption"]
    assert "soccer" in item["matched_terms"]


def test_match_paths_name_the_retrieval_path(client):
    # Every result says which path found it and at what rank (FR-SE-U1).
    item = client.get("/api/search",
                      params={"q": "soccer", "mode": "keyword"}).json()["items"][0]
    assert item["match_paths"] == [{"path": "keyword", "rank": 1}]


def test_keyword_mode_publishes_no_score_basis(client):
    # Keyword ranking exposes no numeric score, so there is nothing to mislabel.
    body = client.get("/api/search", params={"q": "soccer", "mode": "keyword"}).json()
    assert body["score_basis"] is None
    assert body["items"][0]["score"] is None


def test_term_stats_report_document_frequency(client):
    # A term in 1 of 3 images is 33% of the corpus: over the 20% warn threshold.
    stats = client.get("/api/search",
                       params={"q": "dog", "mode": "keyword"}).json()["term_stats"]
    assert stats == [{"term": "dog", "images": 1, "fraction": 0.3333, "common": True}]


def test_term_stats_flag_terms_that_match_nothing(client):
    # A zero-hit term is what explains an empty result page (FR-SE-U3).
    body = client.get("/api/search", params={"q": "zebra", "mode": "keyword"}).json()
    assert body["items"] == []
    assert body["term_stats"] == [
        {"term": "zebra", "images": 0, "fraction": 0.0, "common": False}]


def test_stopwords_are_not_reported_as_common_terms(client):
    # "the" is in most captions but warning about it would be noise.
    stats = client.get("/api/search",
                       params={"q": "the dog", "mode": "keyword"}).json()["term_stats"]
    assert [s["term"] for s in stats] == ["dog"]


def test_paging_stops_at_the_ranking_horizon_instead_of_repeating(client, monkeypatch):
    """Regression: paging past SEARCH_DEPTH used to return duplicate images.

    The depth was previously widened to `offset + top_k` so a caller could page
    further, but that recomputes the fusion against a larger candidate pool, and
    the tail re-ranks — measured on the real corpus, adjacent pages either side
    of row 300 shared 4 images. A ranking is only defined as deep as it was
    computed, so the horizon is now hard and `has_more` tells the truth about it.
    """
    from app import config
    monkeypatch.setattr(config, "SEARCH_DEPTH", 2)

    first = client.get("/api/search", params={
        "q": "a", "mode": "keyword", "top_k": 1, "offset": 0}).json()
    assert len(first["items"]) == 1
    assert first["depth_limit"] == 2 and first["has_more"] is True

    second = client.get("/api/search", params={
        "q": "a", "mode": "keyword", "top_k": 1, "offset": 1}).json()
    assert len(second["items"]) == 1
    # Row 2 is the horizon: more matches exist, but none were ranked.
    assert second["has_more"] is False and second["depth_reached"] is True

    beyond = client.get("/api/search", params={
        "q": "a", "mode": "keyword", "top_k": 1, "offset": 2}).json()
    assert beyond["items"] == [] and beyond["has_more"] is False

    ids = [i["id"] for i in first["items"] + second["items"]]
    assert len(ids) == len(set(ids)), "no image may appear on two pages"


def test_axis_range_filter_on_browse(client):
    # Three samples with difficulty 2, 5, 9.
    assert client.get("/api/samples").json()["total"] == 3
    assert client.get("/api/samples", params={"difficulty_min": 5}).json()["total"] == 2
    assert client.get("/api/samples", params={"difficulty_max": 4}).json()["total"] == 1
    body = client.get("/api/samples",
                      params={"difficulty_min": 4, "difficulty_max": 6}).json()
    assert body["total"] == 1 and body["items"][0]["axes"]["difficulty"] == 5


def test_several_axes_constrained_at_once(client):
    # difficulty>=5 keeps samples 1 and 2; rarity>=5 keeps 0 and 1; both keep 1.
    body = client.get("/api/samples",
                      params={"difficulty_min": 5, "rarity_min": 5}).json()
    assert body["total"] == 1
    axes = body["items"][0]["axes"]
    assert (axes["difficulty"], axes["rarity"]) == (5, 5)


def test_axis_filter_is_applied_inside_the_ranking_not_after_it(client):
    """Criterion 2, in the shape of test_keyword_search_respects_filters_in_sql.

    "rides" matches only the sample scored difficulty 9. If the axis predicate
    were applied to an already-limited page the query would still return that
    row and then drop it, reporting a total computed over the wrong set; applied
    inside the ranking, the candidate never enters it.
    """
    r = client.get("/api/search", params={"q": "rides", "mode": "keyword"})
    assert len(r.json()["items"]) == 1
    r = client.get("/api/search", params={"q": "rides", "mode": "keyword",
                                          "difficulty_max": 5})
    assert r.json()["items"] == []


def test_axis_sort_orders_the_whole_filtered_set(client):
    """Criterion 5: sorting is server-side, so page 1 holds the global extreme."""
    desc = client.get("/api/samples",
                      params={"sort": "difficulty_desc"}).json()["items"]
    assert [i["axes"]["difficulty"] for i in desc] == [9, 5, 2]
    asc = client.get("/api/samples",
                     params={"sort": "difficulty_asc"}).json()["items"]
    assert [i["axes"]["difficulty"] for i in asc] == [2, 5, 9]
    # One item per page: the first page must still be the global maximum.
    top = client.get("/api/samples",
                     params={"sort": "difficulty_desc", "per_page": 1}).json()
    assert top["items"][0]["axes"]["difficulty"] == 9 and top["total"] == 3


def test_axis_sort_applies_to_search_results_too(client):
    items = client.get("/api/search", params={
        "q": "a", "mode": "keyword", "sort": "difficulty_desc"}).json()
    scores = [i["axes"]["difficulty"] for i in items["items"]]
    assert scores == sorted(scores, reverse=True)
    assert items["sort"] == "difficulty_desc"


def test_unknown_sort_key_is_ignored_not_interpolated(client):
    # The axis name reaches SQL as an identifier, so anything unrecognised must
    # fall back to the default order rather than being trusted.
    r = client.get("/api/samples", params={"sort": "difficulty; DROP TABLE samples--"})
    assert r.status_code == 200
    assert r.json()["total"] == 3
    assert client.get("/api/samples").json()["total"] == 3


def test_axis_detail_travels_with_the_card(client):
    # Item 5 depends on this: the components behind a score arrive with the
    # score, so a card can explain itself without a second request.
    item = client.get("/api/samples", params={"per_page": 1}).json()["items"][0]
    assert item["axes"]["detail"]["difficulty"]["agreement"] == 0.11


def test_export_composes_with_axis_filters(client):
    body = client.get("/api/export", params={"difficulty_min": 5}).json()
    assert body["count"] == 2
    assert body["filters"]["axes"] == {"difficulty": [5, None]}


def test_bulk_tag(client):
    ids = [it["id"] for it in client.get("/api/samples").json()["items"][:2]]
    r = client.post("/api/tags/bulk", json={"sample_ids": ids, "name": "Batch-Tag"})
    assert r.status_code == 200 and r.json()["tag"] == "batch-tag"
    assert client.get("/api/samples", params={"tag": "batch-tag"}).json()["total"] == 2


def test_qa_and_eval_degrade_gracefully(client):
    r = client.get("/api/qa/summary")
    assert r.status_code == 200 and r.json()["available"] is False
    r = client.get("/api/eval/retrieval")
    assert r.status_code == 200 and r.json()["available"] is False
    assert client.get("/api/qa/captions").json() == []
    assert client.get("/api/attributes/coverage").json() == []


def test_admin_reload(client):
    r = client.post("/api/admin/reload")
    assert r.status_code == 200
    assert r.json()["image_index"] is False


def test_chat_unavailable_is_graceful(client, monkeypatch):
    # Pin Ollama to a dead port so the degraded path is exercised even on
    # machines where a real Ollama happens to be running.
    from app import config
    monkeypatch.setattr(config, "OLLAMA_URL", "http://127.0.0.1:9")
    r = client.get("/api/chat/status")
    assert r.status_code == 200 and r.json()["available"] is False
    r = client.post("/api/chat",
                    json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 503  # clear setup instructions, not a crash
    assert "Ollama" in r.json()["detail"] or "agent stack" in r.json()["detail"]


def test_export_manifest(client):
    r = client.get("/api/export", params={"split": "test"})
    body = r.json()
    assert body["count"] == 1
    assert body["samples"][0]["captions"]
    assert body["filters"]["embed_model"]  # the slice records how to regenerate it


ORIGINAL_CSV_HEADER = "id,filename,split,captions,tags"


def test_export_csv_and_jsonl(client):
    csv_resp = client.get("/api/export", params={"split": "test", "format": "csv"})
    assert csv_resp.status_code == 200
    assert "text/csv" in csv_resp.headers["content-type"]
    assert "attachment" in csv_resp.headers["content-disposition"]
    lines = csv_resp.text.strip().splitlines()
    # Score columns are APPENDED, never interleaved, so a consumer reading the
    # original five by index is unaffected by their arrival.
    assert lines[0].startswith(ORIGINAL_CSV_HEADER + ",")
    assert "difficulty" in lines[0]
    assert len(lines) == 2  # header + the one test-split row


def test_export_scores_false_reproduces_the_original_shape(client):
    """`scores=false` is the escape hatch for anything that parsed the old CSV.

    Asserted rather than assumed, because the whole argument for defaulting the
    new columns ON is that opting out is exact and costless.
    """
    resp = client.get("/api/export",
                      params={"split": "test", "format": "csv", "scores": "false"})
    assert resp.status_code == 200
    assert resp.text.strip().splitlines()[0] == ORIGINAL_CSV_HEADER

    js = client.get("/api/export",
                    params={"split": "test", "scores": "false"}).json()
    assert set(js["samples"][0]) == {"id", "filename", "split", "captions", "tags"}


def test_export_carries_the_signals_the_tool_computed(client):
    """Exporting "the hardest images" in a file that cannot say how hard any one
    of them is forces every consumer to re-derive work already done."""
    js = client.get("/api/export", params={"split": "test"}).json()
    assert js["filters"]["scores"] is True
    # The manifest must say what the axis numbers mean; percentile ranks are not
    # comparable across corpora and a bare 0-10 column invites treating them as if
    # they were.
    assert "percentile" in js["filters"]["axis_semantics"]
    sample = js["samples"][0]
    assert "axes" in sample and sample["axes"], sample
    assert set(sample["axes"]) <= set(db.AXES)

    jsonl = client.get("/api/export", params={"split": "test", "format": "jsonl"})
    records = [json.loads(ln) for ln in jsonl.text.strip().splitlines()]
    assert records[0]["_manifest"]["split"] == "test"  # provenance first
    assert records[1]["captions"]
