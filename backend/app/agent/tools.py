"""LangChain tools exposing the platform's capabilities to the assistant.

Tools call the same service functions as the REST API (no HTTP hop). Image-
returning tools include a `sample_ids` field in their JSON output; the chat
endpoint parses those to attach renderable image cards to the reply.
"""
import json

from langchain_core.tools import tool

from .. import db


def _cards(items) -> list[dict]:
    return [
        {"sample_id": it.id, "caption": it.match_caption or it.caption,
         "split": it.split, "score": it.score}
        for it in items
    ]


@tool
def search_images(query: str, mode: str = "hybrid", top_k: int = 8) -> str:
    """Search the dataset for images matching a natural-language description.
    mode: 'semantic' (meaning-based), 'keyword' (exact words in captions), or
    'hybrid' (both fused, recommended). Returns matching samples with captions."""
    from ..api.search import run_search

    conn = db.connect()
    try:
        res = run_search(conn, query, mode=mode, top_k=min(top_k, 20))
        return json.dumps({
            "mode_used": res.mode_used, "degraded": res.degraded,
            "results": _cards(res.items),
            "sample_ids": [it.id for it in res.items],
        })
    finally:
        conn.close()


@tool
def find_similar(sample_id: int, top_k: int = 8) -> str:
    """Find images visually/semantically similar to a given sample id."""
    from ..api.deps import first_captions
    from ..ml.index import get_index

    index = get_index()
    if index is None:
        return json.dumps({"error": "embeddings not computed"})
    results = index.similar_to(sample_id, top_k=min(top_k, 20))
    conn = db.connect()
    try:
        caps = first_captions(conn, [sid for sid, _ in results])
        return json.dumps({
            "results": [{"sample_id": sid, "score": round(s, 4),
                         "caption": caps.get(sid)} for sid, s in results],
            "sample_ids": [sid for sid, _ in results],
        })
    finally:
        conn.close()


@tool
def get_sample_details(sample_id: int) -> str:
    """Get full details for one sample: all captions (with image-caption
    agreement scores), user tags, VLM tags, zero-shot attributes, metadata."""
    conn = db.connect()
    try:
        row = conn.execute("SELECT * FROM samples WHERE id = ?", (sample_id,)).fetchone()
        if row is None:
            return json.dumps({"error": f"sample {sample_id} not found"})
        captions = [{"text": r["text"], "agreement": r["agreement"]}
                    for r in conn.execute(
                        "SELECT text, agreement FROM captions WHERE sample_id = ? ORDER BY idx",
                        (sample_id,))]
        attrs = {r["grp"]: r["label"] for r in conn.execute(
            "SELECT grp, label FROM attributes WHERE sample_id = ?", (sample_id,))}
        tags = [r["name"] for r in conn.execute(
            "SELECT t.name FROM tags t JOIN sample_tags st ON st.tag_id = t.id "
            "WHERE st.sample_id = ?", (sample_id,))]
        return json.dumps({
            "sample_id": sample_id, "filename": row["filename"], "split": row["split"],
            "width": row["width"], "height": row["height"],
            "captions": captions, "attributes": attrs, "tags": tags,
            "sample_ids": [sample_id],
        })
    finally:
        conn.close()


@tool
def dataset_overview() -> str:
    """Dataset-level statistics: sample/caption counts, splits, average caption
    length, whether semantic search and QA scores are available."""
    conn = db.connect()
    try:
        from ..ml.index import get_caption_index, get_index

        total = conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
        caps = conn.execute("SELECT COUNT(*) FROM captions").fetchone()[0]
        splits = {r["split"]: r["n"] for r in conn.execute(
            "SELECT split, COUNT(*) AS n FROM samples GROUP BY split")}
        return json.dumps({
            "total_samples": total, "total_captions": caps, "splits": splits,
            "semantic_search": get_index() is not None,
            "caption_qa_scores": get_caption_index() is not None,
        })
    finally:
        conn.close()


@tool
def attribute_coverage() -> str:
    """Zero-shot attribute coverage: how the dataset splits across setting,
    time of day, environment, and main subject. Use to find rare slices
    (long-tail / potential edge cases)."""
    conn = db.connect()
    try:
        total = conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0] or 1
        rows = conn.execute(
            "SELECT grp, label, COUNT(*) AS n FROM attributes "
            "GROUP BY grp, label ORDER BY grp, n DESC").fetchall()
        if not rows:
            return json.dumps({"error": "attributes not computed — run `python -m app.analyze`"})
        out: dict[str, dict[str, str]] = {}
        for r in rows:
            out.setdefault(r["grp"], {})[r["label"]] = f"{r['n']} ({r['n'] / total:.1%})"
        return json.dumps(out)
    finally:
        conn.close()


@tool
def rare_slice_examples(grp: str, label: str, top_k: int = 8) -> str:
    """List example images from a specific attribute slice, e.g. grp='time_of_day',
    label='night'. Useful after attribute_coverage to show samples from a rare slice."""
    conn = db.connect()
    try:
        rows = conn.execute(
            "SELECT s.*, a.confidence FROM attributes a JOIN samples s ON s.id = a.sample_id "
            "WHERE a.grp = ? AND a.label = ? ORDER BY a.confidence DESC LIMIT ?",
            (grp, label, min(top_k, 20))).fetchall()
        from ..api.deps import first_captions

        caps = first_captions(conn, [r["id"] for r in rows])
        return json.dumps({
            "results": [{"sample_id": r["id"], "caption": caps.get(r["id"]),
                         "confidence": round(r["confidence"], 3)} for r in rows],
            "sample_ids": [r["id"] for r in rows],
        })
    finally:
        conn.close()


@tool
def suspect_captions(limit: int = 8) -> str:
    """Most suspect captions in the dataset — lowest image-caption agreement
    (possible annotation errors). Returns caption, score, and sample id."""
    conn = db.connect()
    try:
        rows = conn.execute(
            "SELECT c.text, c.agreement, c.sample_id FROM captions c "
            "WHERE c.agreement IS NOT NULL ORDER BY c.agreement ASC LIMIT ?",
            (min(limit, 20),)).fetchall()
        if not rows:
            return json.dumps({"error": "QA scores not computed — run `python -m app.analyze`"})
        return json.dumps({
            "results": [{"sample_id": r["sample_id"], "caption": r["text"],
                         "agreement": round(r["agreement"], 4)} for r in rows],
            "sample_ids": [r["sample_id"] for r in rows],
        })
    finally:
        conn.close()


@tool
def tag_samples(sample_ids: list[int], tag: str, reason: str = "") -> str:
    """PROPOSE a curation tag for a list of sample ids (e.g. tag interesting
    finds as 'edge-case'). Nothing is written: the user sees the candidates in
    the conversation and approves or rejects — tell them the proposal awaits
    their click. `reason` is one sentence on why these samples qualify."""
    conn = db.connect()
    try:
        tag = tag.strip().lower()
        if not tag or not sample_ids:
            return json.dumps({"error": "need a tag name and at least one sample id"})
        wanted = sample_ids[:200]
        # Which of the requested ids actually exist. Checked BEFORE proposing,
        # because an 8B model asked for "test-split images below 0.10 agreement"
        # was observed inventing round numbers — [1234, 5678, 9012] — and two of
        # them existed. A proposal built on guesses would put a human approval
        # step in front of fiction, which is not better than fiction.
        qmarks = ",".join("?" * len(wanted))
        real = {r["id"] for r in conn.execute(
            f"SELECT id FROM samples WHERE id IN ({qmarks})", wanted)}
        missing = [i for i in wanted if i not in real]
        if not real:
            return json.dumps({
                "error": f"none of those {len(wanted)} sample ids exist in this "
                         f"dataset — nothing was proposed",
                "missing": missing[:20],
                "hint": "Use ids that came from a tool result in this "
                        "conversation, never ids you inferred."})

        keep = [sid for sid in wanted if sid in real]
        block = {
            "kind": "tag_proposal",
            "title": f"Tag {len(keep)} samples as ‘{tag}’?",
            "source": "assistant proposal — nothing is written until you approve",
            "tag": tag,
            "sample_ids": keep,
            "reason": reason.strip()[:500] or None,
            "missing": missing[:20],
        }
        out = {"proposed": True, "tag": tag, "candidates": len(keep),
               "requested": len(wanted), "blocks": [block],
               "next": "The user must approve this in the conversation before "
                       "any tag is written. Report it as a proposal, not as a "
                       "completed action."}
        if missing:
            out["not_found"] = missing[:20]
            out["warning"] = (f"{len(missing)} of the {len(wanted)} ids do not "
                              f"exist and were dropped from the proposal.")
        return json.dumps(out)
    finally:
        conn.close()




@tool
def inspect_album(album: str) -> str:
    """Inspect a saved album by name or id: metadata, measured analysis
    (common signals, splits, coherence, centroid outliers) and its members.
    Use when the user asks about an album — what it contains, which members
    are rare or out of place, whether it is coherent — especially for
    rare-scenario and coverage-gap albums."""
    conn = db.connect()
    try:
        ref = album.strip()
        row = (conn.execute("SELECT * FROM albums WHERE id = ?",
                            (int(ref),)).fetchone()
               if ref.isdigit() else
               conn.execute("SELECT * FROM albums WHERE name = ? COLLATE NOCASE",
                            (ref,)).fetchone())
        if row is None:
            names = [r["name"] for r in conn.execute(
                "SELECT name FROM albums ORDER BY position, id LIMIT 20")]
            return json.dumps({
                "error": f"no album named or numbered {ref!r}",
                "existing_albums": names,
                "hint": "Use one of the existing album names exactly."})
        from ..api.albums import album_analysis

        analysis = album_analysis(album_id=row["id"], conn=conn)
        member_ids = [r["sample_id"] for r in conn.execute(
            "SELECT sample_id FROM album_items WHERE album_id = ? "
            "ORDER BY position, sample_id LIMIT 24", (row["id"],))]
        return json.dumps({
            "album_id": row["id"], "name": row["name"], "origin": row["origin"],
            "summary": row["summary"], "category": row["category"],
            "count": analysis["count"],
            "measured": analysis["measured"],
            "sample_ids": member_ids,
            "note": "'measured' is counted/computed from stored data — cite it "
                    "as measurement. Outliers are the members farthest from "
                    "the album centroid: the first places to look for rare "
                    "scenarios inside the album.",
        })
    finally:
        conn.close()
