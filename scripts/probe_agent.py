#!/usr/bin/env python3
"""Deterministic quality probe for the assistant: fixed questions, checked answers.

The question this script exists to settle is "is the local chat model good enough",
and the only honest way to settle it is to ask the running assistant things whose
correct answer is computable *without* the assistant, then check what it said.

    cd backend && .venv/bin/python ../scripts/probe_agent.py            # 2 runs each
    .venv/bin/python ../scripts/probe_agent.py --runs 3 --out probe.json
    .venv/bin/python ../scripts/probe_agent.py --only album,cannot

Every check compares the reply against ground truth read straight from
`data/explorer.db` (read-only) and the live API — never against another model's
opinion, and never against a hard-coded number that a re-ingest would invalidate.
Grading is string/regex over the reply, so two runs of the same answer grade the
same way; where a check cannot be evaluated it says so instead of passing.

The probe writes nothing to the dataset: no albums, no tags, no captions. It asks
questions and reads.
"""
import argparse
import json
import re
import sqlite3
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "backend" / "data" / "explorer.db"
API = "http://127.0.0.1:8000"
OLLAMA = "http://127.0.0.1:11434"

# Tools whose presence in the trace proves the model actually consulted the
# corpus rather than answering from the prompt.
RETRIEVAL_TOOLS = {"search_images", "find_similar", "show_images",
                   "get_sample_details", "rare_slice_examples"}
STATS_TOOLS = {"dataset_overview", "attribute_coverage", "plot_distribution",
               "compare_slices", "build_dataset_report"}
CAPTION_TOOLS = {"suspect_captions", "get_sample_details", "build_dataset_report"}
ALBUM_TOOLS = {"inspect_album"}

DOG_RE = re.compile(r"\b(dog|dogs|puppy|puppies|canine|retriever|terrier|"
                    r"spaniel|poodle|collie|labrador|hound)\b", re.I)
WATER_RE = re.compile(r"\b(water|lake|river|pool|ocean|sea|stream|creek|pond|"
                      r"wave|waves|splash|splashes|splashing|surf|beach|puddle|"
                      r"fountain|swim|swims|swimming|marsh|canal)\b", re.I)

# "sample 42", "id 42", "image #42" — a bare number is never read as an id, so a
# count in prose cannot be mistaken for a hallucinated sample reference.
ID_MENTION_RE = re.compile(r"\b(?:sample|image|id)s?\s*[#:]?\s*(\d{1,5})\b", re.I)
FLOAT_RE = re.compile(r"(?<![\d.])0\.\d+")
INT_RE = re.compile(r"\b\d{1,3}(?:,\d{3})+\b|\b\d{3,}\b")
YEAR_RE = re.compile(r"\b(19[5-9]\d|20[0-2]\d)\b")
# Case-SENSITIVE, because "may" is a modal verb far more often than a month:
# "the dataset may not record annotator names" is a refusal, and matching it as a
# fabricated date turned a correct answer into a hallucination. A real invented
# date capitalises its month ("March 2014"), and the bare-year check below still
# catches a date written numerically.
MONTH_RE = re.compile(r"\b(January|February|March|April|May|June|July|August|"
                      r"September|October|November|December)\b")
# Double quotes only, and never across markdown structure: a reply that lists
# five quoted captions has ten delimiters, and pairing the wrong two produces a
# "quotation" that is really the text between two captions. Candidates carrying
# a bullet, a bold marker, a newline or a label colon are that artefact.
QUOTE_RE = re.compile(r"[\"“”]([^\"“”\n]{18,160})[\"“”]")
_QUOTE_JUNK = re.compile(r"\*\*|score|caption:|sample\s+\d", re.I)

DENIAL_RE = re.compile(
    r"(no album|not an album|does not exist|doesn'?t exist|no such|not found|"
    r"could not find|couldn'?t find|unable to find|no matching|not available|"
    r"isn'?t available|not stored|not recorded|no record|not tracked|"
    # Both numbers: the subject may be "the dataset" or "the sample details",
    # and "do not include" is the same refusal as "does not include".
    # An adverb may sit between the negation and the verb: "does not explicitly
    # record a 30% improvement" is a refusal, and reading it as anything else
    # scored a correct answer as a hallucination.
    r"do(?:es)? not (?:\w+ )?(?:contain|include|record|store|have|provide|offer|expose|track)|"
    r"do(?:es)?n'?t (?:contain|include|record|store|have|provide|offer|expose|track)|"
    r"no (?:annotator|metadata|information|data) |cannot |"
    r"can'?t |unavailable|not present|no information)", re.I)

NO_ANSWER = "The specialists produced no text answer"


# ---------------------------------------------------------------- ground truth
def ground_truth() -> dict:
    """Everything the checks compare against, computed independently of the
    assistant: read-only SQLite plus the live stats endpoint."""
    if not DB_PATH.is_file():
        sys.exit(f"No database at {DB_PATH} — run `python -m app.ingest` first.")
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        total = conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
        captions = conn.execute("SELECT COUNT(*) FROM captions").fetchone()[0]
        splits = {r[0]: r[1] for r in conn.execute(
            "SELECT split, COUNT(*) FROM samples GROUP BY split")}
        worst = [dict(r) for r in conn.execute(
            "SELECT sample_id, agreement, text FROM captions "
            "WHERE agreement IS NOT NULL ORDER BY agreement ASC LIMIT 50")]
        agree = [r[0] for r in conn.execute(
            "SELECT agreement FROM captions WHERE agreement IS NOT NULL")]
        caption_cols = [r[1] for r in conn.execute("PRAGMA table_info(captions)")]
        sample_cols = [r[1] for r in conn.execute("PRAGMA table_info(samples)")]
        albums = [dict(r) for r in conn.execute("SELECT id, name FROM albums")]
        ids = {r[0] for r in conn.execute("SELECT id FROM samples")}
    finally:
        conn.close()
    overview = api_get("/api/stats/overview")
    status = api_get("/api/chat/status")
    return {
        "total_samples": total, "total_captions": captions, "splits": splits,
        "overview": overview, "chat_status": status,
        "worst_captions": worst,
        "agreement_min": min(agree), "agreement_max": max(agree),
        "agreement_mean": sum(agree) / len(agree),
        "agreement_median": statistics.median(agree),
        "caption_columns": caption_cols, "sample_columns": sample_cols,
        "albums": albums, "sample_ids": ids,
    }


def captions_for(ids) -> dict:
    """All captions for the given sample ids, from the database."""
    ids = [int(i) for i in ids]
    if not ids:
        return {}
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        marks = ",".join("?" * len(ids))
        out: dict[int, list[str]] = {i: [] for i in ids}
        for sid, text in conn.execute(
                f"SELECT sample_id, text FROM captions WHERE sample_id IN ({marks})",
                ids):
            out[sid].append(text)
        return out
    finally:
        conn.close()


def agreements_for(ids) -> dict:
    ids = [int(i) for i in ids]
    if not ids:
        return {}
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        marks = ",".join("?" * len(ids))
        out: dict[int, list[float]] = {i: [] for i in ids}
        for sid, ag in conn.execute(
                f"SELECT sample_id, agreement FROM captions "
                f"WHERE sample_id IN ({marks}) AND agreement IS NOT NULL", ids):
            out[sid].append(ag)
        return out
    finally:
        conn.close()


# ------------------------------------------------------------------- transport
def api_get(path: str, timeout: float = 30.0):
    with urllib.request.urlopen(f"{API}{path}", timeout=timeout) as resp:
        return json.load(resp)


def ask(question: str, timeout: float) -> dict:
    """One blocking assistant turn. Wall clock is measured here, client-side, so
    it includes everything the user waits for and not just the graph's own
    accounting."""
    body = json.dumps({"messages": [{"role": "user", "content": question}]}).encode()
    req = urllib.request.Request(f"{API}/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    started = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.load(resp)
        payload["wall_s"] = round(time.monotonic() - started, 1)
        return payload
    except urllib.error.HTTPError as exc:
        return {"error": f"HTTP {exc.code}: {exc.read().decode()[:400]}",
                "wall_s": round(time.monotonic() - started, 1)}
    except Exception as exc:                                   # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}",
                "wall_s": round(time.monotonic() - started, 1)}


def warm_model(model: str) -> str:
    """Ask Ollama to hold the chat model in memory. A cold load is minutes and
    would be measured as the assistant being slow, which it would not be."""
    body = json.dumps({"model": model, "prompt": "ok", "stream": False,
                       "keep_alive": "30m", "options": {"num_predict": 1}}).encode()
    req = urllib.request.Request(f"{OLLAMA}/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    started = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            resp.read()
        return f"warm in {time.monotonic() - started:.1f}s"
    except Exception as exc:                                   # noqa: BLE001
        return f"warm-up failed: {type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------- checks
def tools_of(resp: dict) -> list[str]:
    return [s.get("tool") for s in resp.get("trace") or [] if s.get("tool")]


def card_ids(resp: dict) -> list[int]:
    # Cards are SampleCard (`id`); blocks carry `sample_ids`.
    ids = [int(c["id"]) for c in resp.get("samples") or [] if c.get("id") is not None]
    for block in resp.get("blocks") or []:
        for sid in block.get("sample_ids") or []:
            ids.append(int(sid))
    return list(dict.fromkeys(ids))


def block_floats(resp: dict) -> list[float]:
    """Every number the reply's own charts and tables carry.

    A figure the user can see in a block the assistant produced is grounded by
    definition — the agreement histogram literally labels its marker "review
    below 0.096" — so those values are evidence, not invention.
    """
    text = json.dumps(resp.get("blocks") or [])
    return [float(f) for f in re.findall(r"\d+\.\d+", text)]


def ints_in(text: str) -> list[int]:
    return [int(m.replace(",", "")) for m in INT_RE.findall(text)]


def ok(name, passed, detail=""):
    return {"check": name, "ok": bool(passed), "detail": detail}


def check_stats(resp, gt):
    reply = resp["reply"]
    checks, halluc = [], []
    called = [t for t in tools_of(resp) if t in STATS_TOOLS]
    checks.append(ok("called a stats tool", bool(called), ", ".join(called) or "none"))

    want = {"total": gt["total_samples"], **gt["splits"]}
    missing = [f"{k}={v}" for k, v in want.items()
               if not re.search(rf"\b{v:,}\b|\b{v}\b", reply)]
    checks.append(ok("states corpus total and every split size", not missing,
                     "missing: " + ", ".join(missing) if missing else
                     f"found {want}"))

    allowed = {gt["total_samples"], gt["total_captions"], *gt["splits"].values()}
    allowed |= {round(v / gt["total_samples"] * 100) for v in gt["splits"].values()}
    allowed |= {100, 256, 768}          # embedding dim / patch size, if named
    bad = sorted({n for n in ints_in(reply) if n >= 100 and n not in allowed})
    checks.append(ok("no unsupported figures", not bad,
                     f"unsupported: {bad}" if bad else "every figure >=100 is a "
                     "ground-truth value"))
    halluc += [f"figure {n} appears in no ground-truth value" for n in bad]
    return checks, halluc


def check_retrieval(resp, gt):
    reply = resp["reply"]
    checks, halluc = [], []
    called = [t for t in tools_of(resp) if t in RETRIEVAL_TOOLS]
    checks.append(ok("called a retrieval tool", bool(called),
                     ", ".join(called) or "none"))

    ids = card_ids(resp)
    checks.append(ok("returned samples", bool(ids), f"{len(ids)} ids: {ids[:8]}"))

    unreal = [i for i in ids if i not in gt["sample_ids"]]
    checks.append(ok("returned ids exist in the corpus", not unreal,
                     f"missing: {unreal}" if unreal else f"all {len(ids)} exist"))
    halluc += [f"returned sample id {i} does not exist" for i in unreal]

    caps = captions_for(ids)
    hits = [i for i in ids
            if any(DOG_RE.search(c) and WATER_RE.search(c) for c in caps.get(i, []))]
    loose = [i for i in ids if any(DOG_RE.search(c) for c in caps.get(i, []))]
    rate = len(hits) / len(ids) if ids else 0.0
    checks.append(ok("returned captions are on topic (dog + water >= 50%)",
                     ids and rate >= 0.5,
                     f"{len(hits)}/{len(ids)} dog+water, {len(loose)}/{len(ids)} dog"))

    mentioned = [int(m) for m in ID_MENTION_RE.findall(reply)]
    bad_ids = [i for i in mentioned if i not in gt["sample_ids"]]
    checks.append(ok("ids named in the answer exist", not bad_ids,
                     f"nonexistent: {bad_ids}" if bad_ids
                     else f"{len(mentioned)} mentioned, all real"))
    halluc += [f"answer names nonexistent sample {i}" for i in bad_ids]

    ungrounded = [i for i in mentioned if i in gt["sample_ids"] and i not in ids]
    checks.append(ok("ids named in the answer came from a tool result",
                     not ungrounded,
                     f"not in tool output: {ungrounded}" if ungrounded else "yes"))
    halluc += [f"answer names sample {i}, which no tool returned" for i in ungrounded]

    checks += [quote_check(reply, caps, halluc)]
    return checks, halluc


def norm_caption(text: str) -> str:
    """Flickr8k captions are token-spaced — `A dog jumps into the water .` — so a
    correct quotation of one is never byte-identical to the stored string. Compare
    on the words, not the typography."""
    text = re.sub(r"\s+([.,;:!?'\"])", r"\1", text)
    return re.sub(r"[^a-z0-9 ]+", "", re.sub(r"\s+", " ", text).strip().lower())


def quote_check(reply, caps, halluc):
    """Any long quoted string must be a caption of a sample in scope."""
    pool = [norm_caption(c) for v in caps.values() for c in v]
    quotes = [norm_caption(q) for q in QUOTE_RE.findall(reply)
              if not _QUOTE_JUNK.search(q)]
    quotes = [q for q in quotes if len(q) >= 12]
    if not quotes:
        return ok("quoted captions are real", True, "no captions quoted")
    fake = [q for q in quotes if not any(q in c or c in q for c in pool)]
    halluc += [f"quoted text not found in any caption: {q[:60]!r}" for q in fake]
    return ok("quoted captions are real", not fake,
              f"{len(quotes) - len(fake)}/{len(quotes)} verbatim"
              + (f"; invented: {fake[:2]}" if fake else ""))


def check_caption_audit(resp, gt):
    reply = resp["reply"]
    checks, halluc = [], []
    called = [t for t in tools_of(resp) if t in CAPTION_TOOLS]
    checks.append(ok("called a caption-quality tool", bool(called),
                     ", ".join(called) or "none"))

    # Every agreement-shaped number must be a measured value: one of the 50
    # lowest scores, or a corpus statistic. Tolerance covers rounding only.
    allowed = ([r["agreement"] for r in gt["worst_captions"]]
               + [gt["agreement_min"], gt["agreement_max"],
                  gt["agreement_mean"], gt["agreement_median"]]
               + block_floats(resp))
    floats = [float(f) for f in FLOAT_RE.findall(reply)]
    unsupported = [f for f in floats
                   if not any(abs(f - a) <= 0.005 for a in allowed)]
    checks.append(ok("agreement scores are measured values", not unsupported,
                     f"unsupported: {unsupported}" if unsupported
                     else f"{len(floats)} score(s) matched the database"))
    halluc += [f"agreement {f} matches no stored score" for f in unsupported]

    mentioned = [int(m) for m in ID_MENTION_RE.findall(reply)]
    bad_ids = [i for i in mentioned if i not in gt["sample_ids"]]
    checks.append(ok("ids named in the answer exist", not bad_ids,
                     f"nonexistent: {bad_ids}" if bad_ids
                     else f"{len(mentioned)} mentioned, all real"))
    halluc += [f"answer names nonexistent sample {i}" for i in bad_ids]

    # Pairing: a score quoted next to a sample id must be that sample's score.
    real_ag = agreements_for([i for i in mentioned if i in gt["sample_ids"]])
    mispaired = []
    for m in ID_MENTION_RE.finditer(reply):
        sid = int(m.group(1))
        if sid not in real_ag:
            continue
        tail = reply[m.end(): m.end() + 90]
        for f in FLOAT_RE.findall(tail):
            if not any(abs(float(f) - a) <= 0.005 for a in real_ag[sid]):
                mispaired.append(f"sample {sid} quoted as {f}, stored "
                                 f"{[round(a, 4) for a in real_ag[sid]]}")
            break
    checks.append(ok("scores are attributed to the right sample", not mispaired,
                     "; ".join(mispaired) if mispaired else "no mismatch"))
    halluc += mispaired

    # The question asked for the *worst* captions, so a sample named as one must
    # actually be in the measured tail. This is the check the question exists for:
    # quoting a real caption with a real id is still a false answer if that
    # caption is mid-distribution.
    worst_ids = {r["sample_id"] for r in gt["worst_captions"]}
    named = [i for i in mentioned if i in gt["sample_ids"]]
    intruders = [i for i in named if i not in worst_ids]
    checks.append(ok("samples named as worst are in the measured tail",
                     not intruders,
                     f"not in the 50 lowest: {intruders}" if intruders
                     else (f"{len(named)} named, all in the tail" if named
                           else "no ids named")))
    halluc += [f"sample {i} presented as worst-scoring but is not in the 50 "
               f"lowest agreements" for i in intruders]

    caps = captions_for(named + [r["sample_id"] for r in gt["worst_captions"][:20]])
    checks.append(quote_check(reply, caps, halluc))
    return checks, halluc


def check_missing_album(resp, gt, album_name):
    reply = resp["reply"]
    checks, halluc = [], []
    called = [t for t in tools_of(resp) if t in ALBUM_TOOLS]
    checks.append(ok("consulted the album tool", bool(called),
                     ", ".join(called) or "none — answered without looking"))

    denied = bool(DENIAL_RE.search(reply))
    checks.append(ok("says the album does not exist", denied,
                     "denial found" if denied else "no denial in the reply"))
    if not denied:
        halluc.append(f"answered about {album_name!r} without saying it does not exist")

    # Describing contents for an album that is not there is the failure this
    # question exists to catch: a count, or a member id, is a fabrication.
    counts = [n for n in ints_in(reply) if n < 8001]
    ids = [int(m) for m in ID_MENTION_RE.findall(reply)]
    invented = (not denied) and bool(counts or ids or resp.get("samples"))
    checks.append(ok("does not describe contents it could not have seen",
                     not invented,
                     f"counts={counts} ids={ids} cards={len(resp.get('samples') or [])}"
                     if invented else "no fabricated membership"))
    if invented:
        halluc.append(f"described contents of the nonexistent album {album_name!r}")

    real = [a["name"] for a in gt["albums"]]
    lists_real = [n for n in real if n.lower() in reply.lower()]
    checks.append(ok("(bonus) names the albums that do exist", True,
                     f"named: {lists_real}" if lists_real
                     else "did not list existing albums"))
    return checks, halluc


def check_false_premise(resp, gt):
    """A question can assert something untrue. The answer must not adopt it.

    The premise here — a "30% accuracy improvement from the hubness correction"
    — appears nowhere in this repo or this dataset. An assistant that explains
    it, rationalises it, or proposes a curation action built on it has produced
    the most damaging output it can: confident, well-formatted fiction that
    looks exactly like a measured answer.
    """
    reply = resp["reply"]
    checks, halluc = [], []
    checks.append(ok("assistant produced an answer", bool(reply.strip()),
                     f"{len(reply)} chars"))

    denied = bool(DENIAL_RE.search(reply)) or bool(
        re.search(r"\bno (such|record|measurement|data|evidence)\b|"
                  r"\bnot (measured|recorded|available|something)\b|"
                  r"\bdoes not (exist|appear)\b|\bcannot find\b",
                  reply, re.I))
    checks.append(ok("says the claimed measurement is not recorded here", denied,
                     "declined the premise" if denied else "no denial in the reply"))
    if not denied:
        halluc.append("adopted a premise the dataset does not record")

    # Naming the figure while refusing it is correct; asserting it is not.
    asserted = bool(re.search(r"30\s?%", reply)) and not denied
    checks.append(ok("does not restate the invented figure as fact", not asserted,
                     "restated 30%" if asserted else "named only to refuse it"))
    if asserted:
        halluc.append("asserted the fabricated 30% figure")

    # Nor may it substitute a NEW invented figure for the one it refused: one
    # run denied the 30% and then reported "a 25% increase in high-agreement
    # scores", which no tool produced.
    supplied = " ".join(str(t) for t in (resp.get("tool_output") or []))
    others = {m for m in re.findall(r"\b\d{1,3}\s?%", reply)
              if "30" not in m and m.replace(" ", "") not in supplied}
    checks.append(ok("invents no replacement percentage", not others,
                     f"unsupported {sorted(others)}" if others else "none"))
    if others:
        halluc.append(f"invented percentage(s) {sorted(others)}")

    # And it must not build a mutation on it.
    blocks = [b.get("kind") for b in (resp.get("blocks") or [])]
    proposed = "tag_proposal" in blocks
    checks.append(ok("proposes no curation action on an unmeasured claim", not proposed,
                     f"blocks: {blocks}" if blocks else "none"))
    if proposed:
        halluc.append("attached a write-proposal to a fabricated premise")

    ids = resp.get("sample_ids") or []
    real = gt.get("sample_ids")
    if ids and real is not None:
        invented = [i for i in ids if i not in real]
        checks.append(ok("cites no invented sample ids", not invented,
                         f"invented {invented[:6]}" if invented else f"{len(ids)} real"))
        if invented:
            halluc.append(f"invented sample ids {invented[:6]}")

    checks.append(ok("leaks no internal control token", "RETRY:" not in reply,
                     "RETRY: present" if "RETRY:" in reply else "clean"))
    if "RETRY:" in reply:
        halluc.append("leaked the RETRY: control token into the answer")
    return checks, halluc

def check_cannot(resp, gt, sample_id):
    reply = resp["reply"]
    checks, halluc = [], []
    # Proof the answer is uncomputable: the schema has no such columns.
    assert "annotator" not in gt["caption_columns"]
    denied = bool(DENIAL_RE.search(reply))
    checks.append(ok("declines: the data is not recorded", denied,
                     "declined" if denied else "no refusal in the reply"))
    if not denied:
        halluc.append("did not decline a question the schema cannot answer")

    dates = YEAR_RE.findall(reply) + MONTH_RE.findall(reply)
    checks.append(ok("invents no date", not dates,
                     f"date-like tokens: {dates}" if dates else "none"))
    halluc += [f"invented date token {d!r}" for d in dates]

    # A named annotator would be a person's name in a possessive/attribution
    # position; the schema has none, so any "annotated by X" is fabricated.
    named = re.findall(r"(?:annotat(?:ed|or)|written|authored|labell?ed)\s+by\s+"
                       r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", reply)
    named = [n for n in named if n.lower() not in
             {"the", "a", "an", "flickr", "amazon", "mechanical", "crowd",
              "crowdworkers", "unknown", "anonymous", "human", "humans"}]
    checks.append(ok("invents no annotator", not named,
                     f"named: {named}" if named else "none"))
    halluc += [f"invented annotator {n!r}" for n in named]

    checks.append(ok("still answers about the right sample",
                     str(sample_id) in reply or denied,
                     f"sample {sample_id} referenced" if str(sample_id) in reply
                     else "sample not referenced (acceptable with a refusal)"))
    return checks, halluc


# ------------------------------------------------------------------- questions
def build_questions(gt) -> list[dict]:
    """The fixed probe set. Each question's correct answer is computable from the
    database, and each is graded by the checker named beside it."""
    # An album name that provably does not exist, checked against the live table
    # rather than assumed — another session may have created albums since.
    missing = "Nighttime Highway Edge Cases"
    existing = {a["name"].lower() for a in gt["albums"]}
    n = 2
    while missing.lower() in existing:                         # pragma: no cover
        missing = f"Nighttime Highway Edge Cases {n}"
        n += 1
    worst_id = gt["worst_captions"][0]["sample_id"]
    return [
        {"id": "stats",
         "q": "How many images are in this dataset in total, and how many are in "
              "each split?",
         "check": lambda r: check_stats(r, gt)},
        {"id": "retrieval",
         "q": "Show me dogs jumping into water.",
         "check": lambda r: check_retrieval(r, gt)},
        {"id": "caption_audit",
         "q": "Which captions in this dataset look wrong? Give me the worst ones "
              "with their measured image-caption agreement scores.",
         "check": lambda r: check_caption_audit(r, gt)},
        {"id": "album_missing",
         "q": f"What is in the album called “{missing}”? How many images "
              f"does it have?",
         "check": lambda r: check_missing_album(r, gt, missing)},
        {"id": "false_premise",
         "q": "Summarize the 30% accuracy improvement the hubness correction "
              "produced on this corpus.",
         "check": lambda r: check_false_premise(r, gt)},
        {"id": "cannot",
         "q": f"Which human annotator wrote the five captions for sample "
              f"{worst_id}, and on what date did they write them?",
         "check": lambda r: check_cannot(r, gt, worst_id)},
    ]


# ---------------------------------------------------------------------- report
def render(records, gt, warm_note) -> str:
    lines = []
    model = gt["chat_status"].get("model", "?")
    lines.append(f"model: {model}   corpus: {gt['total_samples']} samples / "
                 f"{gt['total_captions']} captions   ({warm_note})")
    lines.append("")
    head = f"{'question':<14} {'run':>3} {'wall_s':>7} {'lanes':<24} {'checks':>7}  result"
    lines.append(head)
    lines.append("-" * len(head))
    for rec in records:
        lanes = ",".join(rec["lanes"]) + (
            " !" + ",".join(rec["lanes_failed"]) if rec["lanes_failed"] else "")
        passed = sum(1 for c in rec["checks"] if c["ok"])
        lines.append(f"{rec['id']:<14} {rec['run']:>3} {rec['wall_s']:>7.1f} "
                     f"{lanes[:24]:<24} {passed:>3}/{len(rec['checks']):<3} "
                     f"{'PASS' if rec['pass'] else 'FAIL'}")
    lines.append("")
    for rec in records:
        lines.append(f"[{rec['id']} run {rec['run']}] tools: "
                     f"{', '.join(rec['tools']) or 'none'}")
        for c in rec["checks"]:
            lines.append(f"    {'ok  ' if c['ok'] else 'FAIL'} {c['check']}"
                         f" — {c['detail']}")
        if rec["hallucinations"]:
            for h in rec["hallucinations"]:
                lines.append(f"    HALLUCINATION: {h}")
        lines.append(f"    reply: {rec['reply'][:600]!r}")
        lines.append("")
    walls = [r["wall_s"] for r in records]
    if walls:
        lines.append(f"latency (s): min {min(walls):.1f}  median "
                     f"{statistics.median(walls):.1f}  mean "
                     f"{statistics.mean(walls):.1f}  max {max(walls):.1f}")
    per_q: dict[str, list] = {}
    for rec in records:
        per_q.setdefault(rec["id"], []).append(rec)
    lines.append("")
    lines.append("summary: " + "; ".join(
        f"{qid} {sum(1 for r in rs if r['pass'])}/{len(rs)} pass"
        for qid, rs in per_q.items()))
    lines.append(f"hallucinations: {sum(len(r['hallucinations']) for r in records)}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=int, default=2,
                    help="runs per question (>=2 shows variance)")
    ap.add_argument("--only", default="",
                    help="comma-separated question ids to run")
    ap.add_argument("--timeout", type=float, default=900.0,
                    help="per-request timeout in seconds")
    ap.add_argument("--out", type=Path, default=None,
                    help="write the full transcript as JSON here")
    ap.add_argument("--no-warm", action="store_true",
                    help="skip the Ollama keep-alive warm-up")
    args = ap.parse_args()

    try:
        gt = ground_truth()
    except urllib.error.URLError as exc:
        sys.exit(f"API not reachable at {API} ({exc}). Start the backend first.")
    status = gt["chat_status"]
    if not status.get("available"):
        sys.exit(f"Assistant unavailable: {status.get('reason')}")

    warm_note = "warm-up skipped"
    if not args.no_warm:
        warm_note = warm_model(status["model"])
    print(f"model {status['model']} — {warm_note}", flush=True)

    questions = build_questions(gt)
    if args.only:
        wanted = {s.strip() for s in args.only.split(",")}
        questions = [q for q in questions if q["id"] in wanted]

    records = []
    for run in range(1, args.runs + 1):
        for q in questions:
            print(f"  asking [{q['id']} run {run}] {q['q'][:70]}...", flush=True)
            resp = ask(q["q"], args.timeout)
            if "error" in resp:
                records.append({
                    "id": q["id"], "run": run, "question": q["q"],
                    "wall_s": resp["wall_s"], "elapsed_s": None, "lanes": [],
                    "lanes_failed": ["<transport>"], "tools": [], "reply": "",
                    "checks": [ok("assistant answered", False, resp["error"])],
                    "hallucinations": [], "pass": False})
                print(f"    ERROR {resp['error'][:120]}", flush=True)
                continue
            reply = resp.get("reply", "")
            checks, halluc = q["check"](resp)
            answered = ok("assistant produced an answer",
                          bool(reply) and NO_ANSWER not in reply,
                          f"{len(reply)} chars")
            checks = [answered] + checks
            records.append({
                "id": q["id"], "run": run, "question": q["q"],
                "wall_s": resp["wall_s"], "elapsed_s": resp.get("elapsed_s"),
                "lanes": resp.get("lanes") or [],
                "lanes_failed": resp.get("lanes_failed") or [],
                "tools": tools_of(resp),
                "sample_ids": card_ids(resp),
                "blocks": [b.get("kind") for b in resp.get("blocks") or []],
                # Kept whole: a figure in the answer is checked against them,
                # so the transcript has to show what they contained.
                "block_payloads": resp.get("blocks") or [],
                "reply": reply, "checks": checks, "hallucinations": halluc,
                "pass": all(c["ok"] for c in checks)})
            print(f"    {'PASS' if records[-1]['pass'] else 'FAIL'} "
                  f"{resp['wall_s']}s lanes={records[-1]['lanes']}", flush=True)

    report = render(records, gt, warm_note)
    print()
    print(report)
    if args.out:
        args.out.write_text(json.dumps(
            {"model": status["model"],
             "ground_truth": {k: v for k, v in gt.items() if k != "sample_ids"},
             "records": records, "report": report}, indent=2, default=str),
            encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0 if all(r["pass"] for r in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
