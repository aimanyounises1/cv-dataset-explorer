# Technical walkthrough

How this system is built, layer by layer, with the actual SQL and the reasons
behind it. `docs/CAPABILITIES.md` lists *what* it does; this explains *how* and
*why*. Read `docs/DESIGN.md` for retrieval trade-offs and `docs/AGENTS.md` for
the agent layer.

Every number here was measured on the running system (8,000 images, 40,000
captions, 12.9 MB database) and the query plans are real `EXPLAIN QUERY PLAN`
output, not illustrations.

---

## 1. The shape of the thing

```
 Browser ──── /api/* ────▶ FastAPI ──┬──▶ SQLite (WAL)      metadata, captions,
    │                                │      + FTS5          tags, attributes, axes
    │  /media/*                      │
    │                                ├──▶ NumPy .npy        8000 × 768 embeddings,
    ▼                                │                      exact cosine
 static images                       ├──▶ SigLIP 2          text→vector at query time
                                     │      (torch/MPS)
                                     └──▶ Ollama           optional: agents, VLM tags
```

One process, one file database, one directory of images. Nothing is networked
except the optional local Ollama on `:11434`.

**The layering rule.** Browsing, keyword search and statistics require only
SQLite. Every ML capability is an optional layer that reports its own
availability and degrades to a message explaining the command to run. This is
not politeness — it is what makes the app installable and demoable on a machine
that has not spent 20 minutes computing embeddings.

### A request, end to end

`GET /api/search?q=dog+in+snow&mode=hybrid&split=train&difficulty_min=8`

1. **`deps.build_filters`** composes one `WHERE` clause from split, tag, VLM tag,
   attribute facet, four axis ranges and an optional id list. It returns
   `(where_sql, params)` — never a string with values interpolated.
2. **`filtered_id_set`** runs that clause once and returns a Python `set[int]`.
   This is the candidate mask.
3. **Semantic path** encodes the query text with SigLIP, then scores it against
   the embedding matrix *restricted to the mask* — the filter is applied before
   top-k, never after.
4. **Keyword path** runs FTS5 BM25 with the same `WHERE` clause spliced in as
   `AND`, again before `LIMIT`.
5. **Fusion** combines the two by reciprocal rank.
6. The window `[offset, offset+top_k)` is sliced out, sample rows and captions
   are fetched in **two** queries (not 60), and each card is annotated with which
   path found it and at what rank.

The invariant worth naming: **filters are always applied inside ranking, never
after it.** Filtering a top-60 list down to 3 results, when 500 matching images
existed at rank 61+, is the classic version of this bug.

---

## 2. Data layer

### Schema

```sql
CREATE TABLE samples (
    id INTEGER PRIMARY KEY,
    dataset TEXT NOT NULL,          -- adapter name; the schema is not Flickr8k-specific
    filename TEXT NOT NULL UNIQUE,
    split TEXT NOT NULL,
    width INTEGER, height INTEGER, filesize INTEGER,
    umap_x REAL, umap_y REAL,       -- 2-D projection, for the map only
    cluster INTEGER,                -- k-means label, computed in 768-D
    caption_consistency REAL,       -- mean pairwise similarity of a sample's 5 captions
    legibility INTEGER, rarity INTEGER,   -- the four difficulty axes,
    difficulty INTEGER, clutter INTEGER,  -- 0-10 percentile buckets
    axis_detail TEXT                -- JSON: the raw components behind each axis
);

CREATE TABLE captions (
    id INTEGER PRIMARY KEY,
    sample_id INTEGER NOT NULL REFERENCES samples(id) ON DELETE CASCADE,
    idx INTEGER NOT NULL,           -- 0-4, the caption's position
    text TEXT NOT NULL,
    agreement REAL                  -- SigLIP cosine(image, this caption)
);

CREATE TABLE attributes (           -- zero-shot labels
    sample_id INTEGER NOT NULL REFERENCES samples(id) ON DELETE CASCADE,
    grp TEXT NOT NULL,              -- 'time_of_day', 'setting', 'environment', 'main_subject'
    label TEXT NOT NULL,
    confidence REAL NOT NULL,
    PRIMARY KEY (sample_id, grp)    -- one label per group: single-label by construction
);

CREATE TABLE tags (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE);
CREATE TABLE sample_tags (sample_id INTEGER, tag_id INTEGER, PRIMARY KEY (sample_id, tag_id));
CREATE TABLE vlm_tags   (sample_id INTEGER, tag TEXT, PRIMARY KEY (sample_id, tag));
CREATE TABLE saved_views(id INTEGER PRIMARY KEY, name TEXT UNIQUE,
                         query_string TEXT NOT NULL, created_at TEXT NOT NULL);

-- The 2026-07-28 workspace wave, all additive (CREATE TABLE IF NOT EXISTS):
CREATE TABLE albums (
    id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE,
    summary TEXT, category TEXT, notes TEXT,
    cover_sample_id INTEGER,              -- NULL: cover falls back to first item
    origin TEXT NOT NULL DEFAULT 'manual',-- 'manual' | 'tag' ('agent' reserved)
    position INTEGER NOT NULL DEFAULT 0,  -- ordering among albums
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE album_items (
    album_id INTEGER NOT NULL, sample_id INTEGER NOT NULL,
    position INTEGER NOT NULL,            -- ordering within the album
    added_at TEXT NOT NULL, PRIMARY KEY (album_id, sample_id));
CREATE TABLE activity_events (            -- append-only workspace trail
    id INTEGER PRIMARY KEY, kind TEXT NOT NULL,
    payload TEXT NOT NULL,                -- opaque JSON, size-capped at the API
    created_at TEXT NOT NULL);
CREATE TABLE annotations (                -- regions as rows; images stay immutable
    id INTEGER PRIMARY KEY, sample_id INTEGER NOT NULL,
    kind TEXT NOT NULL,                   -- 'rect' | 'polygon'
    geometry TEXT NOT NULL,               -- JSON, normalized 0..1 coordinates
    label TEXT, created_at TEXT NOT NULL);
```

Four decisions worth defending:

**`attributes` has `PRIMARY KEY (sample_id, grp)`.** A sample gets exactly one
label per group. That is a real constraint, not laziness: the labels come from a
softmax over a group's label bank, so "night *and* day" is not a state the
classifier can produce. Making it multi-label later means dropping that key and
changing every read — the constraint is the documentation.

**Axes are stored, not computed on read.** They are percentile ranks over the
whole corpus, so computing one requires all 8,000 values. Storing them makes
`difficulty >= 8` an indexed range scan instead of a full re-rank per request.
The cost is that they are stale after ingesting more images — stated in the
README rather than hidden.

**`axis_detail` is JSON in a TEXT column.** It holds the raw components (blur,
luminance, agreement, IDF) behind each score. It is only ever read whole, for one
sample, to explain a number in the UI. A normalised table would buy queryability
nobody has asked for and cost a join on every detail view.

**`saved_views.query_string` is opaque.** It stores the URL query string, not a
parsed filter object. So a saved view keeps working when the UI grows a filter
the backend has no column for. The price: the server cannot validate or migrate
one, and a view saved against a removed filter degrades silently.

### Indexes, and what the planner actually does

```sql
CREATE INDEX idx_samples_split        ON samples(split);
CREATE INDEX idx_samples_difficulty   ON samples(difficulty);   -- one per axis
CREATE INDEX idx_samples_rarity       ON samples(rarity);
CREATE INDEX idx_samples_legibility   ON samples(legibility);
CREATE INDEX idx_samples_clutter      ON samples(clutter);
CREATE INDEX idx_attributes_grp_label ON attributes(grp, label);
CREATE INDEX idx_captions_sample      ON captions(sample_id);
CREATE INDEX idx_vlm_tags_tag         ON vlm_tags(tag);
```

`idx_attributes_grp_label` is deliberately a **composite in that order**, because
every read is either "all labels in this group" or "this exact (group, label)" —
never "this label across groups". Coverage confirms it:

```
### attribute coverage
SCAN attributes USING COVERING INDEX idx_attributes_grp_label     -- 1.12 ms
```

*Covering* means the index alone answered the query; the table was never touched.

### WAL, and one connection per request

```python
conn = sqlite3.connect(config.DB_PATH, check_same_thread=False)
conn.execute("PRAGMA journal_mode=WAL")
```

WAL lets readers proceed during a write, which matters because ingest and
analysis run for minutes while the API stays up. `check_same_thread=False` is
needed because FastAPI serves sync endpoints from a thread pool — and the
dependency hands out **a fresh connection per request**, which is also what makes
the temp-table trick in the next section safe.

### FTS5, external content

```sql
CREATE VIRTUAL TABLE captions_fts USING fts5(
    text, content='captions', content_rowid='id', tokenize='porter unicode61');
```

`content='captions'` means the index stores no copy of the text — it points at
the base table by rowid. On 40,000 captions that is the difference between a
12.9 MB database and a noticeably larger one, and it removes any chance of the
two copies disagreeing.

The catch, and it is a sharp one: **there are no sync triggers.** Inserts are
done explicitly during ingest, and a delete requires the documented
`'delete'`-command protocol before removing the base row. Deleting from
`captions` alone leaves the index pointing at a row that no longer exists, and
FTS5 will happily return it. This bit us for real: three smoke-test fixture rows
once leaked into the live database and had to be removed through that protocol.

`porter` gives stemming, so *"jumping"* matches *"jump"*. `unicode61` gives
sane tokenisation of accented text.

---

## 3. The queries

### The composer

Every filtered read goes through one function, `deps.build_filters`. There is no
second place that builds a `WHERE` clause, which is why adding
`max_agreement` recently took one edit and reached `/samples`, `/search` and
`/export` at once.

```python
for axis, (lo, hi) in (axes or {}).items():
    if axis not in AXES:          # never interpolate an unvalidated identifier
        continue
    if lo is not None:
        clauses.append(f"s.{axis} >= ?")   # identifier interpolated, value bound
        params.append(lo)
```

Note the split: the **column name** is interpolated (SQL has no parameter form
for identifiers) but only after a whitelist check against `db.AXES`, while the
**value** is always bound. The same pattern guards sorting, via `SORT_KEYS`.
Every other value in the system is a bound parameter.

### Gallery page

```sql
SELECT s.* FROM samples s
 WHERE s.split = ?
   AND s.difficulty >= ?
   AND s.id IN (SELECT sample_id FROM attributes WHERE grp = ? AND label = ?)
 ORDER BY (s.difficulty IS NULL), s.difficulty DESC, s.id
 LIMIT 60 OFFSET 0;
```
```
SEARCH s USING INDEX idx_samples_split (split=? AND rowid=?)
LIST SUBQUERY 1
  SEARCH attributes USING INDEX idx_attributes_grp_label (grp=? AND label=?)
  CREATE BLOOM FILTER
USE TEMP B-TREE FOR ORDER BY                                      -- 0.58 ms
```

Two details:

`ORDER BY (s.difficulty IS NULL), s.difficulty DESC, s.id` — the leading
expression sorts unscored samples **last in both directions**. A NULL axis means
"not measured", which is not "measured as zero" and must never head a ranking.
The trailing `s.id` makes the order total, so paging cannot repeat or skip a row
when scores tie.

The `USE TEMP B-TREE FOR ORDER BY` is the planner declining to use
`idx_samples_difficulty`, because it already narrowed by split. At this size that
is the right call; it is the first thing to revisit at 10× the data.

### Keyword ranking

```sql
SELECT c.sample_id AS sid, MIN(rank) AS best, c.text
  FROM captions_fts f
  JOIN captions c ON c.id = f.rowid
  JOIN samples  s ON s.id = c.sample_id
 WHERE captions_fts MATCH ? AND s.split = ?
 GROUP BY c.sample_id ORDER BY best LIMIT 300;
```
```
SCAN f VIRTUAL TABLE INDEX 0:M1
SEARCH c USING INTEGER PRIMARY KEY (rowid=?)
SEARCH s USING INTEGER PRIMARY KEY (rowid=?)
USE TEMP B-TREE FOR GROUP BY / ORDER BY                            -- 1.33 ms
```

`rank` is FTS5's BM25 (negative; more negative is better), so `MIN(rank)`
per sample means *"the best-matching of this image's five captions"* — an image
is as relevant as its strongest caption, not its average one.

The filter is inside the `WHERE`, so it applies **before** `LIMIT 300`. That is
the whole point.

### Threshold → sample set

```sql
SELECT COUNT(*) FROM samples s
 WHERE s.id IN (SELECT sample_id FROM captions
                 WHERE agreement IS NOT NULL AND agreement <= ?);
```
```
SEARCH s USING INTEGER PRIMARY KEY (rowid=?)
LIST SUBQUERY 1
  SCAN captions                                                    -- 2.22 ms
  CREATE BLOOM FILTER
```

This is the Quality page's brush, expressed once and reused by `/samples`,
`/search` and `/export` so a triage selection means the same thing everywhere.

**`SCAN captions` is a full table scan** — there is no index on
`captions.agreement`. At 40,000 rows it costs 2.2 ms, which is why it has not
been added: an index that is not needed is a write cost and a size cost. It is
the second thing to revisit at 10× the data, and it is recorded here rather than
discovered later.

### The inversion: `/api/describe`

Every other endpoint runs the arrow forwards — given a filter, return the
samples. This runs it backwards: given any selection the product can make (a
lasso, a quality brush, a pasted id list, a search, a saved view), what
characterises it relative to the corpus. It reuses `build_filters` verbatim, so
it works on every selection idiom without translating anything.

The arithmetic is a lift — share-in-set over share-in-corpus — and lift alone is
a machine for producing confident nonsense on small sets: 3 of 20 samples being
"night" is a 6x multiplier and also three photographs. So every facet is tested
before it is reported, and the raw count travels with the multiplier.

The test is **hypergeometric, not binomial**, and the difference is not pedantry:

```python
p = big_k / big_n
fpc = (big_n - n) / (big_n - 1)      # finite-population correction
var = n * p * (1 - p) * fpc
z = (k - n * p) / sqrt(var)
```

The set is a *subset* of the corpus, drawn without replacement. Without the
correction, filtering to `split=train` — 6,000 of 8,000 samples — would make
every facet look wildly significant purely because *n* is large, when a set
holding three quarters of the corpus cannot differ much from it. With it,
`?split=train` correctly reports **nothing**, while `?legibility_min=9` reports
`night ×5.18, z=30.6` — which independently confirms the legibility axis measures
what it claims.

Two things the panel deliberately does **not** report. Facets from a group the
caller already filtered by are dropped entirely, because `attributes` is
`PRIMARY KEY (sample_id, grp)` — one label per group per sample — so asking for
the night images and being told they are `night ×20.41` is true, useless, and,
being the largest lift on the page, sits at the top pushing the real findings
down. The same constraint forces every *sibling* label to exactly zero, which is
why filtering by `setting:indoor` used to fill the under-represented column with
`outdoor ×0.00`, `day ×0.00`, `dusk ×0.00`. Suppression is by group, not by
label, and it clears four junk rows out of the panel's twelve slots.

### Facets compose, and the drill has to honour that

Each row is counted **inside the current selection**: on the 392 night images,
`setting:indoor ×3.07` means 223 of *those* 392, not 223 of the corpus. So its
drill-through has to *narrow* the selection rather than replace it — which
requires `attr` to be repeatable end to end:

```
?attr=time_of_day:night                    392 images
?attr=setting:indoor                     1,483 images
?attr=time_of_day:night&attr=setting:indoor  223 images   ← the row's own number
```

This did not work, and failed silently in both directions. `attr` was a bare
`str`, so FastAPI bound only the *last* occurrence: a URL naming two facets
returned one of them, and swapping the order returned a different set (706 vs
392 for night/indoors). Nothing errored — the address bar and the result count
simply disagreed. With no way to express an intersection, the describe panel's
drill dropped the selection instead, so a row advertising 223 images opened a set
of 1,483, **6.6× what it claimed**, with the number the user clicked appearing
nowhere on the page they arrived at.

Composition is now the contract on every path that builds a set: `/api/samples`,
`/api/search` (GET and POST), `/api/describe`, and `/api/export` — whose manifest
records both facets, so the file cannot misdescribe the query that produced it.
`build_filters` accepts a bare string or a list and emits one `EXISTS` clause per
facet, keeping the older single-facet callers working unchanged. Contradictory
facets (`time_of_day:night` and `time_of_day:day`) now correctly return **zero**
rather than silently resolving to whichever came second.

In the UI the attribute dropdown became an *add* control rather than a
current-value one — a single-valued `<select>` cannot show "night AND indoor",
and pretending otherwise is what let the URL and the gallery disagree. Applied
facets live in the selection rail as one chip each, removable individually, so
peeling `night` off `night + indoor` widens 223 → 1,483 instead of clearing both.

### Leakage: why the answer is a curve

`/api/stats/leakage` reports held-out images with a near-duplicate in training —
the failure mode with the largest documented effect on reported accuracy that
embeddings alone can detect. The full 8,000² scan costs **0.2 s**.

It returns a *ladder*, not a number, because the answer moves violently with the
cut:

| cosine | pairs | cross-split | contaminated held-out |
|---|---|---|---|
| 0.90 | 2,458 | 1,054 | **241 (12.05%)** |
| 0.92 | 774 | 344 | 118 (5.90%) |
| 0.95 | 46 | 22 | 16 (0.80%) |
| 0.97 | 4 | 0 | 0 |

A headline figure at a hard-coded threshold would be an arbitrary choice wearing
the costume of a measurement. Inspecting the pairs at 0.90 shows they are
near-identical frames from single photo shoots — the same frozen waterfall, the
same abseiler, the same egret — split across train and test.

This also forced a correction elsewhere: `duplicate_pairs` capped at 200, which
is right for a gallery and false for a count. "200 pairs" when there are 2,458 is
not a smaller answer. Counting and listing are now different methods.

### Avoiding N+1

```sql
SELECT sample_id, MIN(idx) AS mi, text FROM captions
 WHERE sample_id IN (?,?,?,…) GROUP BY sample_id;                  -- 0.03 ms
```

One query for all 60 cards on a page, not one per card. The same shape appears
for tags and rows throughout — the codebase has no per-row query in a loop.

### The 32,766 problem

The id-list filter accepts up to 60,000 pasted entries, but SQLite binds each
`IN (?, ?, …)` entry as a host parameter and the default ceiling
(`SQLITE_LIMIT_VARIABLE_NUMBER`) is **32,766**. A list of 40,000 therefore fails
with *"too many SQL variables"* rather than working. Reproduced: 32,000 entries
succeeded, 40,000 raised.

Above `ID_PARAM_LIMIT = 10_000` the entries go into a per-connection temp table
instead:

```sql
CREATE TEMP TABLE _id_filter (entry TEXT PRIMARY KEY);
-- then the predicate costs zero host parameters:
(CAST(s.id AS TEXT) IN (SELECT entry FROM _id_filter)
 OR s.filename    IN (SELECT entry FROM _id_filter))
```

60,000 entries now execute with **0** host parameters. It is safe because
`get_conn` hands out a fresh connection per request, so a temp table lives
exactly as long as the request that made it.

One more trap, found by testing rather than reasoning: `"٣".isdigit()` is `True`
in Python and `int("٣")` returns `3`, so an Arabic-Indic numeral in a pasted list
was silently coerced into a *different sample's* id. The parser is now
`.isascii()`-guarded, and a non-ASCII numeral is treated as a filename, where it
matches nothing.

---

## 4. Retrieval

### Semantic

Query text → SigLIP 2 text encoder → 768-d unit vector → dot product against the
whole matrix, masked to the allowed set:

```python
scores = embeddings @ qvec          # (8000, 768) @ (768,) -> (8000,)
scores[~mask] = -inf
top = np.argpartition(-scores, k)[:k]
```

**Exact brute force, no ANN index.** Measured: the matrix is 8,000 × 768
float32 = **24.6 MB**, a full scan takes **0.18 ms**, and the SigLIP text encode
that must happen first takes **7-8 ms** on MPS. The search is therefore ~2% of
the work of the query it belongs to, and an ANN index would add a dependency, a
build step and approximate recall to remove it. `docs/DESIGN.md` states where
that stops being true — **~400k vectors**, measured by extrapolating the scan
against the encode, not guessed.

Vectors are L2-normalised at write time, so cosine similarity *is* the dot
product and no per-query normalisation is needed.

### Keyword

FTS5 BM25 over captions, plus exact VLM-tag matches appended after caption hits.

### Hybrid: reciprocal rank fusion

```python
for rank, sid in enumerate(ranked_list):
    fused[sid] += 1.0 / (RRF_K + rank + 1)     # RRF_K = 60
```

RRF combines **ranks, not scores** — deliberately. A SigLIP cosine and a BM25
score live on different scales with different distributions; normalising them
into comparability requires assumptions neither one supports. Ranks are ordinal
and always comparable.

`k = 60` is the value Cormack et al. (SIGIR 2009) fixed "during a pilot
investigation" — a convention, not a tuned optimum for this corpus. It is
therefore configurable (`CVDE_RRF_K`) and reported with every fused response.

### Paging is a hard horizon

Both rankings are taken to exactly `SEARCH_DEPTH = 300` and fused **once**, then
the requested window is sliced. Fusing to `offset + page_size` instead would
re-rank on every page — measured, before this was fixed, as **4 duplicate images
across adjacent pages**, because the ranking past row 300 was being recomputed
against a different candidate pool. A ranking is only defined as deep as it was
computed, so results beyond it are not offered and the UI says so.

### Measured accuracy

Caption→image over 1,000 held-out captions, each query caption excluded from its
own candidate pool:

| mode | R@1 | R@5 | R@10 | MRR@10 | median rank | mean candidates | empty queries |
|---|---|---|---|---|---|---|---|
| semantic | 55.2% | 79.0% | 86.4% | 0.6544 | 1 | 8,000 | 0.0% |
| keyword | 4.2% | 5.4% | 5.8% | 0.0467 | > 10 | 2.1 | 85.3% |
| hybrid | 56.0% | 79.1% | 86.3% | 0.6562 | 1 | 8,002 | 0.0% |

**These numbers are not comparable to the ones this table carried before**, and
the difference is not only the two ranking changes below it. Holding out a bank of
captions for the hubness estimate changed which captions the benchmark draws, so
the evaluated sample is different — a row-by-row diff against the old table
measures two things at once and means nothing. The controlled comparison is the
offline A/B in `python -m app.ml.hubness`, which holds the sample fixed: **MRR
0.6280 → 0.6366** (sd 0.0020 over three bank draws), **R@1 53.2 → 53.9**, **R@10
84.1 → 85.3**.

Honest about the significance, because the two halves disagree: a paired
bootstrap on MRR gives a 95% CI of **[+0.0048, +0.0178]**, but McNemar on R@1
gives **p = 0.071**. So MRR and R@10 improve reliably and R@1 only weakly — the
claim is "better ranking overall", not "better top-1". An earlier note in this
repo's history quoted +1.4 R@1 from this change; it did not replicate at +0.7 and
should not be repeated.

**Where the semantic numbers came from.** They were 46.0% / 0.567 until a
one-line change on the query side. SigLIP 2 tokenizes with a *case-sensitive*
SentencePiece vocabulary, and `AutoProcessor` does not apply the model's
canonical lowercase-and-depunctuate preprocessing for you — so a leading capital
becomes its own rare token rather than part of the sentence, and the query vector
moves. Normalizing the query text before encoding is worth **+7.2 points of R@1**,
of which lowercasing the single leading character accounts for +4.6:

| query text | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|
| raw | 46.0% | 70.9% | 80.0% | 0.5672 |
| first character lowercased | 50.6% | 73.9% | 82.3% | 0.6072 |
| lowercased + depunctuated | **53.2%** | **76.0%** | **84.1%** | **0.6280** |

It is query-side only and a no-op on text that is already lowercase without
punctuation, so nothing a user types today can regress — and short 4-word queries
improve too (R@10 15.9% → 21.3%), which is the check that stops a change like
this from being an artefact of whole-caption benchmark queries. The stored
caption vectors were left alone; matching a normalized query against them
measured *better* than matching a raw one, so there was nothing to re-embed.

**Who actually gains, stated precisely.** 925 of the 1,000 benchmark queries begin
with a capital letter, and the gain is concentrated there: 46.1% → 53.6% R@1 on
those, against 45.3% → 48.0% on the 75 already-lowercase ones (n=75, noisy —
punctuation stripping only). So this is not a 7-point improvement for the average
user. It is a 7-point improvement on the benchmark's query distribution, and a
real bug fix for the subset of users who capitalise a sentence — who were
silently getting materially worse results than users who did not.

**What was refuted along the way**, because the negative results are the more
useful ones:

* **Widening the lexical conjunction.** The AND *is* why 90.6% of whole-caption
  queries return nothing, and switching to OR takes keyword R@10 from 3.6% to
  53.0% in isolation — yet it makes *fused* ranking worse (hybrid MRR 0.6313 →
  0.5850), because a broad lexical list displaces stronger semantic candidates.
  Measured, then not shipped.
* **Prompt templating** (`"a photo of {}"` and variants): within noise on
  normalized text. Templates help bare class names in zero-shot classification;
  these queries are already sentences.
* **Skipping a retrieval path that returned nothing:** under RRF an empty list
  contributes no terms, so it is bit-identical to not skipping. Asserted
  element-wise over all 1,000 queries.
* **Stale caption vectors:** cosine against a fresh re-encode of identical text is
  1.0000. They were never stale.
* **Tuning `RRF_K` and the lexical weight** across `k ∈ {5…120}` and
  `w ∈ {0.1…1.5}`: nothing beat the existing `k=60, w=1.0` by more than 0.002 MRR
  once query length was accounted for. The defaults were already right.

The decisive test in every case was **query length**. Several variants beat the
shipped configuration on whole captions and all of them lost at three words —
including a dense-caption fusion path whose apparent gain came from the query
being a sibling of the target's own other captions, which no real user query is.

The benchmark now also encodes its queries live through the same code path
`/api/search` uses, rather than reusing the precomputed caption vector as the
query. That is a stricter test — it measures what a user's query actually goes
through — and it is why `PROTOCOL_VERSION` moved to 4, so results cached under
the old method can never be read back as if they were comparable. It moved to
**5** because holding out the hubness bank changed the evaluated sample, which is
exactly the kind of change a cached row must not survive, and stands at **8**
after the trained-reranker experiment was removed from the product: its rows
left the protocol, so earlier caches cannot be read as this protocol's results.

Two things this table is honest about, both of which cost real work to learn:

**The self-exclusion.** Without `AND c.id != ?`, keyword R@1 was **99.1%** —
pure self-retrieval, finding the exact caption it had been handed. That
contaminated hybrid to 79.0%. The exclusion is the difference between a
benchmark and a mirror.

**The `mean candidates` column.** Keyword's 4.2% is not a ranking failure. It is
**2.1 candidates per query and 85.3% empty**: `fts_escape` ANDs the query's terms,
so a ~12-word caption is a twelve-way conjunction that almost nothing satisfies.
Reporting recall without the denominator would invite exactly the wrong
conclusion.

It ANDs *content* terms, which is the second ranking change. A function word was
previously a hard requirement — "man on a bench" could not match "man on the
bench" because of "the" — so `db.match_terms` now drops stopwords from the
conjunction before `fts_escape` quotes them. Worth **R@10 3.6% → 6.4%** and
**empty 90.6% → 84.0%** on the fixed offline sample. The terms are still *indexed*;
only the conjunction is relaxed, so a query that is nothing but function words
still matches nothing rather than everything.

---

## 5. Derived signals

`python -m app.analyze` computes, once:

- **Caption agreement** — SigLIP cosine between an image and each of its
  captions. Low agreement flags a probable annotation error; the sibling mean
  disambiguates *"this caption is wrong"* from *"this image is unusual"*.
- **Caption consistency** — mean pairwise similarity among a sample's five
  captions. Low means annotators disagreed about what the picture shows.
- **Four difficulty axes** — percentile ranks, 0-10, over blur/luminance
  (legibility), caption-vocabulary IDF and embedding isolation (rarity),
  agreement and consistency (difficulty), and how much the captions name
  (clutter).
- **Zero-shot attributes** — SigLIP against a label bank, with a **top1−top2
  margin gate** (`ATTR_MIN_MARGIN = 0.10`) below which the group is left
  unlabelled and counted as abstained.

The margin gate deserves its paragraph. A probability floor cannot work here:
`setting` has two labels, so its argmax is ≥ 0.5 by construction and no floor
could ever reject one, while `environment` has seven and a 0.30 winner may be a
coin toss. A *margin* means the same thing in both. Measured across all four
groups, abstention rises monotonically with the number of labels — which is
precisely the behaviour a probability floor could not have produced:

| group | labels | scored | abstained |
|---|---|---|---|
| `setting` | 2 | 7,782 | 218 (2.7%) |
| `time_of_day` | 3 | 7,763 | 237 (3.0%) |
| `main_subject` | 6 | 7,512 | 488 (6.1%) |
| `environment` | 7 | 7,241 | 759 (9.5%) |

An abstention is recorded as an absent row, and the coverage chart reports it as
such, so a slice's percentages always sum against the true corpus rather than
against whatever happened to be labelled.

Why percentile ranks rather than raw values: a Laplacian variance, a cosine
distance and an IDF live on incomparable scales, so `blur >= 40` means something
different on every dataset and nothing at all to a person. Ranking makes
`rarity >= 7` and `difficulty >= 7` both mean "top ~30% of this corpus" and lets
four sliders be used together. The cost is that scores are dataset-relative — a 7
here is not a 7 on COCO.

---

## 6. API layer

FastAPI, Pydantic 2 response models on every route, 35 endpoints (the count comes from
`docs/CAPABILITIES.md`, which is generated from the live schema). Full schema at
`/docs`; the grouped inventory is in `docs/CAPABILITIES.md`.

Three conventions:

**Shared dependencies, not repeated parsing.** `axis_bounds` and `id_list` are
`Depends(...)` functions, so the eight axis parameters are declared once,
validated by FastAPI, and documented in OpenAPI — rather than re-parsed per
endpoint.

**Search has a POST twin.** A 60,000-entry id list is roughly 400 kB of query
string, which no URL can carry. `POST /api/search` takes the same parameters in a
body and calls the *same* `run_search`, so there is one ranking implementation
rather than two that can drift.

**Export mirrors search exactly.** `/api/export` takes search's parameters and
delegates to `run_search`, so an export is precisely the result set the user was
looking at, in the same order. The manifest records the query, the axis bounds
and the embedding model, because a slice you cannot regenerate is not curation.

---

## 7. Frontend

React 18 + TypeScript (strict) + Vite. Four dependencies total: `react`,
`react-dom`, `react-router-dom`, `recharts`.

**URL as the single source of truth.** Every filter, the search mode, the sort
key, the axis ranges and the pagination depth live in the query string. That
gives shareable links, a working back button, and "load more" depth that survives
navigating to a sample and back — for free, with no client state library.

**Route-level code splitting.** Every non-default route is `React.lazy`, so the
gallery ships without recharts, the canvas map, or the chat view.

**One source of design tokens.** `src/lib/viz.ts` exports the palette, axis and
tooltip styling used by every chart. Colour ramps are viridis — perceptually
uniform and colourblind-safe — and encodings are redundant (height *and* colour),
so charts survive greyscale.

**Canvas for the map, SVG for charts.** 8,000 points as DOM nodes would be
unusable; the map is canvas 2D and renders a frame in ~0.6 ms, about 28× under
the 60 fps budget. That measurement is also why the map is *not* GPU-rendered:
deck.gl would have added 100-500 kB to fix a bottleneck that does not exist.

**Three columns: inputs, artifact, set.** `App.tsx` is a CSS grid —
`var(--rail-l) minmax(0,1fr) auto`. Navigation and every filter live in a 240px
left rail grouped by job (Find / Audit / Trust / Ask); the centre pane holds the
artifact; the current selection has a permanent home in a 300px right rail.

This replaced a single vertical column that stacked all three, and the defect it
cured is worth stating precisely: **the more precisely you specified a set, the
less of it you could see**, because three auto-opening `<details>` pushed the
grid down. Measured chrome above the first image:

| state | before | after |
|---|---|---|
| cold gallery | 374px | 147px |
| filtered + search | 481px | 159px |
| heavily filtered | ~880px | **106px** |

The last row is the point — it is now roughly *constant* under filtering rather
than growing, and the heaviest case is the cleanest because the query
suggestions correctly disappear once a selection exists.

**`useSelection` is a read, not a store.** The right rail renders on every route,
so the chip-building and export-link logic that was private to `GalleryPage` had
to become reachable from anywhere. It is a hook over `useSearchParams` with
deliberately no context, no reducer and no cache: the moment a selection lives in
two places the interface can show one thing while the URL says another, and every
hand-off in the product stops being trustworthy. The rail is absent — not empty —
when nothing is selected, because browsing the whole corpus is exactly when you
want the grid columns back.

One distinction that cost a regression to learn: `active` (any membership
constraint) and `exportable` (that, *or* a query) are different. Gating the rail
on `active` alone meant a plain search returned 60 results with no way to export
them. The QA suite caught it.

**Every encoding has a key.** The four-bar difficulty sparkline appears on every
card in three different views and had no legend anywhere — the map had one, four
chart renderers had one, the gallery had none, and its whole explanation was an
SVG `<title>`. `AxisLegend` now sits in the result bar and teaches by showing: a
worked example rendered by *the same component the cards use*, so the key cannot
drift from the thing it explains. It is absolutely positioned, so opening it
costs the grid 0px (verified 147px → 147px), and it follows the cards to the
sample page and the assistant's image strip.

**The set-handoff contract.** Every view produces a set, and sets flow between
views: the map's lasso hands its ids to the gallery, the quality brush hands its
threshold, a chart bar in the assistant hands its slice. Nothing is a dead end.

**Render blocks.** Agent answers arrive as typed blocks with a renderer registry
that is a mapped type over the block union — so adding a block kind to the
backend fails `tsc` in the frontend until a renderer exists. See
`docs/AGENTS.md`.

---

## 8. Testing

| Tier | What | Count |
|---|---|---|
| `backend/tests/` | API contracts, degraded modes, id-list limits, provider resolution/fallback, agent graph (parallelism, lane isolation, timeouts), embedder concurrency, block validation | **383 passed** locally (2026-07-29); CI's light install skips the torch/langgraph modules |
| `scripts/ui_smoke.py` | Real Chrome over every workflow, screenshots, console errors, 4xx/5xx | **17 workflows registered**; the last full sweep passed 106/106 checks (run 20260728-233210-9847) |
| `python scripts/capabilities.py --check` | Fails when the docs drift from the running system | — |

The backend figure is what GitHub Actions reports on the light install described
in `docs/TESTING.md`: the modules needing `torch` or `langgraph` skip there, so a
local run with the optional stacks installed executes more than this.

The frontend has no unit tier. That is a deliberate trade: for a UI this size,
the failures that matter are "the view rendered empty", "the control stopped
filtering", "a console error appeared" — and those are caught end-to-end, in a
real browser, by assertions written against behaviour rather than implementation.

Two things the test suite proves that would otherwise be assertions:

- **Parallel fan-out is real.** Lane execution windows are recorded and asserted
  to overlap. A duration threshold alone would pass on a fast machine that still
  ran them serially.
- **Graceful degradation works.** A QA flow intercepts `/api/views` and
  `/api/tags`, returns 500s, and asserts the UI announces them — while a 404
  (an optional router simply absent) must stay quiet. Errors it causes on purpose
  are filed separately so a test proving the app survives failure cannot itself
  report the app as broken.

---

## 9. Performance, and where it stops working

Measured on an M-series Mac:

| Operation | Cost |
|---|---|
| Gallery page (filtered, sorted) | 0.58 ms |
| Keyword ranking to depth 300 | 1.33 ms |
| Threshold → sample set | 2.22 ms |
| Attribute coverage | 1.12 ms |
| Semantic scan, 8,000 × 768 (24.6 MB) | 0.18 ms |
| SigLIP text encode (MPS) | 7-8 ms — ~40× the scan, and the real cost of a semantic query |
| Map frame, 8,000 points | ~0.6 ms |
| Full UI sweep, 17 workflows (run 20260728-233210-9847) | 85 s |

**Known ceilings, in the order they will be hit:**

1. **~400k vectors** — the exact scan reaches the cost of the SigLIP encode it
   waits behind, so it stops being free. Measured: 0.16 ms at 8k, 1.7 ms at 100k,
   4.1 ms at 250k, 18 ms at 1M, against a 6-8 ms encode. Two earlier drafts of
   these docs said 100k and 10⁶ — a 10x disagreement, and both were wrong.
2. **`captions.agreement` has no index** — the threshold query is a full scan.
   Fine at 40k, not at 400k.
3. **Axis percentiles are recomputed corpus-wide** — `app.analyze` is O(n log n)
   over everything, so incremental ingest means a full re-rank.
4. **One SQLite writer** — fine for a single-user local tool, wrong for
   concurrent curation by a team.

**One hard-won concurrency rule.** SigLIP inference is serialized behind a lock,
because two threads running the same model on Apple's Metal backend either
segfault inside `copy_cast_kernel_mps` or deadlock — both reproduced. FastAPI
serves sync endpoints from a thread pool, so *two simultaneous semantic searches*
were always enough to trigger it. A second model copy would cost ~1.5 GB to avoid
a wait of milliseconds; the lock is the right trade.

---

## 10. What is deliberately absent

- **No ANN index.** Exact search is faster than the encode at this scale.
- **No fifth difficulty axis.** Systems like this usually carry a *dynamic
  complexity* axis. There is no honest analogue in still photographs, and
  inventing one to round the count to five would make the panel look more
  complete and the data less true.
- **No multi-label attributes.** The classifier produces a softmax over a group;
  pretending otherwise would be a UI fiction.
- **No cloud anything.** No hosted model, managed database, external vector
  store or paid API. The whole system runs on one machine, offline after ingest.
