import html
import re
import sys

import psycopg

from common import DSN, EMBED_MODEL, embed_query, get_model, to_vector

MIN_CHARS = 200
CHUNK_SIZE = 1500
CHUNK_STEP = 750

RAW_SQL = """
SELECT story_id, object_id::bigint, story_title, comment_text, created_at_i
FROM hn_raw.comments
WHERE comment_text IS NOT NULL
  AND story_title IS NOT NULL
"""

INSERT_SQL = """
INSERT INTO chunks
    (story_id, comment_id, chunk_index, story_title, chunk_text, created_at_i)
VALUES (%s, %s, %s, %s, %s, %s)
ON CONFLICT (comment_id, chunk_index) DO NOTHING
"""

SEARCH_SQL = """
SELECT story_id, story_title, left(chunk_text, 220), embedding <=> %s::vector
FROM chunks
WHERE embedding IS NOT NULL
ORDER BY embedding <=> %s::vector
LIMIT %s
"""

TAG_RE = re.compile(r"<[^>]+>")


# --------------------------------------------------------------------------
# cleaning
# --------------------------------------------------------------------------

def clean(text: str) -> str:
  
    t = text.replace("<p>", "\n\n")
    t = TAG_RE.sub(" ", t)
    t = html.unescape(t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def split(text: str) -> list[str]:
    if len(text) <= CHUNK_SIZE:
        return [text]
    out = []
    for i in range(0, len(text), CHUNK_STEP):
        piece = text[i : i + CHUNK_SIZE]
        if len(piece) < MIN_CHARS and out:
            break
        out.append(piece)
        if i + CHUNK_SIZE >= len(text):
            break
    return out


def clean_rows():
    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute(RAW_SQL)
        for story_id, comment_id, title, body, created in cur:
            text = clean(body)
            if text.startswith(("[deleted]", "[dead]", "[flagged]")):
                continue
            if len(text) < MIN_CHARS:
                continue
            for idx, piece in enumerate(split(text)):
                yield story_id, comment_id, idx, title, piece, created




# --------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------

def stage_count():
    """Pass 1 — clean and count. Writes nothing, safe to re-run freely."""
    kept = dropped_short = dropped_dead = 0

    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute(RAW_SQL)
        for _story_id, _comment_id, _title, body, _created in cur:
            text = clean(body)
            # dead comments are checked first: they are also short, and
            # counting them as "short" would make the diagnostics lie
            if text.startswith(("[deleted]", "[dead]", "[flagged]")):
                dropped_dead += 1
            elif len(text) < MIN_CHARS:
                dropped_short += 1
            else:
                kept += 1

    print(f"kept          {kept}")
    print(f"dropped short {dropped_short}")
    print(f"dropped dead  {dropped_dead}")
    print(f"total         {kept + dropped_short + dropped_dead}")


def stage_insert(batch_size=1000):
    """Pass 2 — write chunks. ON CONFLICT makes re-runs a no-op."""
    batch, total = [], 0

    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        for row in clean_rows():
            batch.append(row)
            if len(batch) >= batch_size:
                cur.executemany(INSERT_SQL, batch)
                conn.commit()
                total += len(batch)
                batch.clear()
                print(f"{total} inserted")
        if batch:                      # the tail — without this it is lost
            cur.executemany(INSERT_SQL, batch)
            conn.commit()
            total += len(batch)

    print(f"done: {total} chunks")


def stage_embed(batch_size=256):
    """Pass 3 — fill in the vectors. Resumable: only touches NULL rows."""
    model = get_model()

    with psycopg.connect(DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, embed_text FROM chunks "
                "WHERE embedding IS NULL ORDER BY id"
            )
            rows = cur.fetchall()

        print(f"embedding {len(rows)} chunks with {EMBED_MODEL}")

        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            # .embed() is the *document* side of the model — see stage_search
            vectors = list(model.embed([text for _, text in batch]))

            payload = [
                (to_vector(vec), chunk_id)
                for (chunk_id, _), vec in zip(batch, vectors)
            ]
            with conn.cursor() as cur:
                cur.executemany(
                    "UPDATE chunks SET embedding = %s::vector WHERE id = %s",
                    payload,
                )
            conn.commit()
            print(f"{min(start + batch_size, len(rows))}/{len(rows)}")

    print("done")


def stage_search(question=None, n=5):
    """Pass 4 — eyeball the retrieval before trusting it in Step 5."""


    question = question or (
        "should I do a masters degree or take the job offer"
    )
    vec = embed_query(question)

    print(f"\nQ: {question}\n")
    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute(SEARCH_SQL, (vec, vec, n))
        for story_id, title, snippet, dist in cur:
            print(f"[{dist:.4f}] {title}  (story {story_id})")
            print(f"    {snippet.strip()}...\n")


STAGES = {
    "count": stage_count,
    "insert": stage_insert,
    "embed": stage_embed,
    "search": stage_search,
}


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "count"
    if name not in STAGES:
        sys.exit(f"unknown stage {name!r} — pick one of {', '.join(STAGES)}")
    if name == "search" and len(sys.argv) > 2:
        stage_search(" ".join(sys.argv[2:]))
    else:
        STAGES[name]()
