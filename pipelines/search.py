import sys

import psycopg

from common import DSN, embed_query, get_reranker


COLUMNS = ("id", "story_id", "comment_id", "story_title", "chunk_text", "score")
OR_TSQUERY = "replace(plainto_tsquery('english', %s)::text, '&', '|')::tsquery"

TEXT_SQL = f"""
SELECT id, story_id, comment_id, story_title, chunk_text,
       ts_rank(tsv, {OR_TSQUERY}) AS score
FROM chunks
WHERE tsv @@ {OR_TSQUERY}
ORDER BY score DESC
LIMIT %s
"""



VECTOR_SQL = """
SELECT id, story_id, comment_id, story_title, chunk_text,
       embedding <=> %s::vector AS score
FROM chunks
WHERE embedding IS NOT NULL
ORDER BY embedding <=> %s::vector
LIMIT %s
"""


def rows_to_dicts(cur) -> list[dict]:
    """Every strategy returns the same keys — see the module docstring."""
    return [dict(zip(COLUMNS, row)) for row in cur]




def text_search(query: str, n: int = 5) -> list[dict]:

    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute(TEXT_SQL, (query, query, n))
        return rows_to_dicts(cur)



def vector_search(query: str, n: int = 5) -> list[dict]:
 
    vec = embed_query(query)
    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute(VECTOR_SQL, (vec, vec, n))
        return rows_to_dicts(cur)



RRF_K = 60    # standard default; larger flattens each list's curve
POOL  = 50    # candidates fetched per strategy before fusing


def rrf(*ranked_lists, k: int = RRF_K) -> dict[int, float]:
  
    scores: dict[int, float] = {}
    for results in ranked_lists:
        for rank, row in enumerate(results, 1):
            scores[row["id"]] = scores.get(row["id"], 0.0) + 1.0 / (k + rank)
    return scores


def hybrid_search(query: str, n: int = 5, pool: int = POOL, k: int = RRF_K) -> list[dict]:
 
    text = text_search(query, pool)
    vector = vector_search(query, pool)

    scores = rrf(text, vector, k=k)
    by_id = {row["id"]: row for row in text + vector}

    top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:n]
    return [{**by_id[chunk_id], "score": score} for chunk_id, score in top]



RERANK_POOL = 30


def hybrid_rerank(query: str, n: int = 5, pool: int = RERANK_POOL) -> list[dict]:
 
    candidates = hybrid_search(query, n=pool)
    if not candidates:
        return []

    scores = get_reranker().rerank(query, [c["chunk_text"] for c in candidates])

    ranked = sorted(zip(scores, candidates), key=lambda pair: pair[0], reverse=True)
    return [{**row, "score": score} for score, row in ranked[:n]]



STRATEGIES = {
    "text": text_search,
    "vector": vector_search,
    "hybrid": hybrid_search,
    "rerank": hybrid_rerank,
}

if __name__ == "__main__":
    name, *rest = sys.argv[1:] or ["vector"]
    q = " ".join(rest) or "is Amazon a good place to start a career"
    for i, r in enumerate(STRATEGIES[name](q), 1):
        print(f"{i}. [{r['score']:.4f}] {r['story_title']}  (story {r['story_id']})")
        print(f"   {r['chunk_text'][:200].strip()}...\n")