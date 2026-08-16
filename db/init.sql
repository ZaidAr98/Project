CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE chunks (
    id           BIGSERIAL PRIMARY KEY,
    story_id     BIGINT NOT NULL,
    comment_id   BIGINT NOT NULL,
    chunk_index  INT    NOT NULL DEFAULT 0,
    story_title  TEXT   NOT NULL,
    chunk_text   TEXT   NOT NULL,
    embed_text   TEXT GENERATED ALWAYS AS (story_title || E'\n\n' || chunk_text) STORED,
    embedding    vector(384),
    tsv          tsvector GENERATED ALWAYS AS (
                     to_tsvector('english', story_title || ' ' || chunk_text)
                 ) STORED,
    created_at_i BIGINT,
    UNIQUE (comment_id, chunk_index)
);

CREATE INDEX chunks_tsv_idx       ON chunks USING GIN (tsv);
CREATE INDEX chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX chunks_story_idx     ON chunks (story_id);

CREATE TABLE conversations (
    id              BIGSERIAL PRIMARY KEY,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    question        TEXT NOT NULL,
    rewritten_query TEXT,
    answer          TEXT,
    strategy        TEXT,
    prompt_variant  TEXT,
    cited_story_ids BIGINT[],
    input_tokens    INT,
    output_tokens   INT,
    cost_usd        NUMERIC(10, 6),
    latency_ms      INT
);

CREATE TABLE feedback (
    id              BIGSERIAL PRIMARY KEY,
    conversation_id BIGINT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    vote            SMALLINT NOT NULL CHECK (vote IN (-1, 1)),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);