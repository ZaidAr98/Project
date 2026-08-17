# What does Hacker News think?

A RAG application over **Ask HN** threads about careers, developer tooling, and
AI's effect on how people work.

Ask it *"Should I do a masters degree or take the job offer?"* and it does not
give you an answer. It gives you **a summary of the disagreement** — the
positions real commenters took, how many took each, and links back to the
original comments.

---

## The problem

Most RAG demos answer questions with one correct answer. This one deliberately
does not.

| | Typical RAG | This project |
| --- | --- | --- |
| Question | *"How do I fix this error?"* | *"Should I take this job?"* |
| Answer | Facts and code | Opinions and experience |
| Right answer? | Yes, one | No — several, and they conflict |

**Why retrieval matters more here, not less.** An LLM already knows the factual
material in a documentation corpus, so retrieving it adds little. What it does
*not* know is what developers in mid-2026 actually said about AI coding tools, or
whether people regretted joining Amazon. That gap is what this knowledge base
fills.

**The constraint that follows:** the app must never present one person's opinion
as settled fact. That drove the prompt design and the whole evaluation strategy.

Example questions:

- *Should I do a masters degree or take the job offer?*
- *Do experienced developers think AI coding tools actually help?*
- *Is it still worth learning to write code by hand?*

---

## Architecture

```
HN Algolia API ──dlt──▶ Postgres + pgvector ──▶ Retrieval ──▶ OpenAI ──▶ Streamlit
                             │                text / vector /        │
                             │                hybrid / rerank        │
                             └────────── conversations + feedback ◀──┘
                                             │
                                          Grafana
```

**One datastore for everything** — Postgres holds the comments, the embeddings,
the full-text index and the monitoring tables. Grafana reads the same database,
and hybrid search becomes one SQL query instead of a cross-system join.

| Layer | Choice |
| --- | --- |
| Ingestion | dlt, `rest_api` source |
| Store | `pgvector/pgvector:pg17` |
| Embeddings | fastembed, `BAAI/bge-small-en-v1.5` (384d) |
| Reranker | fastembed cross-encoder, `ms-marco-MiniLM-L-6-v2` |
| LLM | OpenAI `gpt-5.4-mini` |
| Interface / dashboard | Streamlit / Grafana 11.6.0 |

---

## The data

Source is the **Algolia HN Search API** — no auth, no key, and it returns up to
1,000 records per request where the official Firebase API returns one.

The corpus was scoped deliberately rather than by grabbing everything:

| Stage | Count |
| --- | --- |
| Threads scanned (Jun–Jul 2026) | 2,217 |
| With 10+ comments | 332 |
| `Ask HN:` titles only | 235 |
| Megathreads removed | **226 threads** |
| Raw comments | 10,793 |
| Kept after cleaning (200-char minimum) | 5,845 |
| **Chunks embedded** | **6,281** |

*Who Is Hiring* and *What Are You Working On* are excluded on purpose — they are
the biggest threads on the site and they are job listings and self-promotion,
text that would flood the knowledge base while answering nothing.

> The 226 figure drifts with the live API as threads gain comments. Expected.

**One decision mattered more than any tuning:** every comment is embedded with
its thread title prepended. A comment alone is often meaningless — *"I'd go with
the second option"* says nothing without the question above it. This happens in a
Postgres generated column, so the embedding text can never drift from the source.

---

## How a question is answered

`pipelines/rag.py` — **rewrite → search → cap → prompt → answer → map citations**

| Setting | Value |
| --- | --- |
| Strategy | `hybrid_rerank` (measured winner) |
| Retrieved | 20 |
| Reaching the prompt | 8 |
| Max per thread | 2 |

Two LLM calls: one rewrites the question into a search query, one writes the
answer. The answer prompt gets the **original** question — the stripped version
is only good enough for a search engine.

**The per-thread cap exists because of a measured problem.** Prepending the
thread title works so well that one relevant thread can occupy every result slot,
since all ~96 of its chunks carry the same matching title. Summarising a
disagreement from one thread is not summarising a disagreement.

---

## Retrieval evaluation

300 questions generated from 100 sampled comments, each labelled with the comment
it came from. Scored **strict** (only the originating comment counts) and
**loose** (any chunk from the right thread counts).

```bash
uv run python pipelines/evaluate.py score
```

| Strategy | strict HR | strict MRR | loose HR | loose MRR |
| --- | --- | --- | --- | --- |
| `text` | 0.440 | 0.349 | 0.630 | 0.497 |
| `vector` | 0.467 | 0.352 | 0.693 | 0.611 |
| `hybrid` | 0.570 | 0.422 | 0.743 | 0.626 |
| **`hybrid_rerank`** | **0.627** | **0.504** | **0.820** | **0.702** |

Perfectly monotonic — every strategy beats the one it is built from. **The app
ships `hybrid_rerank`.**

- **Text search is weak alone but not useless.** 0.440 by itself, yet fusing it
  with vector lifted strict HR to 0.570 — a bigger jump than vector gained over
  text. The two genuinely find different things, which justifies hybrid search.
- **Reranking improves MRR more than hit rate** (+19% relative), which is exactly
  what a reranker should do — it reorders rather than finds.
- **The winner still misses 37% of the time on strict scoring.** Reported
  deliberately; an imperfect winner is more trustworthy than a claim of 0.95.

---

## LLM evaluation

Never *"is this answer true?"* — with opinions that is undecidable. Instead:
**"does this answer faithfully represent the comments it retrieved?"**

Retrieval is **frozen first**, so all three prompts answer from byte-identical
context — otherwise you measure search noise and call it prompt quality. 17
questions × 3 prompts = 51 answers, each scored 1–5 by a judge that sees the
question, the comments *and* the answer.

```bash
uv run python pipelines/evaluate_llm.py score
```

| Variant | Groundedness | Coverage | Relevance | Mean | Invented citations |
| --- | --- | --- | --- | --- | --- |
| `terse` | 3.53 | 3.29 | **3.94** | 3.59 | 0 |
| **`cot`** | **4.06** | **4.06** | **3.94** | **4.02** | 0 |
| `citation` | 3.59 | 3.41 | 3.82 | 3.61 | 0 |

**The app ships `cot`.**

**Coverage is weighted highest** because it is this app's actual job. An answer
reporting one of three positions has failed however well it reads — so fluency is
not scored at all.

- **`cot` wins with a traceable cause:** its prompt forces the model to list
  every distinct position under a `Positions:` heading *before* summarising.
- **Relevance is flat across all three prompts**, because relevance is set by
  retrieval, not prompting. The LLM evaluation exposed a retrieval ceiling no
  prompt could fix.
- **Zero invented citations across 51 answers**, verified by a deterministic
  non-LLM check that every `[n]` points at a comment actually supplied.

---

## Monitoring

Every exchange is logged to `conversations` — question, rewritten query, answer,
tokens, cost, latency, cited threads. Thumbs up/down land in `feedback`.

Grafana is **provisioned as code**, so a fresh clone comes up populated instead
of needing a setup wizard. Seven panels: questions asked, total spend, latency
p50/p95, questions over time, feedback split, cumulative cost, and most-cited HN
threads.

<!-- ![Streamlit](docs/streamlit.png) -->
<!-- ![Grafana](docs/grafana.png) -->

---

## Running it

Needs Docker, [uv](https://docs.astral.sh/uv/), and an OpenAI API key. A single
question costs about $0.004.

```bash
# 1. configure
cp .env.example .env          # fill in OPENAI_API_KEY, POSTGRES_PASSWORD,
                              # and DESTINATION__POSTGRES__CREDENTIALS

# 2. database + dashboard        Grafana → http://localhost:3000  (admin/admin)
docker compose up -d

# 3. ingest from the HN API
uv run python pipelines/hn_pipeline.py

# 4. clean, embed, index
uv run python pipelines/build_chunks.py count    # count only, writes nothing
uv run python pipelines/build_chunks.py insert   # rows, embedding NULL
uv run python pipelines/build_chunks.py embed    # fill in the vectors

# 5. the app                     → http://localhost:8501
uv run streamlit run pipelines/app.py
```

> **The first query downloads ~220 MB** of embedding and reranking models. The
> app warms both at startup so it does not distort the latency metrics, but the
> first launch pauses while it downloads. This is not a hang.

Steps 3 and 4 are safely re-runnable — ingestion merges on ID, and the embed
stage only touches rows where `embedding IS NULL`.

**Both evaluations re-run with no API key** (they read committed CSVs):

```bash
uv run python pipelines/evaluate.py score
uv run python pipelines/evaluate_llm.py score
```

**Ask from the command line:**

```bash
uv run python pipelines/rag.py "is Amazon a good place to start a career"
```

---

## Environment variables

| Variable | Purpose |
| --- | --- |
| `OPENAI_API_KEY`, `OPENAI_MODEL` | OpenAI credentials and model |
| `OPENAI_INPUT_COST_PER_1M`, `OPENAI_OUTPUT_COST_PER_1M` | prices for cost logging |
| `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, `POSTGRES_HOST`, `POSTGRES_PORT` | database |
| `DESTINATION__POSTGRES__CREDENTIALS` | full Postgres URL — **dlt reads this, not the vars above** |
| `EMBEDDING_MODEL`, `RERANK_MODEL` | fastembed model names |
| `GRAFANA_USER`, `GRAFANA_PASSWORD` | Grafana login |

The cost variables only affect what the app *records*. OpenAI's API returns token
counts but never a dollar figure, so the multiplication happens locally — they
have no effect on billing.

---

## Project layout

```
db/init.sql                chunks, conversations, feedback + indexes
docker-compose.yaml        postgres + grafana
grafana/                   provisioned datasource and dashboard
pipelines/hn_pipeline.py   dlt ingestion
pipelines/build_chunks.py  clean, chunk, embed
pipelines/search.py        the four retrieval strategies
pipelines/rag.py           the RAG flow
pipelines/evaluate*.py     the two evaluations
pipelines/app.py           Streamlit interface
data/                      committed ground truth and scores
```

---

## Known limitations

Several of these were found *by* the evaluations, which is the point of running
them.

- **There is a corpus gap.** Questions about coding interviews, microservices and
  "is Postgres enough" scored 1–2 on relevance for every prompt. No prompt or
  ranking change fixes that — only ingesting more threads would.
- **The per-thread cap starves single-thread topics.** Of 20 test questions, 17
  filled all 8 context slots from 5–8 threads. The 3 that did not were questions
  mapping onto one dominant thread. A deliberate trade: diversity over volume.
- **The 300 evaluation questions are not 300 independent tests** — 100 comments,
  3 near-paraphrases each.
- **Judge and generator are the same model**, a known bias in LLM-as-judge.
- **Ingestion uses a hard-coded date window**, not a loop, and has no incremental
  cursor — a re-run re-downloads rather than fetching only what is new. `merge`
  keeps the data correct, but it prevents duplicate *rows*, not duplicate *work*.
- **The Streamlit app runs outside Docker.** Postgres and Grafana are in compose.

---

## Where each grading criterion is met

| Criterion | Where |
| --- | --- |
| Problem description | top of this README |
| Retrieval flow | `pipelines/rag.py` — Postgres knowledge base + OpenAI |
| Retrieval evaluation | four strategies compared, `hybrid_rerank` shipped |
| LLM evaluation | three prompts compared, `cot` shipped |
| Interface | `pipelines/app.py` — Streamlit with feedback |
| Ingestion pipeline | `pipelines/hn_pipeline.py` — dlt |
| Monitoring | `feedback` table + 7-panel Grafana dashboard |
| Containerization | `docker-compose.yaml` — see limitations for what is not in it |
| Reproducibility | instructions above, plus `uv.lock` pinning every dependency |

**Best practices:** hybrid search (RRF over text + vector), document re-ranking
(cross-encoder over a 30-candidate pool), and user query rewriting (LLM rewrite
before search) — all in `pipelines/search.py` and `pipelines/rag.py`.
