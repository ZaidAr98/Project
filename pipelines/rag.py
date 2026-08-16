
import os
import re
import time

from dataclasses import dataclass, replace
from search import hybrid_rerank

STRATEGY = "hybrid_rerank"   # Step 6's measured winner — do not drift from this
N_RETRIEVE = 20              # reranked results fetched before capping
N_CONTEXT = 8                # chunks that actually reach the prompt
PER_THREAD = 2               # max chunks from any one story

IN_COST = float(os.getenv("OPENAI_INPUT_COST_PER_1M", "0")) / 1_000_000
OUT_COST = float(os.getenv("OPENAI_OUTPUT_COST_PER_1M", "0")) / 1_000_000

REWRITE_PROMPT = """Rewrite this into a short search query for a keyword and
semantic search over forum comments. Keep the meaningful nouns and verbs, drop
filler words. Return only the query, nothing else.

Question: {question}"""


PROMPTS = {
    "terse": """You are reporting what people on a discussion forum said about a
question. Use only the numbered comments below — never your own knowledge.

Answer in at most 5 sentences. No preamble, no hedging, no restating the
question. If the comments disagree, say so and give both sides. If the comments
do not address the question, say only that.

Question: {question}

Comments:
{context}""",

    "cot": """You are reporting what people on a discussion forum said about a
question. Use only the numbered comments below — never your own knowledge.

Work in two parts.

First, under the heading "Positions:", list every distinct position you can find
in the comments. Give each position in one line, followed by the comment numbers
that hold it.

Then, under the heading "Summary:", write the answer for the reader. Cover every
position you listed, in proportion to how many comments held it. Never present
one person's view as settled fact.

Question: {question}

Comments:
{context}""",

    "citation": """You are reporting what people on a discussion forum said
about a question. Use only the numbered comments below — never your own
knowledge.

Rules:
- These are opinions, not facts. Never present one person's view as settled.
- If the comments disagree, show the disagreement: what each side argued, and
  roughly how many took each position.
- Cite every claim with its source number in square brackets, like [3].
- If the comments do not address the question, say so plainly.

Question: {question}

Comments:
{context}""",
}


@dataclass
class Answer:
    """Maps 1:1 onto the conversations table — Steps 9 and 10 write it straight in."""
    question: str
    rewritten_query: str
    answer: str
    strategy: str
    prompt_variant: str
    chunks: list[dict]
    cited_story_ids: list[int]
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: int


def cap_per_thread(results: list[dict],
                   per_thread: int = PER_THREAD,
                   limit: int = N_CONTEXT) -> list[dict]:
    """Keep at most `per_thread` chunks from any one story, preserving rank order.

    Measured 2026-08-15: every strategy returns 5 of 5 from a single thread,
    because one prepended title lifts all of its chunks together. Summarising a
    disagreement from one thread's chunks isn't summarising a disagreement.
    """
    seen: dict[int, int] = {}
    kept: list[dict] = []
    for row in results:
        story_id = row["story_id"]
        if seen.get(story_id, 0) >= per_thread:
            continue
        seen[story_id] = seen.get(story_id, 0) + 1
        kept.append(row)
        if len(kept) == limit:
            break
    return kept


def build_context(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"[{i}] (thread: {c['story_title']})\n{c['chunk_text']}"
        for i, c in enumerate(chunks, 1)
    )


def cited_story_ids(answer: str, chunks: list[dict]) -> list[int]:
    """Map [n] citations back to story ids, ignoring numbers out of range."""
    nums = {int(m) for m in re.findall(r"\[(\d+)\]", answer)}
    return sorted({chunks[i - 1]["story_id"] for i in nums if 1 <= i <= len(chunks)})


def invalid_citations(answer: str, chunks: list[dict]) -> list[int]:
    """Citation numbers pointing at comments that were never provided — i.e. invented."""
    nums = {int(m) for m in re.findall(r"\[(\d+)\]", answer)}
    return sorted(n for n in nums if not 1 <= n <= len(chunks))



def _call(client, model: str, prompt: str) -> tuple[str, int, int]:
    reply = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    usage = reply.usage
    return (reply.choices[0].message.content.strip(),
            usage.prompt_tokens, usage.completion_tokens)


def retrieve(question: str) -> tuple[str, list[dict], int, int]:
    """Rewrite → search → cap. Returns (rewritten_query, chunks, in_tokens, out_tokens)."""
    from openai import OpenAI

    client = OpenAI()
    model = os.environ["OPENAI_MODEL"]

    query, in1, out1 = _call(client, model,
                             REWRITE_PROMPT.format(question=question))
    chunks = cap_per_thread(hybrid_rerank(query, n=N_RETRIEVE))
    return query, chunks, in1, out1

def answer_from_chunks(question: str,
                       chunks: list[dict],
                       variant: str = "cot") -> Answer:
    """Prompt + LLM call only — no search. Step 8 calls this with frozen chunks."""
    from openai import OpenAI

    started = time.perf_counter()
    client = OpenAI()
    model = os.environ["OPENAI_MODEL"]

    text, tokens_in, tokens_out = _call(client, model, PROMPTS[variant].format(
        question=question, context=build_context(chunks)))

    return Answer(
        question=question,
        rewritten_query="",          # no rewrite on this path
        answer=text,
        strategy=STRATEGY,
        prompt_variant=variant,
        chunks=chunks,
        cited_story_ids=cited_story_ids(text, chunks),
        input_tokens=tokens_in,
        output_tokens=tokens_out,
        cost_usd=tokens_in * IN_COST + tokens_out * OUT_COST,
        latency_ms=int((time.perf_counter() - started) * 1000),
    )


def answer(question: str, variant: str = "cot") -> Answer:
    """The full app path: rewrite → search → cap → answer.

    `cot` is Step 8's measured winner (coverage 4.06 vs 3.41 and 3.29).
    """
    started = time.perf_counter()

    query, chunks, in1, out1 = retrieve(question)
    result = answer_from_chunks(question, chunks, variant)

    tokens_in = result.input_tokens + in1
    tokens_out = result.output_tokens + out1
    return replace(
        result,
        rewritten_query=query,
        input_tokens=tokens_in,
        output_tokens=tokens_out,
        cost_usd=tokens_in * IN_COST + tokens_out * OUT_COST,
        latency_ms=int((time.perf_counter() - started) * 1000),
    )


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    variant = "cot"
    if args and args[0] in PROMPTS:
        variant, *args = args

    q = " ".join(args) or "is Amazon a good place to start a career"
    a = answer(q, variant)
    print(f"variant:   {variant}")
    print(f"rewritten: {a.rewritten_query}\n")
    print(a.answer)
    print(f"\nsources: {a.cited_story_ids}")
    print(f"{len(a.chunks)} chunks from "
          f"{len({c['story_id'] for c in a.chunks})} threads")
    print(f"{a.input_tokens} in / {a.output_tokens} out  "
          f"${a.cost_usd:.5f}  {a.latency_ms} ms")