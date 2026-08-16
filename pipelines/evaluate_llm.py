"""Step 8 — LLM evaluation.

Three prompt variants answer the same questions from the SAME frozen chunks,
then an LLM judge scores each answer on groundedness, coverage and relevance.

    uv run python pipelines/evaluate_llm.py freeze   # search once, save chunks
    uv run python pipelines/evaluate_llm.py answer   # 3 prompts x N questions
    uv run python pipelines/evaluate_llm.py judge    # score every answer
    uv run python pipelines/evaluate_llm.py score    # the results table

Freezing retrieval is what makes this an experiment about prompts rather than
about search noise: every variant reads byte-identical context.
"""

import collections
import csv
import json
import os
import re
import sys
from pathlib import Path

from rag import (PROMPTS, _call, answer_from_chunks, build_context,
                 invalid_citations, retrieve)

FROZEN = Path("data/frozen_context.json")
ANSWERS = Path("data/llm_answers.csv")
SCORES = Path("data/llm_scores.csv")

KEEP = ("id", "story_id", "comment_id", "story_title", "chunk_text")

FIELDS = ["question", "variant", "answer", "n_chunks", "n_cited",
          "invalid_citations", "input_tokens", "output_tokens",
          "cost_usd", "latency_ms"]

SCORE_FIELDS = ["question", "variant", "groundedness", "coverage", "relevance"]

QUESTIONS = [
    # careers
    "I have 15 years of experience and can't get interviews — what do people suggest?",
    "Is it worth leaving a stable job to join a startup?",
    "Do people regret moving into management instead of staying technical?",
    "How do developers deal with burnout?",
    "Is remote work better or worse for a junior developer?",
    "Are coding interviews a fair way to hire people?",
    "Is a computer science degree still worth it?",
    "Is it better to job hop or stay at one company for years?",
    # AI at work
    "Do experienced developers think AI coding tools actually help?",
    "Is it still worth learning to write code by hand?",
    "Will AI replace junior developers?",
    "Do people trust AI-generated code in production?",
    "Has AI made hiring and interviewing worse?",
    # tooling
    "Is self-hosting your own services worth the effort?",
    "What do people think about microservices these days?",
    "Which editor or IDE do developers actually prefer?",
    "Is Postgres enough, or do you need a separate search engine?",
]

JUDGE_PROMPT = """You are grading a summary of forum comments. The comments are
opinions, so do NOT judge whether they are correct. Judge only whether the
summary reports them honestly.

Score each dimension from 1 to 5:

groundedness — is every claim in the summary traceable to one of the comments?
  5 = every claim is supported by a comment. 1 = it states things nobody said.
coverage — does the summary show the range of positions in the comments?
  5 = every distinct position appears, roughly in proportion to how many
  comments held it. 1 = it reports one side and hides the rest.
relevance — does the summary answer the question that was asked?
  5 = answers it directly. 1 = answers something else.

Return JSON only, no other text:
{{"groundedness": <1-5>, "coverage": <1-5>, "relevance": <1-5>}}

Question: {question}

Comments:
{context}

Summary:
{answer}"""


def stage_freeze() -> None:
    """Run the real search path once per question and save the chunks."""
    frozen = []
    for i, question in enumerate(QUESTIONS, 1):
        query, chunks, _, _ = retrieve(question)
        frozen.append({
            "question": question,
            "rewritten_query": query,
            "chunks": [{k: c[k] for k in KEEP} for c in chunks],
        })
        threads = len({c["story_id"] for c in chunks})
        print(f"{i:2}. {len(chunks)} chunks / {threads} threads  — {question}")

    FROZEN.parent.mkdir(parents=True, exist_ok=True)
    FROZEN.write_text(json.dumps(frozen, indent=2), encoding="utf-8")

    thin = sum(1 for f in frozen if len(f["chunks"]) < 5)
    print(f"\nwrote {FROZEN} — {len(frozen)} questions, "
          f"{thin} with fewer than 5 chunks")


def stage_answer() -> None:
    """Every variant answers every question from the SAME frozen chunks."""
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))

    done = set()
    if ANSWERS.exists():
        with ANSWERS.open(encoding="utf-8", newline="") as f:
            done = {(r["question"], r["variant"]) for r in csv.DictReader(f)}

    new = not ANSWERS.exists()
    with ANSWERS.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            writer.writeheader()

        for item in frozen:
            question, chunks = item["question"], item["chunks"]
            for variant in PROMPTS:
                if (question, variant) in done:
                    print(f"skip  {variant:9} {question[:50]}")
                    continue

                a = answer_from_chunks(question, chunks, variant)
                bad = invalid_citations(a.answer, chunks)

                writer.writerow({
                    "question": question,
                    "variant": variant,
                    "answer": a.answer,
                    "n_chunks": len(chunks),
                    "n_cited": len(a.cited_story_ids),
                    "invalid_citations": " ".join(map(str, bad)),
                    "input_tokens": a.input_tokens,
                    "output_tokens": a.output_tokens,
                    "cost_usd": f"{a.cost_usd:.6f}",
                    "latency_ms": a.latency_ms,
                })
                f.flush()
                flag = f"  ⚠ invalid {bad}" if bad else ""
                print(f"ok    {variant:9} {question[:50]}{flag}")

    print(f"\nwrote {ANSWERS}")


def _parse_scores(text: str) -> dict:
    """Models like to wrap JSON in prose or ``` fences — take the outermost object."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"no JSON in judge reply: {text[:200]}")
    return json.loads(m.group(0))


def stage_judge() -> None:
    """One call per answer. The judge sees the question, the comments AND the answer.

    It must see the comments: without them it would have to decide who is right,
    which is meaningless for opinions. With them the question becomes "does this
    summary match its sources", which does have an answer.
    """
    from openai import OpenAI

    frozen = {f["question"]: f["chunks"]
              for f in json.loads(FROZEN.read_text(encoding="utf-8"))}
    with ANSWERS.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    done = set()
    if SCORES.exists():
        with SCORES.open(encoding="utf-8", newline="") as f:
            done = {(r["question"], r["variant"]) for r in csv.DictReader(f)}

    client = OpenAI()
    model = os.environ["OPENAI_MODEL"]

    new = not SCORES.exists()
    with SCORES.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SCORE_FIELDS)
        if new:
            writer.writeheader()

        for r in rows:
            if (r["question"], r["variant"]) in done:
                print(f"skip  {r['variant']:9} {r['question'][:45]}")
                continue

            reply, _, _ = _call(client, model, JUDGE_PROMPT.format(
                question=r["question"],
                context=build_context(frozen[r["question"]]),
                answer=r["answer"],
            ))
            s = _parse_scores(reply)

            writer.writerow({
                "question": r["question"],
                "variant": r["variant"],
                "groundedness": s["groundedness"],
                "coverage": s["coverage"],
                "relevance": s["relevance"],
            })
            f.flush()
            print(f"ok    {r['variant']:9} g={s['groundedness']} "
                  f"c={s['coverage']} r={s['relevance']}  {r['question'][:40]}")

    print(f"\nwrote {SCORES}")


def stage_score() -> None:
    """The results table. Ship the winner as rag.py's default variant."""
    with SCORES.open(encoding="utf-8", newline="") as f:
        scores = list(csv.DictReader(f))
    with ANSWERS.open(encoding="utf-8", newline="") as f:
        answers = {(r["question"], r["variant"]): r for r in csv.DictReader(f)}

    agg = collections.defaultdict(list)
    for s in scores:
        agg[s["variant"]].append(s)

    print(f"{'variant':10}{'ground':>8}{'cover':>8}{'relev':>8}"
          f"{'mean':>8}{'cited':>8}{'bad cites':>11}")
    for variant, rs in agg.items():
        n = len(rs)
        g = sum(int(r["groundedness"]) for r in rs) / n
        c = sum(int(r["coverage"]) for r in rs) / n
        v = sum(int(r["relevance"]) for r in rs) / n
        cited = sum(int(answers[(r["question"], r["variant"])]["n_cited"])
                    for r in rs) / n
        bad = sum(1 for r in rs
                  if answers[(r["question"], r["variant"])]["invalid_citations"].strip())
        print(f"{variant:10}{g:>8.2f}{c:>8.2f}{v:>8.2f}"
              f"{(g + c + v) / 3:>8.2f}{cited:>8.2f}{bad:>11}")


STAGES = {
    "freeze": stage_freeze,
    "answer": stage_answer,
    "judge": stage_judge,
    "score": stage_score,
}

if __name__ == "__main__":
    STAGES[sys.argv[1]]()
