
import csv
import json
import os
import sys
from pathlib import Path

import psycopg
from search import STRATEGIES
from common import DSN

GROUND_TRUTH = Path("data/ground_truth.csv")
SAMPLE_SIZE = 100
QUESTIONS_PER_COMMENT = 3

# md5 ordering, not random() — a reviewer re-running this gets the identical
# sample. Reproducibility is a graded criterion.
SAMPLE_SQL = """
SELECT comment_id, story_id, chunk_text
FROM chunks
WHERE chunk_index = 0 AND length(chunk_text) > 400
ORDER BY md5(comment_id::text)
LIMIT %s
"""

PROMPT = """Below is one person's comment from an online discussion about
careers, developer tooling, or AI's effect on work.

Write {k} different questions that a real person might type into a search box,
where THIS comment would be a genuinely useful answer.

Rules:
- Paraphrase the *idea*. Do not reuse the comment's distinctive words or
  phrases — if the comment says "crippling debt", do not write "crippling debt".
- Write the way people actually type: short, plain, lowercase, no polish.
- Each question must be answerable by this comment in particular, not by any
  comment on the general topic.
- Do not mention comments, threads, forums, or Hacker News.

Return JSON: {{"questions": ["...", "...", "..."]}}

Comment:
{comment}
"""

N = 5   # all metrics are @5


def first_hit(results: list[dict], key: str, target) -> int | None:
    """1-based rank of the first correct result, or None if it never appeared."""
    for rank, row in enumerate(results, 1):
        if row[key] == target:
            return rank
    return None


def hit_rate(ranks: list[int | None]) -> float:
    """Fraction of questions where the right answer showed up at all."""
    return sum(1 for r in ranks if r is not None) / len(ranks)


def mrr(ranks: list[int | None]) -> float:
    """Mean reciprocal rank — rewards being right *early*, not merely present."""
    return sum(1 / r for r in ranks if r is not None) / len(ranks)

def load_ground_truth() -> list[dict]:
    with GROUND_TRUTH.open(newline="", encoding="utf-8") as f:
        return [
            {"question": r["question"],
             "story_id": int(r["story_id"]),
             "comment_id": int(r["comment_id"])}
            for r in csv.DictReader(f)
        ]


def stage_score() -> None:
    """Run every question through every strategy, scored two ways.

    Strict (comment_id) only counts the comment the question was written from.
    Loose (story_id) counts any chunk from the right thread — generous, and
    expected to saturate, since one thread's chunks all share a prepended title.
    Strict is the honest number and the one that picks the winner.
    """
    questions = load_ground_truth()
    comments = len({q["comment_id"] for q in questions})
    print(f"{len(questions)} questions from {comments} comments, @{N}\n")

    header = (f"{'strategy':<10}{'strict HR':>11}{'strict MRR':>12}"
              f"{'loose HR':>11}{'loose MRR':>12}")
    print(header)
    print("-" * len(header))

    for name, search in STRATEGIES.items():
        strict: list[int | None] = []
        loose: list[int | None] = []
        for q in questions:
            results = search(q["question"], N)
            strict.append(first_hit(results, "comment_id", q["comment_id"]))
            loose.append(first_hit(results, "story_id", q["story_id"]))

        print(f"{name:<10}{hit_rate(strict):>11.3f}{mrr(strict):>12.3f}"
              f"{hit_rate(loose):>11.3f}{mrr(loose):>12.3f}")


def stage_generate(sample: int = SAMPLE_SIZE, k: int = QUESTIONS_PER_COMMENT) -> None:
    """Ask the LLM for questions each sampled comment answers.

    Appends as it goes and skips comments already in the CSV, so a crash or a
    Ctrl-C costs only the current call — same re-runnable design as build_chunks.
    """
    from openai import OpenAI

    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute(SAMPLE_SQL, (sample,))
        comments = cur.fetchall()

    client = OpenAI()
    model = os.environ["OPENAI_MODEL"]

    GROUND_TRUTH.parent.mkdir(exist_ok=True)
    existed = GROUND_TRUTH.exists()
    done: set[int] = set()
    if existed:
        with GROUND_TRUTH.open(newline="", encoding="utf-8") as f:
            done = {int(row["comment_id"]) for row in csv.DictReader(f)}
        print(f"resuming — {len(done)} comments already done")

    with GROUND_TRUTH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not existed:
            writer.writerow(["question", "story_id", "comment_id"])

        for i, (comment_id, story_id, text) in enumerate(comments, 1):
            if comment_id in done:
                continue
            try:
                reply = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user",
                               "content": PROMPT.format(k=k, comment=text)}],
                    response_format={"type": "json_object"},
                )
                questions = json.loads(reply.choices[0].message.content)["questions"]
            except Exception as exc:                # one bad comment must not
                print(f"{i}/{len(comments)}  SKIPPED {comment_id}: {exc}")
                continue                            # cost the other 99

            for q in questions:
                writer.writerow([q, story_id, comment_id])
            f.flush()
            print(f"{i}/{len(comments)}  +{len(questions)}  (comment {comment_id})")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "generate"
    if stage == "generate":
        stage_generate()
    elif stage == "score":
        stage_score()
    else:
        sys.exit(f"unknown stage: {stage}")
