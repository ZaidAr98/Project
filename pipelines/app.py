import psycopg
import streamlit as st

from common import DSN, get_model, get_reranker
from rag import PER_THREAD, Answer, answer

HN_ITEM = "https://news.ycombinator.com/item?id={}"

EXAMPLES = [
    "Should I do a masters degree or take the job offer?",
    "Do experienced developers think AI coding tools actually help?",
    "I have 15 years experience and can't get interviews - what do people suggest?",
    "Is it still worth learning to write code by hand?",
]

INSERT_CONVERSATION = """
INSERT INTO conversations (question, rewritten_query, answer, strategy,
                           prompt_variant, cited_story_ids, input_tokens,
                           output_tokens, cost_usd, latency_ms)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
RETURNING id
"""

INSERT_FEEDBACK = "INSERT INTO feedback (conversation_id, vote) VALUES (%s, %s)"


def log_conversation(a: Answer) -> int:
    """Write one exchange, returning its id so feedback can reference it.

    Answer's fields map 1:1 onto the table, so this is a straight unpack of
    everything except `chunks` (rendered, never stored).
    """
    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute(INSERT_CONVERSATION, (
            a.question, a.rewritten_query, a.answer, a.strategy,
            a.prompt_variant, a.cited_story_ids, a.input_tokens,
            a.output_tokens, a.cost_usd, a.latency_ms,
        ))
        return cur.fetchone()[0]


def log_feedback(conversation_id: int, vote: int) -> None:
    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute(INSERT_FEEDBACK, (conversation_id, vote))


@st.cache_resource(show_spinner="Loading the embedding and reranking models...")
def warm_models() -> bool:
    """Build both fastembed models once, at startup rather than on first query.

    get_reranker() is lazy. Measured 2026-08-16: a cold first call spent
    40,073 ms downloading ~220 MB *inside* answer()'s timed region, which would
    land in conversations.latency_ms and dominate every Grafana latency panel.
    """
    get_model()
    get_reranker()
    return True


st.set_page_config(page_title="What does HN think?", page_icon="💬")
warm_models()

st.session_state.setdefault("result", None)
st.session_state.setdefault("conversation_id", None)
st.session_state.setdefault("voted", False)


with st.sidebar:
    st.subheader("Try one of these")
    for example in EXAMPLES:
        if st.button(example):
            st.session_state.question = example

    st.divider()
    st.caption(
        "**Corpus:** 226 Ask HN threads from June-July 2026, 6,281 comment "
        "chunks.\n\n"
        "**Retrieval:** `hybrid_rerank`, the measured winner of four "
        "strategies.\n\n"
        "**Prompt:** `cot`, the measured winner of three variants.\n\n"
        f"At most {PER_THREAD} comments from any one thread reach the prompt, "
        "so an answer reflects several threads rather than one loud one."
    )

st.title("What does Hacker News think?")
st.caption(
    "Ask about careers, developer tooling, or AI's effect on how people work. "
    "Answers report what commenters actually said - including where they "
    "disagree - never a single view presented as settled."
)

with st.form("ask"):
    question = st.text_input(
        "Your question",
        key="question",
        placeholder="Should I do a masters degree or take the job offer?",
    )
    submitted = st.form_submit_button("Ask", type="primary")


if submitted and question.strip():
    with st.spinner("Searching 6,281 comments and writing the answer..."):
        result = answer(question)
    st.session_state.result = result
    st.session_state.conversation_id = log_conversation(result)
    st.session_state.voted = False

result: Answer | None = st.session_state.result

if result is not None:
    st.markdown(result.answer)

    cited = set(result.cited_story_ids)
    threads = len({c["story_id"] for c in result.chunks})

    with st.expander(
        f"Sources - {len(result.chunks)} comments from {threads} threads"
    ):
        for i, chunk in enumerate(result.chunks, 1):
            mark = " · **cited**" if chunk["story_id"] in cited else ""
            link = HN_ITEM.format(chunk["comment_id"])
            st.markdown(f"[{i}] [{chunk['story_title']}]({link}){mark}")
         
            st.markdown("> " + chunk["chunk_text"].replace("\n", "\n> "))

    st.divider()

    if st.session_state.voted:
        st.caption("Thanks - your vote was recorded.")
    else:
        st.caption("Was this answer useful?")
        up, down, _ = st.columns([1, 1, 8])
        if up.button("👍"):
            log_feedback(st.session_state.conversation_id, 1)
            st.session_state.voted = True
            st.rerun()
        if down.button("👎"):
            log_feedback(st.session_state.conversation_id, -1)
            st.session_state.voted = True
            st.rerun()

    st.caption(
        f"Searched for *{result.rewritten_query}* · "
        f"{result.latency_ms:,} ms · "
        f"{result.input_tokens:,} in / {result.output_tokens:,} out tokens · "
        f"${result.cost_usd:.5f} · "
        f"`{result.strategy}` · `{result.prompt_variant}`"
    )
