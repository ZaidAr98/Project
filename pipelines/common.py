import os

from dotenv import load_dotenv

load_dotenv()

DSN = (
    f"postgresql://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}"
    f"@{os.getenv('POSTGRES_HOST', 'localhost')}:{os.getenv('POSTGRES_PORT', '5432')}"
    f"/{os.environ['POSTGRES_DB']}"
)
RERANK_MODEL = os.getenv("RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

_model = None
_reranker = None

def get_model():
    global _model
    if _model is None:
        from fastembed import TextEmbedding   # slow import, keep it lazy
        _model = TextEmbedding(EMBED_MODEL)
    return _model


def to_vector(values) -> str:
    return "[" + ",".join(map(str, values)) + "]"


def embed_query(query: str) -> str:
  
    return to_vector(next(iter(get_model().query_embed([query]))))


def get_reranker():
    global _reranker
    if _reranker is None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder  # slow import
        _reranker = TextCrossEncoder(model_name=RERANK_MODEL)
    return _reranker