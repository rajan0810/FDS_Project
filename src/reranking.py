import pandas as pd
from typing import List
from sentence_transformers import CrossEncoder
from .config import CONFIG

class Reranker:
    """
    Context Reranking pipeline stage.

    Wraps a cross-encoder model that jointly scores (query, chunk_text) pairs, which is
    strictly more accurate than the bi-encoder cosine-similarity score used for initial
    retrieval (that score is computed independently for the query and chunk, then compared;
    a cross-encoder attends over both texts together).
    """

    def __init__(self, model_name: str = None):
        if model_name is None:
            model_name = CONFIG["RERANK_MODEL"]
        print(f"Loading reranker model: {model_name}...")
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: pd.DataFrame, top_n: int = None) -> pd.DataFrame:
        if top_n is None:
            top_n = CONFIG["RERANK_TOP_N"]

        if len(candidates) == 0:
            return candidates.copy()

        pairs = [(query, text) for text in candidates["text"].tolist()]
        scores = self.model.predict(pairs)

        reranked = candidates.copy()
        reranked["rerank_score"] = scores
        reranked = reranked.sort_values(by="rerank_score", ascending=False)

        return reranked.head(top_n)
