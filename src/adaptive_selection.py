import pandas as pd
import numpy as np
from typing import Tuple
from .config import CONFIG
from .redundancy import flag_redundant_chunks

def select_by_relevance(reranked_chunks: pd.DataFrame,
                         min_relevance: float = None,
                         max_chunks: int = None,
                         min_chunks: int = None) -> pd.DataFrame:
    """
    Adaptive Context Selection stage (relevance cutoff only).

    Walks the reranked candidates in order and stops when either:
      - `max_chunks` have been accepted, or
      - the chunk's original bi-encoder `similarity_score` drops below `min_relevance`,
        provided at least `min_chunks` have already been accepted.

    Note: the *relevance* cutoff is applied against the bi-encoder cosine similarity score
    (bounded in [0, 1], matching MIN_RELEVANCE_SCORE's calibration), not the cross-encoder
    rerank score — the reranker only changes candidate *order/pool*, not the relevance
    threshold semantics.
    """
    if min_relevance is None:
        min_relevance = CONFIG["MIN_RELEVANCE_SCORE"]
    if max_chunks is None:
        max_chunks = CONFIG["MAX_CHUNKS"]
    if min_chunks is None:
        min_chunks = CONFIG["MIN_CHUNKS"]

    keep_positions = []
    for pos in range(len(reranked_chunks)):
        if len(keep_positions) >= max_chunks:
            break

        row = reranked_chunks.iloc[pos]
        if row["similarity_score"] < min_relevance and len(keep_positions) >= min_chunks:
            break

        keep_positions.append(pos)

    return reranked_chunks.iloc[keep_positions].copy()

def remove_redundant_chunks(selected_chunks: pd.DataFrame,
                             chunk_embeddings: np.ndarray,
                             redundancy_thresh: float = None) -> Tuple[pd.DataFrame, int]:
    """
    Redundancy Removal stage. Delegates to `src.redundancy.flag_redundant_chunks`.
    `chunk_embeddings` must be the full corpus embedding matrix; rows are looked up by the
    global index preserved on `selected_chunks`.
    """
    if len(selected_chunks) == 0:
        return selected_chunks.copy(), 0

    candidate_embeddings = chunk_embeddings[selected_chunks.index.to_numpy()]
    return flag_redundant_chunks(selected_chunks, candidate_embeddings, redundancy_thresh)

def order_context(chunks: pd.DataFrame) -> pd.DataFrame:
    """
    Context Ordering stage. Arranges the final selected evidence chunks in original
    document order (by start paragraph), which mitigates "lost in the middle" effects
    since the LLM sees evidence in the same narrative order as the source document.
    """
    if len(chunks) == 0:
        return chunks.copy()
    return chunks.sort_values(by="start_paragraph")

def adaptive_select(retrieved_chunks: pd.DataFrame,
                     chunk_embeddings: np.ndarray,
                     global_metadata: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Convenience wrapper chaining the three discrete stages:
    select_by_relevance -> remove_redundant_chunks -> order_context.
    Kept for backward compatibility with existing call sites.
    """
    relevance_selected = select_by_relevance(retrieved_chunks)
    deduplicated, removed_as_redundant = remove_redundant_chunks(relevance_selected, chunk_embeddings)
    final_selected = order_context(deduplicated)

    return final_selected, removed_as_redundant
