import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from .config import CONFIG

def flag_redundant_chunks(candidates: pd.DataFrame, candidate_embeddings: np.ndarray, threshold: float = None) -> Tuple[pd.DataFrame, int]:
    """
    Redundancy Removal pipeline stage.

    Given a set of already relevance-ranked candidate chunks (highest relevance first)
    and their embeddings (aligned row-for-row with `candidates`), greedily keep a chunk
    unless its cosine similarity to any already-kept chunk is >= threshold. This removes
    near-duplicate/semantically-redundant chunks while preserving the higher-relevance
    chunk of any redundant pair.

    Returns (deduplicated_dataframe, num_removed_as_redundant).
    """
    if threshold is None:
        threshold = CONFIG["REDUNDANCY_THRESHOLD"]

    if len(candidates) < 2:
        return candidates.copy(), 0

    keep_positions = []
    kept_embeddings = []
    removed = 0

    for pos in range(len(candidates)):
        emb = candidate_embeddings[pos].reshape(1, -1)
        if kept_embeddings:
            sims = cosine_similarity(emb, np.vstack(kept_embeddings))[0]
            if np.max(sims) >= threshold:
                removed += 1
                continue
        keep_positions.append(pos)
        kept_embeddings.append(candidate_embeddings[pos])

    deduplicated = candidates.iloc[keep_positions].copy()
    return deduplicated, removed

def calculate_pairwise_redundancy(retrieved_embeddings: np.ndarray, retrieved_ids: List[str], threshold: float = None) -> List[Tuple[str, str, float, bool]]:
    if threshold is None:
        threshold = CONFIG["REDUNDANCY_THRESHOLD"]
        
    pairs_info = []
    
    if len(retrieved_embeddings) < 2:
        return pairs_info
        
    sim_matrix = cosine_similarity(retrieved_embeddings)
    
    for i in range(len(sim_matrix)):
        for j in range(i + 1, len(sim_matrix)):
            score = float(sim_matrix[i, j])
            is_redundant = score >= threshold
            pairs_info.append((retrieved_ids[i], retrieved_ids[j], score, is_redundant))
                
    return pairs_info
