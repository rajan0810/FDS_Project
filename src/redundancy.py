import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from .config import CONFIG

def flag_redundant_chunks(retrieved_metadata: pd.DataFrame, chunk_embeddings: np.ndarray, threshold: float = None) -> List[Tuple[str, str, float]]:
    if threshold is None:
        threshold = CONFIG["REDUNDANCY_THRESHOLD"]
        
    redundant_pairs = []
    
    if len(retrieved_metadata) < 2:
        return redundant_pairs
        
    # Get embeddings for just the retrieved chunks
    retrieved_indices = retrieved_metadata.index
    # We assume the chunk_embeddings passed here are aligned with the global chunk index,
    # and the retrieved_metadata index contains the original global indices.
    # Wait, in retrieve(), retrieved_metadata is a copy of original metadata. 
    # Let's just use the chunk IDs to find the correct embeddings or re-embed.
    # Actually, it's easier to pass the embeddings of the retrieved chunks directly.
    pass
    
    return redundant_pairs

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
