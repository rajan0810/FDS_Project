import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from .config import CONFIG

def adaptive_select(retrieved_chunks: pd.DataFrame, 
                    chunk_embeddings: np.ndarray, 
                    global_metadata: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Prototype adaptive context selection heuristic.
    Stops when marginal relevance is too low, or redundancy is too high, or budget met.
    """
    min_chunks = CONFIG["MIN_CHUNKS"]
    max_chunks = CONFIG["MAX_CHUNKS"]
    min_relevance = CONFIG["MIN_RELEVANCE_SCORE"]
    marginal_thresh = CONFIG["MARGINAL_SCORE_THRESHOLD"]
    redundancy_thresh = CONFIG["REDUNDANCY_THRESHOLD"]
    
    selected_indices = []
    selected_embeddings = []
    removed_as_redundant = 0
    
    for idx, row in retrieved_chunks.iterrows():
        global_idx = row.name # The index in retrieved_chunks should be the global index
        
        # Stop condition 1: Max chunks reached
        if len(selected_indices) >= max_chunks:
            break
            
        # Stop condition 2: Marginal relevance too low
        if row["similarity_score"] < min_relevance:
            if len(selected_indices) >= min_chunks:
                break
                
        # If it's the first chunk, just add it
        if len(selected_indices) == 0:
            selected_indices.append(global_idx)
            selected_embeddings.append(chunk_embeddings[global_idx])
            continue
            
        # Stop condition 3 & Redundancy Check
        curr_emb = chunk_embeddings[global_idx].reshape(1, -1)
        sims_to_selected = cosine_similarity(curr_emb, selected_embeddings)[0]
        
        max_sim_to_selected = np.max(sims_to_selected)
        
        if max_sim_to_selected >= redundancy_thresh:
            removed_as_redundant += 1
            continue
            
        # Accept chunk
        selected_indices.append(global_idx)
        selected_embeddings.append(chunk_embeddings[global_idx])
        
    final_selected = global_metadata.loc[selected_indices].copy()
    
    # Sort by original position
    final_selected = final_selected.sort_values(by="start_paragraph")
    
    return final_selected, removed_as_redundant
