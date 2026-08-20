import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity

def retrieve(query_embedding: np.ndarray, chunk_embeddings: np.ndarray, chunk_metadata: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    if len(chunk_embeddings) == 0:
        return pd.DataFrame()
        
    query_embedding = query_embedding.reshape(1, -1)
    similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
    
    top_indices = np.argsort(similarities)[::-1][:k]
    
    retrieved = chunk_metadata.iloc[top_indices].copy()
    retrieved["similarity_score"] = similarities[top_indices]
    retrieved["rank"] = np.arange(1, len(retrieved) + 1)
    
    return retrieved

def map_gold_evidence_to_chunks(question_evidence: List[str], paper_paragraphs: List[Dict[str, Any]], chunk_metadata: pd.DataFrame) -> List[str]:
    """
    Given a list of gold evidence strings, find the corresponding chunk IDs.
    For this prototype, we'll try basic string matching to find the paragraph, then get the chunk containing it.
    """
    gold_chunk_ids = set()
    
    # Fast exact match fallback
    for evidence_str in question_evidence:
        evidence_str_clean = evidence_str.strip()
        if not evidence_str_clean:
            continue
            
        for _, row in chunk_metadata.iterrows():
            if evidence_str_clean in row["text"]:
                gold_chunk_ids.add(row["chunk_id"])
                
    return list(gold_chunk_ids)
