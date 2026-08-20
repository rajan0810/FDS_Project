import pandas as pd
from typing import List, Dict, Any
from .config import CONFIG

def create_chunks(paragraphs: List[Dict[str, Any]], chunk_size: int = None, chunk_overlap: int = None) -> List[Dict[str, Any]]:
    if chunk_size is None:
        chunk_size = CONFIG["CHUNK_SIZE_TOKENS"]
    if chunk_overlap is None:
        chunk_overlap = CONFIG["CHUNK_OVERLAP_TOKENS"]
        
    chunks = []
    current_chunk = []
    current_token_count = 0
    chunk_idx = 0
    
    if not paragraphs:
        return chunks
        
    paper_id = paragraphs[0]["paper_id"]
    
    i = 0
    while i < len(paragraphs):
        para = paragraphs[i]
        
        # If a single paragraph is larger than chunk size, we'll still add it as one chunk 
        # for simplicity in this prototype, or we could split it. Let's just add it.
        if current_token_count + para["token_count"] > chunk_size and len(current_chunk) > 0:
            # Finalize current chunk
            chunks.append(_finalize_chunk(current_chunk, chunk_idx, paper_id))
            chunk_idx += 1
            
            # Start new chunk with overlap
            # Backtrack to find overlap
            overlap_tokens = 0
            overlap_chunk = []
            for p in reversed(current_chunk):
                if overlap_tokens + p["token_count"] <= chunk_overlap:
                    overlap_chunk.insert(0, p)
                    overlap_tokens += p["token_count"]
                else:
                    break
                    
            current_chunk = overlap_chunk
            current_token_count = overlap_tokens
            
        current_chunk.append(para)
        current_token_count += para["token_count"]
        i += 1
        
    if current_chunk:
        chunks.append(_finalize_chunk(current_chunk, chunk_idx, paper_id))
        
    return chunks

def _finalize_chunk(paragraphs: List[Dict], chunk_idx: int, paper_id: str) -> Dict[str, Any]:
    text = "\n\n".join([p["text"] for p in paragraphs])
    token_count = sum(p["token_count"] for p in paragraphs)
    start_para = paragraphs[0]["global_paragraph_index"]
    end_para = paragraphs[-1]["global_paragraph_index"]
    section_names = list(set([p["section_name"] for p in paragraphs]))
    
    # Store IDs for exact mapping
    paragraph_ids = [p["paragraph_id"] for p in paragraphs]
    
    return {
        "chunk_id": f"{paper_id}_C{chunk_idx:03d}",
        "paper_id": paper_id,
        "section_name": " | ".join(section_names),
        "start_paragraph": start_para,
        "end_paragraph": end_para,
        "paragraph_ids": paragraph_ids,
        "text": text,
        "token_count": token_count
    }
