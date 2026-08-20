import os
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple
from .config import CONFIG, BASE_DIR

EMBEDDINGS_DIR = BASE_DIR / "data" / "embeddings"
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

class EmbeddingManager:
    def __init__(self, model_name=None):
        if model_name is None:
            model_name = CONFIG["EMBEDDING_MODEL"]
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

def embed_chunks(chunks: List[Dict[str, Any]], manager: EmbeddingManager) -> Tuple[np.ndarray, pd.DataFrame]:
    if not chunks:
        return np.array([]), pd.DataFrame()
        
    df = pd.DataFrame(chunks)
    texts = df["text"].tolist()
    
    embeddings = manager.get_embeddings(texts)
    return embeddings, df

def save_embeddings(embeddings: np.ndarray, metadata: pd.DataFrame, prefix: str):
    np.save(EMBEDDINGS_DIR / f"{prefix}_embeddings.npy", embeddings)
    metadata.to_csv(EMBEDDINGS_DIR / f"{prefix}_metadata.csv", index=False)
    
def load_embeddings(prefix: str) -> Tuple[np.ndarray, pd.DataFrame]:
    emb_path = EMBEDDINGS_DIR / f"{prefix}_embeddings.npy"
    meta_path = EMBEDDINGS_DIR / f"{prefix}_metadata.csv"
    
    if emb_path.exists() and meta_path.exists():
        embeddings = np.load(emb_path)
        metadata = pd.read_csv(meta_path)
        # Parse paragraph_ids back to list
        if "paragraph_ids" in metadata.columns:
            import ast
            metadata["paragraph_ids"] = metadata["paragraph_ids"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        return embeddings, metadata
    return None, None
