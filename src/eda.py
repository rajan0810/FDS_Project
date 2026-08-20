import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List, Dict, Any
from .config import BASE_DIR

EDA_OUT = BASE_DIR / "outputs" / "eda"
FIG_OUT = EDA_OUT / "figures"
EDA_OUT.mkdir(parents=True, exist_ok=True)
FIG_OUT.mkdir(parents=True, exist_ok=True)

def analyze_context_lengths(paper_records, question_records, chunk_records):
    papers_data = []
    for p in paper_records:
        total_tokens = sum(para["token_count"] for para in p.paragraphs)
        papers_data.append({"paper_id": p.paper_id, "tokens": total_tokens})
    
    df_papers = pd.DataFrame(papers_data)
    
    chunks_data = []
    for c in chunk_records:
        chunks_data.append({"chunk_id": c["chunk_id"], "tokens": c["token_count"]})
    df_chunks = pd.DataFrame(chunks_data)
    
    # Save CSVs
    df_papers.to_csv(EDA_OUT / "paper_lengths.csv", index=False)
    
    # Plot paper lengths
    plt.figure(figsize=(10, 6))
    sns.histplot(df_papers["tokens"], bins=30, kde=True)
    plt.title("Paper Token Length Distribution")
    plt.xlabel("Tokens")
    plt.ylabel("Count")
    plt.savefig(FIG_OUT / "paper_length_distribution.png")
    plt.close()

    # Plot chunk lengths
    plt.figure(figsize=(10, 6))
    sns.histplot(df_chunks["tokens"], bins=30, kde=True)
    plt.title("Chunk Token Length Distribution")
    plt.xlabel("Tokens")
    plt.ylabel("Count")
    plt.savefig(FIG_OUT / "chunk_length_distribution.png")
    plt.close()
    
    return df_papers.describe()

def analyze_evidence_positions(paper_records, question_records, chunk_metadata):
    # This is a simplified position analysis
    positions = []
    for q in question_records:
        if not q.evidence: continue
        
        # We need the global index of the paragraphs that match the evidence
        paper = next((p for p in paper_records if p.paper_id == q.paper_id), None)
        if not paper: continue
        
        total_paras = len(paper.paragraphs)
        if total_paras == 0: continue
        
        for ev in q.evidence:
            ev_clean = ev.strip()
            # find paragraph index
            for para in paper.paragraphs:
                if ev_clean in para["text"]:
                    pos = para["global_paragraph_index"] / total_paras
                    positions.append(pos)
                    break # just take the first match
                    
    df_pos = pd.DataFrame({"normalized_position": positions})
    df_pos.to_csv(EDA_OUT / "evidence_positions.csv", index=False)
    
    plt.figure(figsize=(10, 6))
    sns.histplot(df_pos["normalized_position"], bins=20, kde=True)
    plt.title("Evidence Position Distribution")
    plt.xlabel("Normalized Position in Document")
    plt.ylabel("Count")
    plt.savefig(FIG_OUT / "evidence_position_distribution.png")
    plt.close()
    
    # Categories
    beginning = sum(1 for p in positions if p < 0.33)
    middle = sum(1 for p in positions if 0.33 <= p < 0.66)
    end = sum(1 for p in positions if p >= 0.66)
    
    return {"Beginning": beginning, "Middle": middle, "End": end}
    
def generate_redundancy_heatmap(sim_matrix, chunk_ids, suffix=""):
    plt.figure(figsize=(8, 6))
    sns.heatmap(sim_matrix, annot=False, cmap="YlGnBu", xticklabels=chunk_ids, yticklabels=chunk_ids)
    plt.title("Chunk Redundancy Heatmap")
    plt.savefig(FIG_OUT / f"redundancy_heatmap_{suffix}.png")
    plt.close()
