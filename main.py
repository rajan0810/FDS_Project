import sys
import argparse
import time
import random
import numpy as np
import pandas as pd
from src.config import CONFIG, get_gemini_api_key
from src.dataset import load_qasper_dataset
from src.preprocessing import flatten_qasper_paper, extract_questions
from src.chunking import create_chunks
from src.embeddings import EmbeddingManager, embed_chunks, save_embeddings, load_embeddings
from src.retrieval import retrieve, map_gold_evidence_to_chunks
from src.redundancy import calculate_pairwise_redundancy
from src.reranking import Reranker
from src.adaptive_selection import adaptive_select
from src.generation import generate_answer
from src.evaluation import evaluate_answer, calculate_retrieval_metrics
from src.eda import (
    analyze_context_lengths, analyze_evidence_positions,
    analyze_relevance, analyze_redundancy, analyze_cross_document_similarity,
)
from src.experiments import run_experiment_sweep

random.seed(CONFIG["RANDOM_SEED"])
np.random.seed(CONFIG["RANDOM_SEED"])

def print_separator(title=None):
    print("=" * 50)
    if title:
        print(title)
        print("=" * 50)

def cmd_setup():
    print_separator("SYSTEM CHECK")
    print("Python: OK")
    print("Packages: OK")
    
    if get_gemini_api_key():
        print("Gemini API key: FOUND")
    else:
        print("Gemini API key: MISSING (Please check .env)")
        
    print("Loading QASPER (this may take a moment)...")
    try:
        ds = load_qasper_dataset()
        print(f"QASPER: OK (Validation set size: {len(ds)})")
    except Exception as e:
        print(f"QASPER: ERROR ({e})")
        
    print("Loading Embedding Model...")
    try:
        manager = EmbeddingManager()
        print("Embedding model: OK")
    except Exception as e:
        print(f"Embedding model: ERROR ({e})")

    print("\nReady to run.")

def cmd_inspect():
    ds = load_qasper_dataset()
    sample = ds[0]
    
    paper_record = flatten_qasper_paper(sample)
    question_records = extract_questions(sample)
    
    print_separator("DATASET INSPECTION")
    print(f"Number of papers in valid split: {len(ds)}")
    
    print_separator("PAPER")
    print(f"Title: {paper_record.title}")
    print(f"Paragraphs: {len(paper_record.paragraphs)}")
    
    if question_records:
        q = question_records[0]
        print_separator("QUESTION")
        print(q.question)
        print_separator("REFERENCE ANSWER")
        for ans in q.reference_answers:
            print(ans)
        print_separator("GOLD EVIDENCE")
        for ev in q.evidence:
            print(ev)

def cmd_eda():
    print_separator("EDA RUN")
    ds = load_qasper_dataset()

    max_papers = CONFIG["MAX_PAPERS_DEMO"]
    paper_records = []
    question_records = []
    chunk_records = []

    print(f"Processing {max_papers} papers for EDA...")
    for i in range(min(max_papers, len(ds))):
        sample = ds[i]
        pr = flatten_qasper_paper(sample)
        qr = extract_questions(sample)
        cr = create_chunks(pr.paragraphs)

        paper_records.append(pr)
        question_records.extend(qr)
        chunk_records.extend(cr)

    print("⸻\n1. Context Length Analysis\n⸻")
    print("Analyze document and token-length distributions...")
    stats = analyze_context_lengths(paper_records, question_records, chunk_records)
    print(stats)

    # Embed chunks once per paper so Relevance, Redundancy, and Cross-Document
    # Similarity analyses can all reuse the same vectors.
    emb_manager = EmbeddingManager()
    paper_embeddings = {}
    paper_chunk_data = {}
    for pr in paper_records:
        chunks = create_chunks(pr.paragraphs)
        chunk_embeddings, chunk_metadata = embed_chunks(chunks, emb_manager)
        paper_embeddings[pr.paper_id] = chunk_embeddings
        paper_chunk_data[pr.paper_id] = chunk_metadata

    print("⸻\n2. Relevance Analysis\n⸻")
    print("Usefulness of available context for each query...")
    relevance_records = []
    redundancy_records = []
    sample_sim_matrix, sample_chunk_ids = None, None

    for pr in paper_records:
        chunk_embeddings = paper_embeddings[pr.paper_id]
        chunk_metadata = paper_chunk_data[pr.paper_id]
        if len(chunk_metadata) == 0:
            continue

        paper_questions = [q for q in question_records if q.paper_id == pr.paper_id]
        for q in paper_questions:
            query_emb = emb_manager.get_embeddings([q.question])[0]
            retrieved = retrieve(query_emb, chunk_embeddings, chunk_metadata, k=CONFIG["MAX_CHUNKS"])
            gold_chunk_ids = map_gold_evidence_to_chunks(q.evidence, pr.paragraphs, chunk_metadata)

            relevance_records.append({
                "question_id": q.question_id,
                "avg_topk_similarity": float(retrieved["similarity_score"].mean()) if len(retrieved) else 0.0,
                "max_similarity": float(retrieved["similarity_score"].max()) if len(retrieved) else 0.0,
                "evidence_covered": bool(set(retrieved["chunk_id"]) & set(gold_chunk_ids)) if gold_chunk_ids else False,
            })

            retrieved_embeddings = chunk_embeddings[retrieved.index.tolist()]
            retrieved_ids = retrieved["chunk_id"].tolist()
            pairs = calculate_pairwise_redundancy(retrieved_embeddings, retrieved_ids)
            num_redundant = sum(1 for p in pairs if p[3])
            redundancy_records.append({
                "question_id": q.question_id,
                "num_pairs": len(pairs),
                "num_redundant": num_redundant,
                "redundancy_rate": (num_redundant / len(pairs)) if pairs else 0.0,
            })

            if sample_sim_matrix is None and len(retrieved_embeddings) >= 2:
                from sklearn.metrics.pairwise import cosine_similarity
                sample_sim_matrix = cosine_similarity(retrieved_embeddings)
                sample_chunk_ids = retrieved_ids

    relevance_df = analyze_relevance(relevance_records)
    if len(relevance_df):
        coverage_pct = 100 * relevance_df["evidence_covered"].mean()
        print(f"Questions analyzed: {len(relevance_df)}")
        print(f"Average top-K retrieval similarity: {relevance_df['avg_topk_similarity'].mean():.3f}")
        print(f"Gold-evidence retrieval coverage: {coverage_pct:.1f}%")

    print("⸻\n3. Evidence Position Analysis\n⸻")
    print("Study whether relevant information occurs at beginning, middle, or end of long contexts...")
    pos_stats = analyze_evidence_positions(paper_records, question_records, chunk_records)
    for pos, count in pos_stats.items():
        print(f" - {pos}: {count} occurrences")

    print("⸻\n4. Redundancy Analysis\n⸻")
    print("Identify duplicate or semantically similar information...")
    redundancy_df = analyze_redundancy(redundancy_records, sample_sim_matrix, sample_chunk_ids)
    if len(redundancy_df):
        print(f"Average redundancy rate among retrieved chunks: {redundancy_df['redundancy_rate'].mean()*100:.1f}%")

    print("⸻\n5. Cross-Document Similarity Analysis\n⸻")
    print("Measure context similarity between different papers in the corpus...")
    paper_ids = [pid for pid, emb in paper_embeddings.items() if len(emb) > 0]
    paper_vectors = np.array([paper_embeddings[pid].mean(axis=0) for pid in paper_ids])
    cross_doc_df = analyze_cross_document_similarity(paper_ids, paper_vectors)
    if len(cross_doc_df):
        print(f"Paper pairs compared: {len(cross_doc_df)}")
        print(f"Average cross-document similarity: {cross_doc_df['similarity'].mean():.3f}")

    print("EDA plots saved to outputs/eda/figures/")

def cmd_experiment(num_samples=None):
    print_separator("EXPERIMENT SWEEP: Context-vs-Accuracy, Latency, Cost-Quality Tradeoff")
    results = run_experiment_sweep(num_samples)
    print("\nCost-Quality Tradeoff Summary (per configuration):")
    print(results["cost_quality_tradeoff"].to_string(index=False))
    print("\nArtifacts saved to outputs/experiments/")

def cmd_demo():
    print_separator("ADAPTIVE CONTEXT SELECTION DEMO")
    
    ds = load_qasper_dataset()
    # Find a sample with questions and evidence
    sample = None
    pr = None
    qr = None
    for item in ds:
        q_records = extract_questions(item)
        has_evidence = False
        for q in q_records:
            if q.evidence:
                has_evidence = True
                qr = q
                break
        if has_evidence:
            sample = item
            pr = flatten_qasper_paper(sample)
            break
            
    if not sample:
        print("No suitable question found.")
        return
        
    print(f"Question:\n{qr.question}\n")
    print("⸻")
    
    # 1. Chunking
    chunks = create_chunks(pr.paragraphs)
    chunk_df = pd.DataFrame(chunks)
    print(f"Paper parsed into {len(pr.paragraphs)} paragraphs and {len(chunks)} chunks.")
    
    # 2. Vector Embeddings
    print("⸻")
    print("STEP 0 — VECTOR EMBEDDINGS")
    print(f"Model: {CONFIG['EMBEDDING_MODEL']}")
    print(f"Converting {len(chunks)} text chunks into 384-dimensional dense vectors using Hugging Face sentence-transformers...")
    
    emb_manager = EmbeddingManager()
    chunk_embeddings, chunk_metadata = embed_chunks(chunks, emb_manager)
    
    # 3. Retrieval
    print("⸻")
    print("STEP 1 — VECTOR SEARCH & RETRIEVAL")
    print("Converting the question into a 384-dimensional vector...")
    query_emb = emb_manager.get_embeddings([qr.question])[0]
    print(f"Calculating Cosine Similarity between the Question vector and all {len(chunks)} Chunk vectors...")
    print(f"Retrieving Top {CONFIG['MAX_CHUNKS']} most similar chunks:")
    
    retrieved = retrieve(query_emb, chunk_embeddings, chunk_metadata, k=CONFIG["MAX_CHUNKS"])
    gold_chunk_ids = map_gold_evidence_to_chunks(qr.evidence, pr.paragraphs, chunk_metadata)

    print(f"Top {CONFIG['MAX_CHUNKS']} candidates:")
    for i, row in retrieved.iterrows():
        is_relevant = "[RELEVANT]" if row["chunk_id"] in gold_chunk_ids else "[NOT RELEVANT]"
        print(f"{row['rank']}. {row['chunk_id']} | score={row['similarity_score']:.3f} | {is_relevant}")

    # 2b. Context Reranking
    print("⸻")
    print("STEP 2 — CONTEXT RERANKING")
    print(f"Cross-encoder model: {CONFIG['RERANK_MODEL']}")
    reranker = Reranker()
    retrieved = reranker.rerank(qr.question, retrieved)
    print(f"Re-scored and re-sorted {len(retrieved)} candidates by cross-encoder relevance.")

    # 4. Redundancy Analysis
    print("⸻\n4. Redundancy Analysis\n⸻")
    print("Identify duplicate or semantically similar information...")
    
    # Extract embeddings for the retrieved chunks using their original indices
    retrieved_embeddings = chunk_embeddings[retrieved.index.tolist()]
    retrieved_ids = retrieved["chunk_id"].tolist()
    
    redundancy_pairs = calculate_pairwise_redundancy(retrieved_embeddings, retrieved_ids)
    
    redundant_only = [p for p in redundancy_pairs if p[3]]
    non_redundant = [p for p in redundancy_pairs if not p[3]]
    
    # Print a few non-redundant examples
    for id1, id2, score, _ in non_redundant[:5]:
        print(f"{id1} <-> {id2} = {score:.2f} | No redundancy")
        
    if len(non_redundant) > 5:
        print(f"... and {len(non_redundant) - 5} more non-redundant pairs")
        
    # Always print the redundant pairs
    for id1, id2, score, _ in redundant_only:
        print(f"{id1} <-> {id2} = {score:.2f} | Potential redundancy: YES")

    # 5. Adaptive Context Selection
    print("⸻")
    print("STEP 3 — ADAPTIVE SELECTION")
    start_time = time.perf_counter()
    selected_context_df, removed = adaptive_select(retrieved, chunk_embeddings, chunk_metadata)
    selection_time = time.perf_counter() - start_time
    
    print(f"Candidates considered: {len(retrieved)}")
    print(f"Selected: {len(selected_context_df)}")
    print(f"Removed as redundant: {removed}")
    
    # 6. Context Compression
    print("⸻")
    print("STEP 4 — CONTEXT COMPRESSION")
    orig_tokens = sum(p["token_count"] for p in pr.paragraphs)
    retrieved_tokens = retrieved["token_count"].sum()
    selected_tokens = selected_context_df["token_count"].sum()
    compression = 100 * (1 - (selected_tokens / orig_tokens))
    
    print(f"Original paper:       {orig_tokens} tokens")
    print(f"Retrieved context:    {retrieved_tokens} tokens")
    print(f"Final context:        {selected_tokens} tokens")
    print(f"Compression:          {compression:.2f}%")
    
    # 6. Baseline vs Adaptive Execution for Accuracy and Latency Analysis
    print("⸻\n5. Context vs Accuracy Analysis\n⸻")
    print("Measure how different context sizes affect answer quality...")
    
    print("\n--- BASELINE LLM (Full Top-K Context) ---")
    baseline_context = "\n\n".join(retrieved["text"].tolist())
    base_start = time.perf_counter()
    base_result = generate_answer(qr.question, baseline_context)
    base_latency = time.perf_counter() - base_start
    base_f1 = evaluate_answer(base_result["answer"], qr.reference_answers)["f1"]
    
    print(f"Baseline F1 Score: {base_f1:.3f}")
    
    print("\n--- ADAPTIVE LLM (Reduced Context) ---")
    adaptive_context = "\n\n".join(selected_context_df["text"].tolist())
    adapt_start = time.perf_counter()
    adapt_result = generate_answer(qr.question, adaptive_context)
    adapt_latency = time.perf_counter() - adapt_start
    adapt_f1 = evaluate_answer(adapt_result["answer"], qr.reference_answers)["f1"]
    
    print(f"Adaptive F1 Score: {adapt_f1:.3f}")
    
    # Check Accuracy change
    if adapt_f1 >= base_f1:
        print(f"-> Accuracy Maintained/Improved! (Adaptive F1 >= Baseline F1)")
    else:
        print(f"-> Accuracy Decreased (Delta: {adapt_f1 - base_f1:.3f})")
    
    print("⸻\n6. Latency Analysis\n⸻")
    print("Increasing context length vs Inference Time and Resource Usage...")
    print(f"Baseline Context Tokens: {retrieved_tokens}")
    print(f"Adaptive Context Tokens: {selected_tokens} (-{compression:.2f}% resource usage)")
    print(f"Baseline Inference Time: {base_latency:.2f} seconds")
    print(f"Adaptive Inference Time: {adapt_latency:.2f} seconds")
    
    speedup = base_latency - adapt_latency
    print(f"-> Time Saved: {speedup:.2f} seconds per query")
    
    print("\n==================================================")
    print("FINAL RESULT")
    print("==================================================")
    print(f"Final Adaptive Answer:\n{adapt_result['answer']}")

def cmd_interactive():
    print_separator("INTERACTIVE ADAPTIVE CONTEXT DEMO")
    
    ds = load_qasper_dataset()
    sample = random.choice(ds)
    pr = flatten_qasper_paper(sample)
    
    print(f"Paper Title: {sample['title']}")
    print(f"Abstract: {sample['abstract']}\n")
    print(f"Loaded {len(pr.paragraphs)} paragraphs from the paper.")
    
    # Print some example questions so the user doesn't have to invent one
    print("\n💡 Example Questions you could ask:")
    sample_questions = extract_questions(sample)
    for i, q in enumerate(sample_questions[:3]):
        print(f" {i+1}. {q.question}")
    print()
    
    question = input("Ask a question about this paper: ")
    
    print("\n⸻\nSTEP 0 — VECTOR EMBEDDINGS\n⸻")
    chunks = create_chunks(pr.paragraphs)
    print(f"Converting {len(chunks)} text chunks into vectors...")
    
    emb_manager = EmbeddingManager()
    chunk_embeddings, chunk_metadata = embed_chunks(chunks, emb_manager)
    
    print("⸻\nSTEP 1 — VECTOR SEARCH & RETRIEVAL\n⸻")
    query_emb = emb_manager.get_embeddings([question])[0]
    retrieved = retrieve(query_emb, chunk_embeddings, chunk_metadata, k=CONFIG["MAX_CHUNKS"])
    
    print(f"Top {CONFIG['MAX_CHUNKS']} candidates retrieved:")
    for i, row in retrieved.iterrows():
        print(f"{row['rank']}. {row['chunk_id']} | score={row['similarity_score']:.3f}")

    print("⸻\nSTEP 1b — CONTEXT RERANKING\n⸻")
    reranker = Reranker()
    retrieved = reranker.rerank(question, retrieved)
    print(f"Re-scored and re-sorted {len(retrieved)} candidates by cross-encoder relevance.")

    print("⸻\nSTEP 2 — REDUNDANCY\n⸻")
    retrieved_embeddings = chunk_embeddings[retrieved.index.tolist()]
    retrieved_ids = retrieved["chunk_id"].tolist()
    redundancy_pairs = calculate_pairwise_redundancy(retrieved_embeddings, retrieved_ids)
    
    redundant_only = [p for p in redundancy_pairs if p[3]]
    non_redundant = [p for p in redundancy_pairs if not p[3]]
    
    # Print a few non-redundant examples
    for id1, id2, score, _ in non_redundant[:5]:
        print(f"{id1} <-> {id2} = {score:.2f} | No redundancy")
        
    if len(non_redundant) > 5:
        print(f"... and {len(non_redundant) - 5} more non-redundant pairs")
        
    # Always print the redundant pairs
    for id1, id2, score, _ in redundant_only:
        print(f"{id1} <-> {id2} = {score:.2f} | Potential redundancy: YES")
        
    if not redundant_only:
        print("-> No chunks breached the redundancy threshold.")
        
    print("⸻\nSTEP 3 — ADAPTIVE SELECTION\n⸻")
    selected_context_df, removed = adaptive_select(retrieved, chunk_embeddings, chunk_metadata)
    print(f"Candidates considered: {len(retrieved)}")
    print(f"Selected: {len(selected_context_df)}")
    print(f"Removed as redundant: {removed}")
    
    print("⸻\nSTEP 4 — GEMINI ANSWER\n⸻")
    context_text = "\n\n".join(selected_context_df["text"].tolist())
    gen_result = generate_answer(question, context_text)
    
    print(f"\nFinal Adaptive Answer:\n{gen_result['answer']}\n")
    print(f"Latency: {gen_result['latency']:.2f} seconds")

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py [setup|inspect|eda|demo|interactive|experiment] [--num-samples N]")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "setup":
        cmd_setup()
    elif cmd == "inspect":
        cmd_inspect()
    elif cmd == "eda":
        cmd_eda()
    elif cmd == "demo":
        cmd_demo()
    elif cmd == "interactive":
        cmd_interactive()
    elif cmd == "experiment":
        parser = argparse.ArgumentParser()
        parser.add_argument("--num-samples", type=int, default=None)
        args = parser.parse_args(sys.argv[2:])
        cmd_experiment(args.num_samples)
    else:
        print(f"Unknown command: {cmd}")

if __name__ == "__main__":
    main()
