import time
import random
import psutil
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any
from .config import CONFIG, BASE_DIR
from .dataset import load_qasper_dataset
from .preprocessing import flatten_qasper_paper, extract_questions
from .chunking import create_chunks
from .embeddings import EmbeddingManager, embed_chunks
from .retrieval import retrieve, map_gold_evidence_to_chunks
from .reranking import Reranker
from .adaptive_selection import select_by_relevance, remove_redundant_chunks, order_context
from .generation import generate_answer
from .evaluation import evaluate_answer, calculate_rouge, groundedness_score, compression_ratio, throughput

EXPERIMENTS_OUT = BASE_DIR / "outputs" / "experiments"
EXPERIMENTS_FIG_OUT = EXPERIMENTS_OUT / "figures"
EXPERIMENTS_OUT.mkdir(parents=True, exist_ok=True)
EXPERIMENTS_FIG_OUT.mkdir(parents=True, exist_ok=True)

_PROCESS = psutil.Process()

def _rss_mb() -> float:
    return _PROCESS.memory_info().rss / (1024 * 1024)

def _sample_questions_with_evidence(ds: List[Dict[str, Any]], num_samples: int):
    """Pick `num_samples` (paper, question) pairs that have gold evidence, deterministically
    (relies on RANDOM_SEED already having been applied to the `random` module by the caller)."""
    candidates = []
    for item in ds:
        q_records = extract_questions(item)
        for q in q_records:
            if q.evidence and q.reference_answers:
                candidates.append((item, q))

    if len(candidates) <= num_samples:
        return candidates
    return random.sample(candidates, num_samples)

def run_experiment_sweep(num_samples: int = None) -> Dict[str, pd.DataFrame]:
    """
    Runs the Context-vs-Accuracy, Latency, and Cost-Quality Tradeoff analyses.

    For each sampled question, builds a baseline context (top-K retrieved chunks, no
    adaptive filtering, K = the largest budget in CONTEXT_SIZE_SWEEP) and, for every budget
    in CONTEXT_SIZE_SWEEP, an adaptive context (retrieve -> rerank -> select_by_relevance
    (max_chunks=budget) -> remove_redundant_chunks -> order_context). Each context is sent
    to the LLM once, and answer-quality, cost, and latency metrics are recorded per run.

    This makes real calls to the configured Gemini model — cost/runtime scale with
    num_samples * (len(CONTEXT_SIZE_SWEEP) + 1).
    """
    if num_samples is None:
        num_samples = CONFIG.get("EXPERIMENT_NUM_SAMPLES", 20)

    sweep_budgets = CONFIG.get("CONTEXT_SIZE_SWEEP", [2, 4, 6, 10])
    max_budget = max(sweep_budgets)

    print(f"Loading QASPER dataset for experiment sweep ({num_samples} questions x {len(sweep_budgets)} configs)...")
    ds = load_qasper_dataset()
    sampled = _sample_questions_with_evidence(ds, num_samples)
    print(f"Sampled {len(sampled)} questions with gold evidence.")

    emb_manager = EmbeddingManager()
    reranker = Reranker()

    paper_cache: Dict[str, Dict[str, Any]] = {}

    def get_paper_context(item):
        paper_id = item["id"]
        if paper_id in paper_cache:
            return paper_cache[paper_id]

        pr = flatten_qasper_paper(item)
        chunks = create_chunks(pr.paragraphs)
        chunk_embeddings, chunk_metadata = embed_chunks(chunks, emb_manager)
        paper_cache[paper_id] = {
            "pr": pr,
            "chunk_embeddings": chunk_embeddings,
            "chunk_metadata": chunk_metadata,
        }
        return paper_cache[paper_id]

    accuracy_rows = []
    latency_rows = []

    for i, (item, q) in enumerate(sampled):
        print(f"[{i+1}/{len(sampled)}] {q.question[:70]}...")
        paper_ctx = get_paper_context(item)
        pr = paper_ctx["pr"]
        chunk_embeddings = paper_ctx["chunk_embeddings"]
        chunk_metadata = paper_ctx["chunk_metadata"]

        if len(chunk_metadata) == 0:
            continue

        query_emb = emb_manager.get_embeddings([q.question])[0]
        retrieved = retrieve(query_emb, chunk_embeddings, chunk_metadata, k=max_budget)
        gold_chunk_ids = map_gold_evidence_to_chunks(q.evidence, pr.paragraphs, chunk_metadata)
        orig_tokens = sum(p["token_count"] for p in pr.paragraphs)

        # --- Baseline: full top-K retrieved context, no adaptive filtering ---
        baseline_text = "\n\n".join(retrieved["text"].tolist())
        baseline_tokens = int(retrieved["token_count"].sum())

        mem_before = _rss_mb()
        base_start = time.perf_counter()
        base_result = generate_answer(q.question, baseline_text)
        base_latency = time.perf_counter() - base_start
        mem_after = _rss_mb()

        base_eval = evaluate_answer(base_result["answer"], q.reference_answers)
        base_rouge = max(
            (calculate_rouge(base_result["answer"], ref) for ref in q.reference_answers),
            key=lambda r: r["rougeL"],
            default={"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0},
        )
        base_grounded = groundedness_score(base_result["answer"], baseline_text, emb_manager)

        accuracy_rows.append({
            "question_id": q.question_id, "config": "baseline", "max_chunks": max_budget,
            "num_selected_chunks": len(retrieved), "context_tokens": baseline_tokens,
            "f1": base_eval["f1"], "exact_match": base_eval["exact_match"],
            "rouge1": base_rouge["rouge1"], "rouge2": base_rouge["rouge2"], "rougeL": base_rouge["rougeL"],
            "groundedness": base_grounded["groundedness"],
        })
        latency_rows.append({
            "question_id": q.question_id, "config": "baseline", "max_chunks": max_budget,
            "context_tokens": baseline_tokens, "latency_seconds": base_latency,
            "memory_delta_mb": mem_after - mem_before,
            "prompt_tokens": base_result["prompt_tokens"], "completion_tokens": base_result["completion_tokens"],
            "total_tokens": base_result["total_tokens"],
        })

        # --- Adaptive: sweep over context-size budgets ---
        reranked = reranker.rerank(q.question, retrieved)

        for budget in sweep_budgets:
            relevance_selected = select_by_relevance(reranked, max_chunks=budget)
            deduplicated, _removed = remove_redundant_chunks(relevance_selected, chunk_embeddings)
            ordered = order_context(deduplicated)

            adaptive_text = "\n\n".join(ordered["text"].tolist()) if len(ordered) > 0 else ""
            adaptive_tokens = int(ordered["token_count"].sum()) if len(ordered) > 0 else 0

            mem_before = _rss_mb()
            adapt_start = time.perf_counter()
            adapt_result = generate_answer(q.question, adaptive_text)
            adapt_latency = time.perf_counter() - adapt_start
            mem_after = _rss_mb()

            adapt_eval = evaluate_answer(adapt_result["answer"], q.reference_answers)
            adapt_rouge = max(
                (calculate_rouge(adapt_result["answer"], ref) for ref in q.reference_answers),
                key=lambda r: r["rougeL"],
                default={"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0},
            )
            adapt_grounded = groundedness_score(adapt_result["answer"], adaptive_text, emb_manager)

            accuracy_rows.append({
                "question_id": q.question_id, "config": f"adaptive_k{budget}", "max_chunks": budget,
                "num_selected_chunks": len(ordered), "context_tokens": adaptive_tokens,
                "f1": adapt_eval["f1"], "exact_match": adapt_eval["exact_match"],
                "rouge1": adapt_rouge["rouge1"], "rouge2": adapt_rouge["rouge2"], "rougeL": adapt_rouge["rougeL"],
                "groundedness": adapt_grounded["groundedness"],
            })
            latency_rows.append({
                "question_id": q.question_id, "config": f"adaptive_k{budget}", "max_chunks": budget,
                "context_tokens": adaptive_tokens, "latency_seconds": adapt_latency,
                "memory_delta_mb": mem_after - mem_before,
                "prompt_tokens": adapt_result["prompt_tokens"], "completion_tokens": adapt_result["completion_tokens"],
                "total_tokens": adapt_result["total_tokens"],
            })

    accuracy_df = pd.DataFrame(accuracy_rows)
    latency_df = pd.DataFrame(latency_rows)

    accuracy_df.to_csv(EXPERIMENTS_OUT / "context_vs_accuracy.csv", index=False)
    latency_df.to_csv(EXPERIMENTS_OUT / "latency_vs_context.csv", index=False)

    # --- Cost-Quality Tradeoff: aggregate per configuration ---
    merged = accuracy_df.merge(
        latency_df[["question_id", "config", "latency_seconds", "memory_delta_mb", "total_tokens"]],
        on=["question_id", "config"],
    )
    tradeoff_df = merged.groupby("config").agg(
        max_chunks=("max_chunks", "first"),
        avg_f1=("f1", "mean"),
        avg_exact_match=("exact_match", "mean"),
        avg_rougeL=("rougeL", "mean"),
        avg_groundedness=("groundedness", "mean"),
        avg_context_tokens=("context_tokens", "mean"),
        avg_total_tokens=("total_tokens", "mean"),
        avg_latency_seconds=("latency_seconds", "mean"),
        n_questions=("question_id", "count"),
    ).reset_index()

    total_time_per_config = merged.groupby("config")["latency_seconds"].sum()
    tradeoff_df["throughput_qps"] = tradeoff_df.apply(
        lambda r: throughput(r["n_questions"], total_time_per_config[r["config"]]), axis=1
    )
    tradeoff_df.to_csv(EXPERIMENTS_OUT / "cost_quality_tradeoff.csv", index=False)

    _plot_context_vs_accuracy(tradeoff_df)
    _plot_latency_vs_context(tradeoff_df)
    _plot_cost_quality_tradeoff(tradeoff_df)

    return {
        "context_vs_accuracy": accuracy_df,
        "latency_vs_context": latency_df,
        "cost_quality_tradeoff": tradeoff_df,
    }

def _plot_context_vs_accuracy(tradeoff_df: pd.DataFrame):
    plt.figure(figsize=(10, 6))
    ordered = tradeoff_df.sort_values("max_chunks")
    sns.lineplot(data=ordered, x="max_chunks", y="avg_f1", marker="o")
    plt.title("Context Size vs. Answer Quality (F1)")
    plt.xlabel("Max Chunks in Context (adaptive budget; baseline = largest budget, no filtering)")
    plt.ylabel("Average F1 Score")
    plt.savefig(EXPERIMENTS_FIG_OUT / "context_vs_accuracy.png")
    plt.close()

def _plot_latency_vs_context(tradeoff_df: pd.DataFrame):
    plt.figure(figsize=(10, 6))
    ordered = tradeoff_df.sort_values("avg_context_tokens")
    sns.lineplot(data=ordered, x="avg_context_tokens", y="avg_latency_seconds", marker="o")
    plt.title("Context Length vs. Inference Latency")
    plt.xlabel("Average Context Tokens")
    plt.ylabel("Average Latency (seconds)")
    plt.savefig(EXPERIMENTS_FIG_OUT / "latency_vs_context.png")
    plt.close()

def _plot_cost_quality_tradeoff(tradeoff_df: pd.DataFrame):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=tradeoff_df, x="avg_total_tokens", y="avg_f1", hue="config", s=120)
    plt.title("Cost-Quality Tradeoff (Token Usage vs. Answer Quality)")
    plt.xlabel("Average Total Tokens (prompt + completion)")
    plt.ylabel("Average F1 Score")
    plt.savefig(EXPERIMENTS_FIG_OUT / "cost_quality_tradeoff.png")
    plt.close()
