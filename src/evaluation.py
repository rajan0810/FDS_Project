import re
import string
import math
from collections import Counter
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity

_ROUGE_SCORER = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction: str, ground_truth: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))

def evaluate_answer(prediction: str, ground_truths: List[str]) -> Dict[str, float]:
    if not ground_truths:
        return {"exact_match": 0.0, "f1": 0.0}
        
    em_scores = [exact_match_score(prediction, truth) for truth in ground_truths]
    f1_scores = [f1_score(prediction, truth) for truth in ground_truths]
    
    return {
        "exact_match": max(em_scores),
        "f1": max(f1_scores)
    }

def calculate_retrieval_metrics(retrieved_chunk_ids: List[str], gold_chunk_ids: List[str], k: int) -> Dict[str, float]:
    if not gold_chunk_ids:
        # If there is no gold evidence, we can't properly evaluate retrieval in QASPER
        return {"precision@k": 0.0, "recall@k": 0.0, "f1@k": 0.0, "mrr": 0.0}
        
    retrieved_k = retrieved_chunk_ids[:k]
    
    relevant_retrieved = len(set(retrieved_k).intersection(set(gold_chunk_ids)))
    
    precision = relevant_retrieved / k if k > 0 else 0.0
    recall = relevant_retrieved / len(gold_chunk_ids)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    mrr = 0.0
    for rank, chunk_id in enumerate(retrieved_k, 1):
        if chunk_id in gold_chunk_ids:
            mrr = 1.0 / rank
            break
            
    return {
        f"precision@{k}": precision,
        f"recall@{k}": recall,
        f"f1@{k}": f1,
        "mrr": mrr,
        "evidence_recall": recall # General metric
    }

def ndcg_at_k(retrieved_chunk_ids: List[str], gold_chunk_ids: List[str], k: int) -> float:
    """
    Normalized Discounted Cumulative Gain @ K, using binary relevance
    (1 if a retrieved chunk is a gold-evidence chunk, else 0).

    DCG@k  = sum_{i=1..k} rel_i / log2(i + 1)
    IDCG@k = DCG@k of the ideal ranking (all relevant chunks first)
    NDCG@k = DCG@k / IDCG@k   (0.0 if IDCG@k == 0, i.e. no gold evidence)
    """
    if not gold_chunk_ids:
        return 0.0

    gold_set = set(gold_chunk_ids)
    retrieved_k = retrieved_chunk_ids[:k]

    relevances = [1.0 if cid in gold_set else 0.0 for cid in retrieved_k]
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))

    ideal_relevances = sorted(relevances, reverse=True)
    num_ideal_relevant = min(len(gold_set), k)
    ideal_relevances = [1.0] * num_ideal_relevant + [0.0] * (k - num_ideal_relevant)
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_relevances))

    return dcg / idcg if idcg > 0 else 0.0

def calculate_rouge(prediction: str, reference: str) -> Dict[str, float]:
    """ROUGE-1 / ROUGE-2 / ROUGE-L F-measure between a generated answer and a reference answer."""
    if not prediction or not reference:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    scores = _ROUGE_SCORER.score(reference, prediction)
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
    }

def groundedness_score(answer: str, context_text: str, embedding_manager=None) -> Dict[str, float]:
    """
    Groundedness = how well the generated answer is supported by the retrieved context,
    computed without any extra LLM calls, as a hybrid of two signals:

      1. token_overlap   = fraction of the answer's normalized content tokens that also
                            appear in the context text (a lexical "is this stated in the
                            context?" check).
      2. embedding_sim    = cosine similarity between the answer's sentence embedding and
                            the context's sentence embedding (a semantic "is this consistent
                            with the context?" check), using the same EmbeddingManager as
                            the rest of the pipeline. Only computed if `embedding_manager`
                            is provided.

    combined = mean of the available components. A low score suggests the answer may not
    be well-grounded in (i.e. may hallucinate beyond) the supplied context.
    """
    if not answer or not context_text:
        return {"token_overlap": 0.0, "embedding_similarity": 0.0, "groundedness": 0.0}

    answer_tokens = normalize_answer(answer).split()
    context_normalized = normalize_answer(context_text)
    context_token_set = set(context_normalized.split())

    if not answer_tokens:
        token_overlap = 0.0
    else:
        matched = sum(1 for tok in answer_tokens if tok in context_token_set)
        token_overlap = matched / len(answer_tokens)

    components = [token_overlap]
    embedding_similarity = None

    if embedding_manager is not None:
        embs = embedding_manager.get_embeddings([answer, context_text])
        embedding_similarity = float(cosine_similarity(embs[0:1], embs[1:2])[0][0])
        components.append(embedding_similarity)

    combined = float(np.mean(components))

    return {
        "token_overlap": token_overlap,
        "embedding_similarity": embedding_similarity if embedding_similarity is not None else 0.0,
        "groundedness": combined,
    }

def compression_ratio(original_tokens: int, selected_tokens: int) -> float:
    """Percentage reduction in token count achieved by adaptive context selection."""
    if original_tokens <= 0:
        return 0.0
    return 100.0 * (1 - (selected_tokens / original_tokens))

def throughput(num_queries: int, total_time_seconds: float) -> float:
    """Queries processed per second over a batch/experiment run."""
    if total_time_seconds <= 0:
        return 0.0
    return num_queries / total_time_seconds
