import re
import string
from collections import Counter
from typing import List, Dict, Any
import pandas as pd

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
