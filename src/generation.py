import time
from typing import Dict, Any, List
from google import genai
from google.genai import types
from .config import get_gemini_api_key, get_gemini_model

def generate_answer(query: str, selected_context: str, model_name: str = None) -> Dict[str, Any]:
    api_key = get_gemini_api_key()
    if not api_key:
        return {
            "answer": "Error: GEMINI_API_KEY not found.",
            "latency": 0.0,
            "model": "none"
        }
        
    if model_name is None:
        model_name = get_gemini_model()
        
    client = genai.Client(api_key=api_key)
    
    prompt = f"""You are answering a question using only the supplied evidence.

Question:
{query}

Context:
{selected_context}

Instructions:
1. Answer the question directly.
2. Use only information supported by the supplied context.
3. Do not invent facts.
4. If the context does not contain enough information, explicitly say so.
5. Keep the answer concise but complete."""

    start_time = time.perf_counter()
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
        )
        answer = response.text
        usage = getattr(response, "usage_metadata", None)
        if usage is not None:
            prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
            completion_tokens = getattr(usage, "candidates_token_count", 0) or 0
            total_tokens = getattr(usage, "total_token_count", 0) or (prompt_tokens + completion_tokens)
    except Exception as e:
        answer = f"Error generating answer: {str(e)}"

    latency = time.perf_counter() - start_time

    return {
        "answer": answer,
        "latency": latency,
        "model": model_name,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }
