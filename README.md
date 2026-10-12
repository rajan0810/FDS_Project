# Adaptive Context Selection for Long-Context Language Models

## Project Objective
This project investigates whether an LLM can answer questions about long documents accurately while receiving significantly less context than the entire document. It explores adaptive context selection strategies to minimize token usage and latency without sacrificing answer quality.

## Research Question
“Can we reduce the amount of context given to an LLM while maintaining or improving answer quality?”

## Dataset
Uses the **QASPER** dataset (`allenai/qasper`), which contains scientific research papers, questions, and reference answers with annotated gold evidence.

## Installation

```bash
# Optional: Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

## Gemini API Setup

Create a `.env` file in the root directory and add your API key:
```
GEMINI_API_KEY="your_api_key_here"
GEMINI_MODEL="gemini-3.1-pro"
```

## Commands

### 1. Setup Check
Run this to verify dependencies, dataset loading, and API configuration.
```bash
python main.py setup
```

### 2. Inspect Dataset
View sample structures from QASPER.
```bash
python main.py inspect
```

### 3. Exploratory Data Analysis (EDA)
Runs the full EDA suite — Context Length, Relevance, Evidence Position, Redundancy, and Cross-Document Similarity analysis — and saves CSVs/plots to `outputs/eda/`.
```bash
python main.py eda
```

### 4. Pipeline Demo
Runs a complete end-to-end adaptive context selection pipeline (retrieve -> rerank -> select -> deduplicate -> order -> generate) for a single question, comparing baseline vs. adaptive context.
```bash
python main.py demo
```

### 5. Interactive Demo
Ask your own question about a randomly sampled paper and see each pipeline stage's output.
```bash
python main.py interactive
```

### 6. Experiment Sweep
Runs the Context-vs-Accuracy, Latency, and Cost-Quality Tradeoff analyses across multiple context-size budgets on a sample of questions (makes real Gemini API calls). Results are saved to `outputs/experiments/`.
```bash
python main.py experiment --num-samples 20
```

## Project Structure
- `src/`: Core logic (preprocessing, chunking, embeddings, retrieval, reranking, adaptive selection, redundancy removal, generation, evaluation, EDA, experiment sweeps).
- `data/`: Caches and embeddings.
- `outputs/`: Saved EDA plots/CSVs and experiment sweep results.

## Limitations
- **Prototype Status**: The adaptive selection heuristic is a prototype and not a learned policy.
- **Evaluation**: The Groundedness metric is a lexical-overlap + embedding-similarity heuristic, not an LLM-as-a-judge; token counts use a word-count heuristic rather than the model's real tokenizer.
