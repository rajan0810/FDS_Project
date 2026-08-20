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
Runs length and position analysis on the dataset and saves plots to `outputs/eda/figures/`.
```bash
python main.py eda
```

### 4. Interactive Demo
Runs a complete end-to-end adaptive context selection pipeline for a single question.
```bash
python main.py demo
```

## Project Structure
- `src/`: Core logic (preprocessing, chunking, embeddings, retrieval, selection, generation).
- `data/`: Caches and embeddings.
- `outputs/`: Saved EDA plots, retrieval results, and experimental data.

## Limitations
- **Prototype Status**: The adaptive selection heuristic is a prototype and not a learned policy.
- **Evaluation**: The grounding metric uses exact string matching and overlap. More advanced LLM-as-a-judge could be added later.
