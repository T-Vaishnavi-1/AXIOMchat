# AXIOMChat — Conversational Context Pruning Engine

## Problem
Enterprise LLM sessions accumulate stale context over time — dead reasoning chains, 
superseded tool outputs, off-topic exchanges. Sending bloated history on every API 
call increases cost and degrades response quality.

## Approach
Model conversation history as a Directed Acyclic Graph (DAG), score edges and nodes 
for semantic relevance, and prune nodes that are structurally safe and semantically 
redundant.

## Architecture
- **Graph Engine** — DAG with semantic edge weights (topic overlap), bottom-up score 
  propagation, and a two-layer deletion gate (structural + semantic)
- **Datahub** — curated topic taxonomy with keyword signatures for lexical relevance scoring
- **Embedding Layer** — sentence-transformers (all-MiniLM-L6-v2) with cosine similarity 
  against a session anchor vector

## Key Design Decisions
- A node is only deletable if its out-degree is 0 AND all parent edges have zero semantic weight
- Branching nodes score themselves independently — child relevance doesn't propagate upward across branches
- Session anchor = mean embedding of non-noise nodes, robust to greetings and filler

## Stack
`networkx` · `sentence-transformers` · `scikit-learn` · `numpy` · `pandas` · `matplotlib`

## Usage
```bash
poetry install
poetry run python src/graph_engine.py
```

## Notebook
Open `notebooks/context_pruning_study.ipynb` for full pipeline walkthrough with 
score distribution analysis and DAG visualization.

## Roadmap
- [ ] LLM-based scorer (V1) via Anthropic API
- [ ] Recency decay factor for stale context detection
- [ ] GNN-based score aggregation (V2)
- [ ] Enterprise API middleware layer
