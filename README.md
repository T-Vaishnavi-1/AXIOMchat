# AXIOMchat — Research: Fixed-Anchor Topic-Drift Detection

This is a research project, not a shipped tool. It exists to answer a
specific question: can you detect when an LLM chat session has drifted from
its original intent, cheaply enough to run on every message, without the
correctness risk of actually deleting anything? See `FINDINGS.md` for what's
actually been tested and learned so far.

## Problem

Long LLM chat sessions accumulate tangents — the topic drifts, sometimes
usefully, sometimes not. The original version of this project tried to solve
that by pruning: modeling history as a DAG and deleting nodes judged
irrelevant. That design had real correctness bugs (see `FINDINGS.md`) and was
abandoned in favor of a lower-stakes goal: flag drift, never delete.

## Approach

1. **Multi-facet anchor extraction** — once `k` user messages have
   accumulated, strip structured-inventory lines (e.g. a resume's skills
   table — see `FINDINGS.md` #13), then run a two-stage LLM pipeline:
   summarize what's been discussed (detail-preserving, not compressed to one
   abstract goal), then extract a short **list** of distinct topic/task
   facets — not one blended sentence. A single averaged anchor can't
   represent a session that covers several real facets (`FINDINGS.md` #11);
   this keeps several, frozen once computed and never re-derived, to avoid a
   circular dependency between "what's relevant" and "what's the summary."
2. **Scoring cascade** — every message is scored cheap-to-expensive: a noise
   filter, then cosine similarity against *every* anchor facet, keeping the
   best match (banding into `unrelated` / `ambiguous` / `relevant`), with an
   LLM call — considering all facets — reserved only for the `ambiguous`
   band.
3. **The library never deletes.** Every path returns a score, band, and the
   best-matching facet to the caller — a human or another system decides
   what, if anything, to do with a low-relevance message.

## Architecture

| Module | Responsibility |
|---|---|
| `src/datahub.py` | Noise filtering (`is_noise`), a keyword topic taxonomy, and `strip_structured_lists` (drops resume-table-style lines before summarization) |
| `src/embedder.py` | `Embedder` (sentence-transformers) and `MockEmbedder`, plus shared cosine similarity and a provider factory (`AXIOM_EMBEDDER=mock\|real`) |
| `src/llm_client.py` | `LLMClient` interface — `OllamaClient` (local, default), `AnthropicClient` (hosted), `MockLLMClient`, and a provider factory |
| `src/anchor.py` | `AnchorExtractor` — the two-stage summarize → extract-facets pipeline, returns a list of facet anchors |
| `src/relevance.py` | `DriftScorer` — the scoring cascade, best-match-across-facets |
| `src/store.py` | SQLite persistence — `sessions`, `session_anchors` (one row per facet), `messages`, FTS5 (populated, not yet consumed by any scoring logic) |
| `src/api.py` | FastAPI layer exposing `/sessions` and `/sessions/{id}/messages` |

## Key design decisions

- **The anchor is fixed once computed, never continuously re-derived.** A
  moving anchor would "chase" the conversation and absorb drift into its own
  baseline, defeating the point of having a fixed reference to measure
  against.
- **Multiple facet anchors, not one.** A one-shot compression to "a single
  concrete goal" produces anchors too abstract to be useful (e.g. "get a job
  at a company"). A single distilled anchor is better but still structurally
  can't represent a session with several real facets — real testing showed
  86% of a real conversation's messages misclassified as `unrelated` even
  with a good single anchor. Splitting into multiple facets, scored by best
  match, measurably fixed the specific failures (`FINDINGS.md` #11, #12).
- **Prompt-level constraints (a target facet count) proved unreliable** on a
  local 8B model — capping or ranging the count either lost specificity or
  got ignored outright. The fix that actually worked was cleaning the input
  (stripping structured lists) rather than further prompt tuning
  (`FINDINGS.md` #13).
- **`k` is a trigger, not the anchor's content.** It controls how long to wait
  before computing the anchor, not how many raw messages get stuffed into a
  prompt — deliberately kept tunable so anchor quality vs. `k` could actually
  be measured, not just assumed.

## Stack

Python · sentence-transformers · Ollama · Anthropic SDK · NumPy · SQLite (FTS5) · FastAPI

## Usage

```bash
poetry install
poetry run python -m src.relevance   # mock smoke test, no API key or Ollama required
```

The API server (`uvicorn src.api:app --reload`) and the real embedder/LLM
paths exist in code — see `FINDINGS.md` for which of them have actually been
run and verified versus which are still untested.

## Open items

- [ ] Wire `messages_fts` (FTS5) into the scoring cascade as a secondary cheap signal — currently populated, never queried
- [ ] Fix LLM disambiguation's unreliable "reason" field (see `FINDINGS.md` #6)
- [ ] Empirically tune the `low`/`high` cosine thresholds against a labeled eval set — currently untuned placeholders
- [ ] Verify `AnthropicClient` end-to-end (needs an API key, not currently configured)
