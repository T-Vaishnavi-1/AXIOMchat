from dataclasses import dataclass

import numpy as np


@dataclass
class RelevanceResult:
    is_noise: bool
    cosine_score: float | None
    band: str  # "noise" | "unrelated" | "ambiguous" | "relevant"
    matched_facet: str | None = None
    embedding: np.ndarray | None = None
    llm_verdict: str | None = None
    llm_reason: str | None = None


class DriftScorer:
    """
    Scores messages against a set of session anchor facets via a
    cheap-to-expensive cascade: noise filter -> cosine similarity banding
    (best-matching facet, not a single averaged point) -> LLM disambiguation
    (only for the ambiguous band).
    """

    def __init__(self, embedder, datahub, llm_client, anchor_texts: list[str],
                 low: float = 0.3, high: float = 0.6):
        self.embedder = embedder
        self.datahub = datahub
        self.llm_client = llm_client
        self.anchor_texts = anchor_texts
        self.anchor_embeddings = embedder.embed(anchor_texts)
        self.low = low
        self.high = high

    def score_message(self, text: str) -> RelevanceResult:
        if self.datahub.is_noise(text):
            return RelevanceResult(is_noise=True, cosine_score=None, band="noise")

        message_embedding = self.embedder.embed([text])[0]
        scores = [
            self.embedder.cosine_score(message_embedding, anchor_embedding)
            for anchor_embedding in self.anchor_embeddings
        ]
        best_idx = int(np.argmax(scores))
        score = scores[best_idx]
        matched_facet = self.anchor_texts[best_idx]

        if score < self.low:
            band = "unrelated"
        elif score >= self.high:
            band = "relevant"
        else:
            band = "ambiguous"

        result = RelevanceResult(is_noise=False, cosine_score=score, band=band,
                                  matched_facet=matched_facet, embedding=message_embedding)

        if band == "ambiguous":
            verdict, reason = self.llm_client.classify_relevance(self.anchor_texts, text)
            result.llm_verdict = verdict
            result.llm_reason = reason

        return result


if __name__ == "__main__":
    import json

    from src.anchor import AnchorExtractor
    from src.datahub import Datahub
    from src.embedder import MockEmbedder
    from src.llm_client import MockLLMClient
    from src.store import Store

    with open("examples/sample_conversation.json") as f:
        messages = json.load(f)

    embedder = MockEmbedder()
    datahub = Datahub("data/datahub.json")
    llm_client = MockLLMClient()

    anchor_texts = AnchorExtractor(llm_client, datahub, k=2).extract(messages)
    print(f"Anchors ({len(anchor_texts)}):")
    for a in anchor_texts:
        print(f"  - {a[:100]}")
    print()

    scorer = DriftScorer(embedder, datahub, llm_client, anchor_texts)

    store = Store(":memory:")
    store.create_session("demo-session", anchor_texts, scorer.anchor_embeddings)

    for m in messages:
        result = scorer.score_message(m["text"])
        store.add_message(
            m["id"], "demo-session", m["role"], m["text"],
            embedding=result.embedding, is_noise=result.is_noise, band=result.band,
        )
        score_str = f"{result.cosine_score:.3f}" if result.cosine_score is not None else "  -  "
        verdict_str = f" | {result.llm_verdict}" if result.llm_verdict else ""
        print(f"{m['id']:>4} [{result.band:^9}] score={score_str}{verdict_str} : {m['text'][:50]}")

    store.close()
