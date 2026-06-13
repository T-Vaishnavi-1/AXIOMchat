import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pandas as pd
import json


class Embedder:
    """Real embedder — use on your local machine."""
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        print(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True)

    def cosine_score(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(cosine_similarity([a], [b])[0][0])


class MockEmbedder:
    """
    Drop-in replacement when HuggingFace is unavailable.
    Uses deterministic hash-seeded vectors — same text always gets same vector.
    Replace with Embedder() on your local machine.
    """
    def __init__(self, dim=384):
        self.dim = dim

    def _text_vector(self, text: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(text.lower().strip())) % (2**32))
        return rng.random(self.dim).astype(np.float32)

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._text_vector(t) for t in texts])

    def cosine_score(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(cosine_similarity([a], [b])[0][0])


class SemanticScorer:
    """
    Scores each node against the session anchor via cosine similarity.
    Also computes semantic edge weights between parent-child pairs.
    """
    def __init__(self, embedder):
        self.embedder = embedder

    def score_nodes(self, nodes: dict) -> dict:
        ids   = list(nodes.keys())
        texts = [nodes[i] for i in ids]
        embs  = self.embedder.embed(texts)
        return {id_: {"text": nodes[id_], "embedding": embs[i]}
                for i, id_ in enumerate(ids)}

    def session_anchor(self, scored_nodes: dict, noise_ids: set) -> np.ndarray:
        """Mean embedding of non-noise nodes — session center of mass."""
        valid = [v["embedding"] for k, v in scored_nodes.items()
                 if k not in noise_ids]
        if not valid:
            raise ValueError("All nodes are noise — cannot build session anchor.")
        return np.mean(valid, axis=0)

    def relevance_scores(self, scored_nodes: dict, anchor: np.ndarray) -> dict:
        return {
            id_: self.embedder.cosine_score(v["embedding"], anchor)
            for id_, v in scored_nodes.items()
        }

    def edge_weight(self, emb_a: np.ndarray, emb_b: np.ndarray) -> float:
        return self.embedder.cosine_score(emb_a, emb_b)


def score_report(messages: list[dict], scorer: SemanticScorer,
                 datahub_path="datahub.json", threshold=0.3) -> pd.DataFrame:
    """
    Given a flat list of {"id", "parent_id", "role", "text"} messages,
    returns a DataFrame with scores, edge weights, and deletion candidates.
    """
    with open(datahub_path) as f:
        datahub = json.load(f)
    global_noise = set(p.lower() for p in datahub["global_noise"])

    def is_noise(text):
        t = text.strip().lower()
        return t in global_noise or len(t.split()) <= 2

    nodes     = {m["id"]: m["text"] for m in messages}
    noise_ids = {id_ for id_, t in nodes.items() if is_noise(t)}
    scored    = scorer.score_nodes(nodes)
    anchor    = scorer.session_anchor(scored, noise_ids)
    scores    = scorer.relevance_scores(scored, anchor)

    id_to_parent = {m["id"]: m["parent_id"] for m in messages}
    edge_weights = {}
    for id_, parent_id in id_to_parent.items():
        if parent_id and parent_id in scored:
            edge_weights[id_] = scorer.edge_weight(
                scored[parent_id]["embedding"],
                scored[id_]["embedding"]
            )
        else:
            edge_weights[id_] = None

    rows = []
    for m in messages:
        id_ = m["id"]
        rows.append({
            "id":          id_,
            "role":        m["role"],
            "text":        m["text"][:60],
            "is_noise":    id_ in noise_ids,
            "relevance":   round(scores[id_], 3),
            "edge_weight": round(edge_weights[id_], 3) if edge_weights[id_] is not None else None,
        })

    df = pd.DataFrame(rows)
    df["deletable"] = df.apply(
        lambda r: r["is_noise"] or r["relevance"] < threshold, axis=1
    )
    return df


if __name__ == "__main__":
    messages = [
        {"id": "m1", "parent_id": None,  "role": "user",      "text": "Hi there!"},
        {"id": "m2", "parent_id": "m1",  "role": "user",      "text": "How do I allocate memory on a CUDA device?"},
        {"id": "m3", "parent_id": "m2",  "role": "assistant", "text": "Use cudaMalloc to allocate device memory. It works similarly to malloc but allocates on the GPU."},
        {"id": "m4", "parent_id": "m2",  "role": "user",      "text": "Can I use thrust with CUDA kernels?"},
        {"id": "m5", "parent_id": "m1",  "role": "user",      "text": "Thanks, got it!"},
        {"id": "m6", "parent_id": None,  "role": "user",      "text": "I want to learn something today"},
        {"id": "m7", "parent_id": "m3",  "role": "user",      "text": "What about cudaFree? When should I call it?"},
        {"id": "m8", "parent_id": "m3",  "role": "user",      "text": "Does this work on TPUs too?"},
    ]

    embedder = Embedder()        # swap to Embedder() on your machine
    scorer   = SemanticScorer(embedder)
    df       = score_report(messages, scorer)

    print("\n── Score Report ────────────────────────────────────")
    print(df.to_string(index=False))
    print(f"\nDeletion candidates: {list(df[df['deletable']]['id'])}")