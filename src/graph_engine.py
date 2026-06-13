import networkx as nx
from src.datahub import Datahub

# ── Datahub ─────────────────────────────────────────────────────────────────

class Datahub:
    def __init__(self, path="datahub.json"):
        with open(path) as f:
            data = json.load(f)
        self.topics      = data["topics"]
        self.global_noise = set(p.lower() for p in data["global_noise"])

    def is_noise(self, text: str) -> bool:
        """True if the text is a known noise pattern."""
        t = text.strip().lower()
        return t in self.global_noise or len(t.split()) <= 2

    def match_topics(self, text: str) -> set:
        """Returns set of topic names whose keywords appear in text."""
        t = text.lower()
        matched = set()
        for topic, entry in self.topics.items():
            for kw in entry["keywords"]:
                if kw in t:
                    matched.add(topic)
                    break
        return matched

    def topic_overlap(self, text_a: str, text_b: str) -> float:
        """
        Edge weight between two nodes — Jaccard similarity of their matched topics.
        Returns 0.0 if either node matches no topics (overlap undefined).
        """
        a = self.match_topics(text_a)
        b = self.match_topics(text_b)
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)


# ── Scorer ───────────────────────────────────────────────────────────────────

class LLMScorer:
    """
    V1 scorer: calls Anthropic API to rate relevance of a node
    relative to a session summary.
    """
    def __init__(self):
        # api key handled by environment
        self.model = "claude-sonnet-4-20250514"

    def score(self, node_text: str, session_summary: str) -> float:
        """Returns relevance score 0.0 - 1.0."""
        import urllib.request
        prompt = f"""You are evaluating whether a message in an LLM chat session is relevant to the session's core topic.

Session summary: {session_summary}

Message: {node_text}

Rates the relevance of this message to the session summary on a scale from 0.0 to 1.0.
- 1.0 = directly relevant, load-bearing context
- 0.5 = tangentially related
- 0.0 = completely irrelevant, noise, or social filler

Responds with ONLY a single float number, nothing else."""

        body = json.dumps({
            "model": self.model,
            "max_tokens": 10,
            "messages": [{"role": "user", "content": prompt}]
        }).encode()

        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=body,
            headers={
                "content-type": "application/json",
                "anthropic-version": "2023-06-01"
            }
        )
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read())
        raw = result["content"][0]["text"].strip()
        return float(raw)


# ── Engine ───────────────────────────────────────────────────────────────────

class ChatGraphEngine:
    def __init__(self, datahub: Datahub, scorer: LLMScorer = None, threshold: float = 0.3):
        self.G         = nx.DiGraph()
        self.datahub   = datahub
        self.scorer    = scorer
        self.threshold = threshold

    # ── Core graph ops ───────────────────────────────────────────────────────

    def add_message(self, msg_id, parent_id, text):
        self.G.add_node(msg_id, text=text, score=None, final_score=None)
        if parent_id is not None:
            if not self.G.has_node(parent_id):
                raise ValueError(f"Parent '{parent_id}' does not exist.")
            parent_text = self.G.nodes[parent_id]["text"]
            weight = self.datahub.topic_overlap(parent_text, text)
            self.G.add_edge(parent_id, msg_id, weight=weight)

    # ── Scoring ──────────────────────────────────────────────────────────────

    def _infer_session_summary(self) -> str:
        """
        Build session anchor: concatenates text of nodes that look like
        substantive outputs (long assistant-style messages, not noise).
        """
        candidates = [
            self.G.nodes[n]["text"]
            for n in self.G.nodes
            if not self.datahub.is_noise(self.G.nodes[n]["text"])
        ]
        return " | ".join(candidates) if candidates else "general software engineering session"

    def score_session(self):
        """
        Two-pass scoring:
          Pass 1 — scores each node individually via LLM or noise heuristic
          Pass 2 — propagates scores bottom-up, stopping at branching nodes
        """
        summary = self._infer_session_summary()
        print(f"  Session anchor: \"{summary[:80]}...\"" if len(summary) > 80 else f"  Session anchor: \"{summary}\"")

        # Pass 1: individual scores
        for n in self.G.nodes:
            text = self.G.nodes[n]["text"]
            if self.datahub.is_noise(text):
                self.G.nodes[n]["score"] = 0.0
            elif self.scorer:
                self.G.nodes[n]["score"] = self.scorer.score(text, summary)
            else:
                # fallback: topic match ratio as proxy score
                topics = self.datahub.match_topics(text)
                self.G.nodes[n]["score"] = min(1.0, len(topics) * 0.3)

        # Pass 2: bottom-up propagation (post-order traversal)
        for n in reversed(list(nx.topological_sort(self.G))):
            children    = list(self.G.successors(n))
            own_score   = self.G.nodes[n]["score"]

            # branching nodes score themselves only — don't inherit
            if len(children) >= 2:
                self.G.nodes[n]["final_score"] = own_score
            elif len(children) == 1:
                child_final = self.G.nodes[children[0]]["final_score"]
                self.G.nodes[n]["final_score"] = max(own_score, child_final)
            else:
                # leaf
                self.G.nodes[n]["final_score"] = own_score

    # ── Deletion logic ───────────────────────────────────────────────────────

    def can_delete(self, msg_id):
        if not self.G.has_node(msg_id):
            return False, "does not exist"

        # structural gate: must be a leaf
        strong_children = [
          c for c in self.G.successors(msg_id)
          if self.G[msg_id][c]['weight'] > 0.0]
        if strong_children:
               return False, f"has {len(strong_children)} strongly linked reply(s): {strong_children}"

        # semantic gate: score must be below threshold (if scored)
        final_score = self.G.nodes[msg_id].get("final_score")
        if final_score is not None and final_score >= self.threshold:
            return False, f"score {final_score:.2f} >= threshold {self.threshold}"

        return True, f"leaf, score {final_score:.2f}" if final_score is not None else "leaf node (unscored)"

    def delete_message(self, msg_id):
        ok, reason = self.can_delete(msg_id)
        if not ok:
            print(f"❌ BLOCKED '{msg_id}': {reason}")
            return False
        self.G.remove_node(msg_id)
        print(f"✅ DELETED '{msg_id}' ({reason})")
        return True

    def get_deletable_nodes(self):
        return [n for n in self.G.nodes if self.can_delete(n)[0]]

    # ── Component queries ────────────────────────────────────────────────────

    def get_components(self):
        undirected = self.G.to_undirected()
        isolated, threads = [], []
        for component in nx.connected_components(undirected):
            if len(component) == 1:
                isolated.append(list(component)[0])
            else:
                threads.append(sorted(component))
        return isolated, threads

    # ── Inspection ───────────────────────────────────────────────────────────

    def summary(self):
        isolated, threads = self.get_components()
        deletable = self.get_deletable_nodes()
        print(f"\n── Graph summary ───────────────────────────────────")
        print(f"Nodes     : {list(self.G.nodes)}")
        edges = [(u, v, round(d['weight'], 2)) for u, v, d in self.G.edges(data=True)]
        print(f"Edges     : {edges}")
        print(f"Isolated  : {isolated}")
        print(f"Threads   : {threads}")
        for n in self.G.nodes:
            fs = self.G.nodes[n].get('final_score')
            score_str = f"{fs:.2f}" if fs is not None else "unscored"
            print(f"  {n}: \"{self.G.nodes[n]['text'][:40]}\" | final_score={score_str}")
        print(f"Deletable : {deletable}")
        print(f"────────────────────────────────────────────────────\n")


# ── Smoke test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    from src.datahub import Datahub

    with open("examples/sample_conversation.json") as f:
        messages = json.load(f)

    datahub = Datahub("data/datahub.json")
    engine  = ChatGraphEngine(datahub, scorer=None, threshold=0.25)

    for m in messages:
        engine.add_message(m["id"], m["parent_id"], m["text"])

    engine.score_session()
    engine.summary()

    print("── Deletion pass ──")
    for n in list(engine.G.nodes):
        engine.delete_message(n)
