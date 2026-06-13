import json

class Datahub:
    def __init__(self, path="data/datahub.json"):
        with open(path) as f:
            data = json.load(f)
        self.topics       = data["topics"]
        self.global_noise = set(p.lower() for p in data["global_noise"])

    def is_noise(self, text: str) -> bool:
        t = text.strip().lower()
        return t in self.global_noise or len(t.split()) <= 2

    def match_topics(self, text: str) -> set:
        t = text.lower()
        matched = set()
        for topic, entry in self.topics.items():
            for kw in entry["keywords"]:
                if kw in t:
                    matched.add(topic)
                    break
        return matched

    def topic_overlap(self, text_a: str, text_b: str) -> float:
        """Jaccard similarity of matched topics — lexical edge weight."""
        a = self.match_topics(text_a)
        b = self.match_topics(text_b)
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)