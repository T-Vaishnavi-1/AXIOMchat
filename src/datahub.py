import json
import re


def _looks_like_list_line(line: str) -> bool:
    """
    Heuristic for a structured-inventory line (e.g. a resume's
    "Programming C++, Python, Java, SQL" skills-table row): mostly-short,
    comma/pipe-delimited segments, no sentence-ending punctuation. Not a
    general-purpose list detector — a cheap, deliberately narrow heuristic
    targeting the specific pattern that caused facet extraction to treat
    individual tool/language names as discussion topics (see FINDINGS.md).
    """
    stripped = line.strip()
    if not stripped or stripped[-1:] in ".?!":
        return False
    segments = [s.strip() for s in re.split(r"[,|]", stripped) if s.strip()]
    if len(segments) < 3:
        return False
    short = sum(1 for s in segments if len(s.split()) <= 3)
    return short / len(segments) >= 0.7


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

    def strip_structured_lists(self, text: str) -> str:
        """
        Drops lines that look like a structured inventory rather than
        genuine prose discussion, so e.g. a resume's skills table doesn't
        get treated as active discussion topics during anchor extraction.
        Prose (project descriptions, sentences ending in punctuation) is
        left untouched.
        """
        kept = [line for line in text.splitlines() if not _looks_like_list_line(line)]
        return "\n".join(kept).strip()
