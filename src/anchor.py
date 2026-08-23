class AnchorExtractor:
    """
    One-shot session-anchor extraction, computed once k user messages have
    accumulated and then frozen. k is a trigger (how long to wait before
    computing), not the anchor's content — the window doesn't get collapsed
    straight into a single point in embedding space, which lost too much
    information (a message about a different real facet of the same session
    scored as if it were drift, since it wasn't close to one averaged goal).
    Instead this returns multiple distinct facet anchors:
      1. summarize()      — detail-preserving summary of the k-message window
      2. extract_facets() — distills that summary into a short list of
                             distinct topics/tasks, not one blended sentence
    Falls back to a single broader-window summary anchor if step 2 comes back
    UNCLEAR. Produces anchor *text* only — embedding it is DriftScorer's job.
    """

    def __init__(self, llm_client, datahub, k: int = 3):
        self.llm_client = llm_client
        self.datahub = datahub
        self.k = k

    def extract(self, messages: list[dict]) -> list[str]:
        """
        messages: [{"role": ..., "text": ...}, ...]. Only role == "user"
        messages are eligible to seed the window — the anchor represents
        the user's intent, so assistant output must never be a candidate,
        even if it's long and clears the noise filter.

        Returns a list of facet-anchor texts (always at least one).
        """
        user_texts = [m["text"] for m in messages if m.get("role") == "user"]
        non_noise = [t for t in user_texts if not self.datahub.is_noise(t)]
        window = non_noise[: self.k]

        if not window:
            return [self.llm_client.extract_anchor_fallback(user_texts)]

        # strip structured inventories (e.g. a resume's skills table) before
        # summarizing, so individual tool/language names aren't mistaken for
        # active discussion topics later in extract_facets()
        window = [self.datahub.strip_structured_lists(t) for t in window]

        summary = self.llm_client.summarize(window)
        facets = self.llm_client.extract_facets(summary)

        if facets == ["UNCLEAR"]:
            # widen to the full transcript (including assistant turns) for
            # the fallback — more context can help here, since this is
            # already the "cast a wider net" escape hatch
            full_texts = [m["text"] for m in messages]
            return [self.llm_client.extract_anchor_fallback(full_texts)]

        return facets
