import os
from abc import ABC, abstractmethod

import ollama as ollama_sdk
from anthropic import Anthropic


SUMMARY_PROMPT = """Here are messages from a user in a conversation:
{messages}

Summarize what this person has been discussing — the specific topics,
projects, questions, or concerns they've raised. Preserve concrete details
(names, technologies, specific asks) rather than compressing everything down
to one abstract goal."""

FACETS_PROMPT = """Here is a summary of a conversation's opening:
{summary}

List the distinct topics, projects, or tasks this person is actively working
on or discussing, each as a short, specific phrase. Merge closely related
items into one facet rather than listing near-duplicates separately (e.g.
"CUDA programming" and "parallel computing" describe the same skill — list
it once). Respond with one facet per line, nothing else — no introduction,
no numbering, no preamble sentence. If nothing concrete emerges, respond
with exactly the word UNCLEAR."""

FALLBACK_PROMPT = """Here is the conversation so far:
{history}

Summarize what this person is actually trying to accomplish, in one sentence,
ignoring examples or tangents."""

RELEVANCE_PROMPT = """This session covers these topics:
{anchors}

Message: {message}

Does this message serve any of the topics above, or does it pursue something
different entirely? Respond in exactly this format on one line:
VERDICT: one-sentence reason
where VERDICT is either RELEVANT or OFF_TOPIC."""


def _format_list(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def _parse_facets(raw: str) -> list[str]:
    """
    One facet per line, most-common formats stripped. Lines ending in ':'
    are dropped as likely preamble ("Here are the distinct topics:") rather
    than an actual facet — belt-and-suspenders alongside the prompt's own
    instruction not to add one, since models don't always follow that.
    """
    lines = [line.strip(" -*") for line in raw.strip().splitlines()]
    return [line for line in lines if line and not line.endswith(":")]


def _parse_verdict(raw: str) -> tuple[str, str]:
    """Parses a 'VERDICT: reason' line into (verdict, reason)."""
    verdict, _, reason = raw.strip().partition(":")
    verdict = verdict.strip().upper()
    if verdict not in ("RELEVANT", "OFF_TOPIC"):
        # model didn't follow the format — fall back to a keyword check
        verdict = "OFF_TOPIC" if "OFF_TOPIC" in raw.upper() else "RELEVANT"
    return verdict, reason.strip()


class LLMClient(ABC):
    @abstractmethod
    def _complete(self, prompt: str) -> str:
        """Sends a prompt to the underlying model, returns the raw text reply."""

    def summarize(self, messages: list[str]) -> str:
        """First pass: a detail-preserving summary, not a compressed goal."""
        prompt = SUMMARY_PROMPT.format(messages=_format_list(messages))
        return self._complete(prompt).strip()

    def extract_facets(self, summary: str) -> list[str]:
        """
        Second pass: distills the summary into a short list of distinct
        topic/task facets (rather than one blended sentence), so a later
        message can match whichever facet it's actually about instead of
        being judged against a single averaged point. Returns ['UNCLEAR']
        if nothing concrete emerges.
        """
        prompt = FACETS_PROMPT.format(summary=summary)
        raw = self._complete(prompt).strip()
        if raw.upper() == "UNCLEAR":
            return ["UNCLEAR"]
        facets = _parse_facets(raw)
        return facets if facets else ["UNCLEAR"]

    def extract_anchor_fallback(self, history: list[str]) -> str:
        """Broader-summary intent extraction, used when extract_facets is unclear."""
        prompt = FALLBACK_PROMPT.format(history=_format_list(history))
        return self._complete(prompt).strip()

    def classify_relevance(self, anchors: list[str], message: str) -> tuple[str, str]:
        """Returns (verdict, reason) where verdict is 'RELEVANT' or 'OFF_TOPIC'."""
        prompt = RELEVANCE_PROMPT.format(anchors=_format_list(anchors), message=message)
        return _parse_verdict(self._complete(prompt))


class OllamaClient(LLMClient):
    """Local inference via a running Ollama server (default: localhost:11434)."""

    def __init__(self, model: str | None = None):
        self.model = model or os.environ.get("AXIOM_LLM_MODEL", "qwen2.5:3b")

    def _complete(self, prompt: str) -> str:
        response = ollama_sdk.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"]


class AnthropicClient(LLMClient):
    """Hosted inference via the Anthropic API — upgrade path for accuracy."""

    def __init__(self, model: str | None = None):
        self.model = model or os.environ.get("AXIOM_ANTHROPIC_MODEL", "claude-sonnet-5")
        self._client = Anthropic()  # reads ANTHROPIC_API_KEY from the environment

    def _complete(self, prompt: str) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text


class MockLLMClient(LLMClient):
    """
    Deterministic, no network calls — lets the rest of the pipeline be tested
    without a running Ollama server or an Anthropic API key. Dispatches on
    fixed substrings from the prompt templates above rather than on which
    public method was called, since _complete is the only hook subclasses get.

    For summarization/faceting, echoes the actual content back (rather than a
    fixed placeholder string) so the mock anchors share real vocabulary with
    the conversation they were built from. For faceting specifically, splits
    the echoed content into sentence-ish chunks so multi-facet scoring has
    more than one distinct anchor to exercise, without claiming any real
    topic-identification intelligence.
    """

    def _complete(self, prompt: str) -> str:
        if "This session covers these topics:" in prompt:
            return "RELEVANT: mock verdict, always on-topic for testing"
        if "Summarize what this person has been discussing" in prompt:
            return self._echo_content(prompt, "conversation:\n")
        if "List the distinct topics" in prompt:
            content = self._echo_content(prompt, "opening:\n")
            facets = [s.strip() for s in content.replace("\n", " ").split(".") if s.strip()]
            return "\n".join(facets) if facets else content
        if "conversation so far" in prompt:
            return self._echo_content(prompt, "so far:\n")
        return "Mock summary of the conversation so far"

    @staticmethod
    def _echo_content(prompt: str, marker: str) -> str:
        start = prompt.index(marker) + len(marker)
        end = prompt.index("\n\n", start)
        return prompt[start:end].strip()


def get_llm_client() -> LLMClient:
    """Reads AXIOM_LLM_PROVIDER (default: ollama) and returns the matching client."""
    provider = os.environ.get("AXIOM_LLM_PROVIDER", "ollama").lower()
    if provider == "ollama":
        return OllamaClient()
    if provider == "anthropic":
        return AnthropicClient()
    if provider == "mock":
        return MockLLMClient()
    raise ValueError(f"Unknown AXIOM_LLM_PROVIDER: {provider!r}")
