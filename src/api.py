import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.anchor import AnchorExtractor
from src.datahub import Datahub
from src.embedder import Embedder, MockEmbedder, get_embedder
from src.llm_client import get_llm_client
from src.relevance import DriftScorer
from src.store import Store

app = FastAPI(title="AXIOMchat")

_embedder: Embedder | MockEmbedder | None = None
_datahub: Datahub | None = None
_store: Store | None = None
_scorers: dict[str, DriftScorer] = {}  # in-memory cache; reconstructed from Store on miss


@app.on_event("startup")
def startup():
    global _embedder, _datahub, _store
    _embedder = get_embedder()  # AXIOM_EMBEDDER=mock|real (default: real)
    _datahub = Datahub("data/datahub.json")
    _store = Store("axiomchat.db")


def _get_scorer(session_id: str) -> DriftScorer:
    """
    Looks up a live DriftScorer for this session, reconstructing it from Store
    if it's not in the in-memory cache (e.g. after a restart/redeploy) — the
    anchor facets survive in SQLite even if the in-process cache doesn't.
    """
    if session_id in _scorers:
        return _scorers[session_id]

    anchors = _store.get_session_anchors(session_id)
    if anchors is None:
        raise HTTPException(status_code=404, detail="session not found")

    anchor_texts = [text for text, _ in anchors]
    llm_client = get_llm_client()
    scorer = DriftScorer(_embedder, _datahub, llm_client, anchor_texts)
    _scorers[session_id] = scorer
    return scorer


class MessageIn(BaseModel):
    role: str
    text: str


class CreateSessionRequest(BaseModel):
    messages: list[MessageIn]


class CreateSessionResponse(BaseModel):
    session_id: str
    anchor_texts: list[str]


@app.post("/sessions", response_model=CreateSessionResponse)
def create_session(req: CreateSessionRequest):
    llm_client = get_llm_client()
    anchor_texts = AnchorExtractor(llm_client, _datahub).extract(
        [m.model_dump() for m in req.messages]
    )

    scorer = DriftScorer(_embedder, _datahub, llm_client, anchor_texts)
    session_id = str(uuid.uuid4())
    _scorers[session_id] = scorer
    _store.create_session(session_id, anchor_texts, scorer.anchor_embeddings)

    return CreateSessionResponse(session_id=session_id, anchor_texts=anchor_texts)


class ScoreMessageRequest(BaseModel):
    text: str
    role: str = "user"


class ScoreMessageResponse(BaseModel):
    band: str
    is_noise: bool
    cosine_score: float | None
    matched_facet: str | None
    llm_verdict: str | None
    llm_reason: str | None


@app.post("/sessions/{session_id}/messages", response_model=ScoreMessageResponse)
def score_message(session_id: str, req: ScoreMessageRequest):
    scorer = _get_scorer(session_id)
    result = scorer.score_message(req.text)

    _store.add_message(
        str(uuid.uuid4()), session_id, req.role, req.text,
        embedding=result.embedding, is_noise=result.is_noise, band=result.band,
    )

    return ScoreMessageResponse(
        band=result.band, is_noise=result.is_noise, cosine_score=result.cosine_score,
        matched_facet=result.matched_facet,
        llm_verdict=result.llm_verdict, llm_reason=result.llm_reason,
    )


@app.get("/health")
def health():
    return {"status": "ok"}
