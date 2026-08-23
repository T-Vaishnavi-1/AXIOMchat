import sqlite3
import threading
from datetime import datetime, timezone

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS session_anchors (
    session_id TEXT NOT NULL,
    facet_text TEXT NOT NULL,
    facet_embedding BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    text TEXT NOT NULL,
    embedding BLOB,
    is_noise INTEGER,
    band TEXT,
    created_at TEXT
);

CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    text, content='messages', content_rowid='rowid'
);
"""


def _to_blob(vector: np.ndarray) -> bytes:
    return vector.astype(np.float32).tobytes()


def _from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


class Store:
    """
    Minimal SQLite persistence for sessions/messages. A session has multiple
    anchor facets (see anchor.py/relevance.py — a single averaged anchor lost
    too much information), stored one row per facet in session_anchors rather
    than as a single column, so the schema doesn't need to encode a list into
    one field. Embeddings are stored as plain BLOBs and compared via
    brute-force numpy cosine at query time — a single chat session is at most
    a few hundred messages, not a scale that justifies a vector database.

    One connection is opened once and reused (cheap for a local SQLite file),
    but a FastAPI app runs synchronous route handlers in a worker thread pool
    — a different thread per request than the one that created the
    connection. sqlite3 blocks cross-thread use of a connection by default,
    so check_same_thread=False lifts that restriction; a lock then serializes
    actual access, since lifting the check alone doesn't make concurrent
    access from multiple threads safe, only permitted.
    """

    def __init__(self, path: str = "axiomchat.db"):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self.conn.executescript(SCHEMA)
            self.conn.commit()

    def create_session(self, session_id: str, anchor_texts: list[str],
                        anchor_embeddings: np.ndarray):
        with self._lock:
            self.conn.execute(
                "INSERT INTO sessions (id, created_at) VALUES (?, ?)",
                (session_id, datetime.now(timezone.utc).isoformat()),
            )
            self.conn.executemany(
                "INSERT INTO session_anchors (session_id, facet_text, facet_embedding) "
                "VALUES (?, ?, ?)",
                [(session_id, text, _to_blob(emb))
                 for text, emb in zip(anchor_texts, anchor_embeddings)],
            )
            self.conn.commit()

    def get_session_anchors(self, session_id: str) -> list[tuple[str, np.ndarray]] | None:
        with self._lock:
            rows = self.conn.execute(
                "SELECT facet_text, facet_embedding FROM session_anchors WHERE session_id = ?",
                (session_id,),
            ).fetchall()
        if not rows:
            return None
        return [(text, _from_blob(blob)) for text, blob in rows]

    def add_message(self, message_id: str, session_id: str, role: str, text: str,
                     embedding: np.ndarray | None, is_noise: bool, band: str):
        blob = _to_blob(embedding) if embedding is not None else None
        with self._lock:
            cur = self.conn.execute(
                "INSERT INTO messages (id, session_id, role, text, embedding, is_noise, band, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (message_id, session_id, role, text, blob, int(is_noise), band,
                 datetime.now(timezone.utc).isoformat()),
            )
            self.conn.execute(
                "INSERT INTO messages_fts (rowid, text) VALUES (?, ?)",
                (cur.lastrowid, text),
            )
            self.conn.commit()

    def close(self):
        self.conn.close()
