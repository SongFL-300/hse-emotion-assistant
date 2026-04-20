from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3
import time
import json
from array import array
import threading


def _floats_to_blob(values: list[float]) -> bytes:
    arr = array("f", values)
    return arr.tobytes()


def _blob_to_floats(blob: bytes) -> list[float]:
    arr = array("f")
    arr.frombytes(blob)
    return list(arr)


@dataclass(frozen=True)
class RagChunk:
    chunk_id: str
    source: str
    chunk_index: int
    text: str
    embedding: list[float]
    meta: dict


class RagStore:
    def __init__(self, db_path: Path):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._init_schema()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS rag_chunks (
              chunk_id TEXT PRIMARY KEY,
              source TEXT NOT NULL,
              chunk_index INTEGER NOT NULL,
              text TEXT NOT NULL,
              embedding BLOB NOT NULL,
              dim INTEGER NOT NULL,
              meta_json TEXT NOT NULL,
              created_at REAL NOT NULL
            );
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_rag_source ON rag_chunks(source);"
        )
        self._conn.commit()

    def count(self) -> int:
        with self._lock:
            cur = self._conn.execute("SELECT COUNT(1) FROM rag_chunks;")
            row = cur.fetchone()
            return int(row[0]) if row else 0

    def clear(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM rag_chunks;")
            self._conn.commit()

    def upsert_chunk(
        self,
        *,
        chunk_id: str,
        source: str,
        chunk_index: int,
        text: str,
        embedding: list[float],
        meta: dict | None = None,
    ) -> None:
        meta_json = json.dumps(meta or {}, ensure_ascii=False)
        blob = _floats_to_blob(embedding)
        dim = len(embedding)
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO rag_chunks(chunk_id, source, chunk_index, text, embedding, dim, meta_json, created_at)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                  source=excluded.source,
                  chunk_index=excluded.chunk_index,
                  text=excluded.text,
                  embedding=excluded.embedding,
                  dim=excluded.dim,
                  meta_json=excluded.meta_json,
                  created_at=excluded.created_at;
                """,
                (chunk_id, source, int(chunk_index), text, blob, dim, meta_json, time.time()),
            )
            self._conn.commit()

    def iter_chunks(self) -> list[RagChunk]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT chunk_id, source, chunk_index, text, embedding, meta_json FROM rag_chunks;"
            )
            out: list[RagChunk] = []
            for chunk_id, source, chunk_index, text, blob, meta_json in cur.fetchall():
                out.append(
                    RagChunk(
                        chunk_id=str(chunk_id),
                        source=str(source),
                        chunk_index=int(chunk_index),
                        text=str(text),
                        embedding=_blob_to_floats(blob),
                        meta=json.loads(meta_json) if meta_json else {},
                    )
                )
            return out
