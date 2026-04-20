from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
from http import HTTPStatus
from pathlib import Path
import re

from ..config import DashScopeConfig, RagConfig
from .pdf_ingest import extract_pdf_chunks
from .store import RagStore


class RagIndexingError(RuntimeError):
    pass


def _read_text_file(path: Path) -> str:
    data = path.read_bytes()
    for enc in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _chunk_text(text: str, *, chunk_chars: int, overlap: int) -> list[str]:
    text = text.replace("\r\n", "\n")
    if chunk_chars <= 0:
        return [text]
    overlap = max(0, min(overlap, chunk_chars - 1))
    chunks: list[str] = []
    i = 0
    while i < len(text):
        j = min(len(text), i + chunk_chars)
        chunk = text[i:j].strip()
        if chunk:
            chunks.append(chunk)
        if j >= len(text):
            break
        i = j - overlap
    return chunks


def _clean_plain_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(?<=[\u3400-\u9fff])\s+(?=[\u3400-\u9fff])", "", text)
    text = re.sub(r"[ \t\u3000]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


@dataclass(frozen=True)
class IndexResult:
    indexed_chunks: int


class RagIndexer:
    def __init__(self, *, dashscope: DashScopeConfig, rag: RagConfig, store: RagStore):
        self._dashscope_cfg = dashscope
        self._rag_cfg = rag
        self._store = store

    def index_paths(self, paths: list[Path]) -> IndexResult:
        texts: list[tuple[str, int, str, dict]] = []
        for p in paths:
            p = Path(p)
            if not p.exists():
                continue
            if p.is_dir():
                for sub in p.rglob("*"):
                    if sub.is_file():
                        texts.extend(self._extract(sub))
            else:
                texts.extend(self._extract(p))

        if not texts:
            return IndexResult(indexed_chunks=0)

        embeddings = self._embed([t[2] for t in texts], text_type="document")
        if len(embeddings) != len(texts):
            raise RagIndexingError("Embedding 返回数量与文本数量不一致。")

        for (source, chunk_index, chunk_text, meta), emb in zip(texts, embeddings):
            chunk_id = sha1(f"{source}::{chunk_index}::{chunk_text}".encode("utf-8")).hexdigest()
            self._store.upsert_chunk(
                chunk_id=chunk_id,
                source=source,
                chunk_index=chunk_index,
                text=chunk_text,
                embedding=emb,
                meta=meta,
            )

        return IndexResult(indexed_chunks=len(texts))

    def _extract(self, path: Path) -> list[tuple[str, int, str, dict]]:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            chunks = extract_pdf_chunks(
                path,
                chunk_chars=self._rag_cfg.chunk_chars,
                overlap=self._rag_cfg.chunk_overlap,
            )
            return [(c.source, c.chunk_index, c.text, c.meta) for c in chunks]  # type: ignore[return-value]
        if suffix not in {".txt", ".md"}:
            return []
        text = _clean_plain_text(_read_text_file(path))
        chunks = _chunk_text(
            text,
            chunk_chars=self._rag_cfg.chunk_chars,
            overlap=self._rag_cfg.chunk_overlap,
        )
        source = str(path.as_posix())
        return [(source, i, c, {"source": source, "chunk_index": i}) for i, c in enumerate(chunks)]  # type: ignore[return-value]

    def _embed(self, inputs: list[str], *, text_type: str) -> list[list[float]]:
        try:
            import dashscope  # type: ignore
            from dashscope import TextEmbedding  # type: ignore
        except ImportError as exc:
            raise RagIndexingError("dashscope 未安装。请先执行: pip install dashscope") from exc

        dashscope.api_key = self._dashscope_cfg.api_key
        if self._dashscope_cfg.base_http_api_url:
            dashscope.base_http_api_url = self._dashscope_cfg.base_http_api_url

        out: list[list[float]] = []
        batch = 8
        for i in range(0, len(inputs), batch):
            chunk = inputs[i : i + batch]
            resp = TextEmbedding.call(
                model=self._rag_cfg.embedding_model,
                input=chunk,
                dimension=self._rag_cfg.embedding_dimension,
                text_type=text_type,
            )
            if resp.status_code != HTTPStatus.OK:
                raise RagIndexingError(
                    f"Embedding 调用失败: request_id={resp.request_id} code={resp.code} message={resp.message}"
                )
            embeddings = resp.output["embeddings"]
            for e in embeddings:
                out.append(list(e["embedding"]))
        return out
