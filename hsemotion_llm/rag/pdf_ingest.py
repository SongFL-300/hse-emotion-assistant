from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import sha1
import json
from pathlib import Path
import re
from typing import Iterable


_CJK_RE = re.compile(r"[\u3400-\u9fff]")
_PAGE_NUM_RE = re.compile(r"^\s*(第?\s*\d+\s*页?|[0-9IVXLCMivxlcm]+)\s*$")
_MULTISPACE_RE = re.compile(r"[ \t\u3000]+")
_MULTINEWLINE_RE = re.compile(r"\n{3,}")
_INLINE_CJK_SPACE_RE = re.compile(r"(?<=[\u3400-\u9fff])\s+(?=[\u3400-\u9fff])")


@dataclass(frozen=True)
class ExtractedChunk:
    source: str
    chunk_index: int
    text: str
    meta: dict


def extract_pdf_chunks(path: Path, *, chunk_chars: int, overlap: int) -> list[ExtractedChunk]:
    cached = _load_cached_chunks(path, chunk_chars=chunk_chars, overlap=overlap)
    if cached is not None:
        return cached
    page_texts = _extract_pdf_pages(path)
    if not page_texts:
        return []
    page_texts = _drop_common_headers_footers(page_texts)
    paragraphs = _pages_to_paragraphs(page_texts)
    chunks = _chunk_paragraphs(paragraphs, chunk_chars=chunk_chars, overlap=overlap)
    source = str(path.as_posix())
    out = [
        ExtractedChunk(
            source=source,
            chunk_index=i,
            text=chunk["text"],
            meta={
                "source": source,
                "chunk_index": i,
                "page_start": chunk["page_start"],
                "page_end": chunk["page_end"],
                "source_name": path.name,
                "kb_type": "professional_pdf",
            },
        )
        for i, chunk in enumerate(chunks)
    ]
    _save_cached_chunks(path, chunk_chars=chunk_chars, overlap=overlap, chunks=out)
    return out


def _extract_pdf_pages(path: Path) -> list[str]:
    try:
        import fitz  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("缺少 pymupdf，无法解析 PDF。") from exc

    try:
        from rapidocr_onnxruntime import RapidOCR  # type: ignore
    except Exception:
        RapidOCR = None  # type: ignore

    doc = fitz.open(path)
    ocr_engine = RapidOCR() if RapidOCR is not None else None
    pages: list[str] = []
    try:
        for page in doc:
            text = _clean_page_text(page.get_text("text"))
            if len(text) < 80 and ocr_engine is not None:
                text = _ocr_page(page, ocr_engine)
            pages.append(text)
    finally:
        doc.close()
    return pages


def _ocr_page(page, ocr_engine) -> str:  # noqa: ANN001
    try:
        import fitz  # type: ignore
    except Exception:
        return ""
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
    image_bytes = pix.tobytes("png")
    try:
        result, _ = ocr_engine(image_bytes)
    except Exception:
        return ""
    if not result:
        return ""
    lines = [str(item[1]).strip() for item in result if item and len(item) >= 2]
    return _clean_page_text("\n".join(lines))


def _clean_page_text(text: str) -> str:
    text = (text or "").replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _INLINE_CJK_SPACE_RE.sub("", text)
    text = _MULTISPACE_RE.sub(" ", text)

    raw_lines = [line.strip() for line in text.splitlines()]
    lines: list[str] = []
    for line in raw_lines:
        if not line:
            lines.append("")
            continue
        if _PAGE_NUM_RE.match(line):
            continue
        if len(line) <= 2 and not _CJK_RE.search(line):
            continue
        if _looks_like_noise(line):
            continue
        lines.append(line)

    merged: list[str] = []
    for line in lines:
        if not line:
            if merged and merged[-1] != "":
                merged.append("")
            continue
        if not merged or merged[-1] == "":
            merged.append(line)
            continue
        prev = merged[-1]
        if _should_join(prev, line):
            merged[-1] = f"{prev}{line}"
        else:
            merged.append(line)

    text = "\n".join([line for line in merged if line is not None]).strip()
    text = _MULTINEWLINE_RE.sub("\n\n", text)
    return text


def _looks_like_noise(line: str) -> bool:
    if len(line) < 3:
        return True
    cjk = len(_CJK_RE.findall(line))
    alpha = sum(ch.isalpha() for ch in line)
    digit = sum(ch.isdigit() for ch in line)
    punct = sum((not ch.isalnum()) and (not _CJK_RE.match(ch)) for ch in line)
    total = max(1, len(line))
    if cjk == 0 and alpha + digit < 3:
        return True
    if cjk > 0 and punct / total > 0.55:
        return True
    if cjk == 0 and alpha / total < 0.25 and digit / total < 0.25:
        return True
    return False


def _should_join(prev: str, cur: str) -> bool:
    if not prev or not cur:
        return False
    if prev.endswith(("。", "！", "？", "；", ":", "：", ".", "!", "?", ";")):
        return False
    if cur.startswith(("•", "-", "·", "●", "1.", "2.", "3.", "#")):
        return False
    if len(prev) < 16:
        return False
    return True


def _drop_common_headers_footers(page_texts: list[str]) -> list[str]:
    if len(page_texts) < 4:
        return page_texts

    header_counter: Counter[str] = Counter()
    footer_counter: Counter[str] = Counter()
    split_pages: list[list[str]] = []

    for text in page_texts:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        split_pages.append(lines)
        for line in lines[:2]:
            if len(line) <= 40:
                header_counter[line] += 1
        for line in lines[-2:]:
            if len(line) <= 40:
                footer_counter[line] += 1

    threshold = max(3, int(len(page_texts) * 0.08))
    common_headers = {line for line, count in header_counter.items() if count >= threshold}
    common_footers = {line for line, count in footer_counter.items() if count >= threshold}

    cleaned: list[str] = []
    for lines in split_pages:
        body = [line for line in lines if line not in common_headers and line not in common_footers]
        cleaned.append("\n".join(body).strip())
    return cleaned


def _pages_to_paragraphs(page_texts: list[str]) -> list[dict]:
    paragraphs: list[dict] = []
    for page_no, text in enumerate(page_texts, start=1):
        if not text:
            continue
        for block in text.split("\n\n"):
            block = block.strip()
            if not block or len(block) < 10:
                continue
            paragraphs.append({"page": page_no, "text": block})
    return paragraphs


def _chunk_paragraphs(paragraphs: list[dict], *, chunk_chars: int, overlap: int) -> list[dict]:
    if not paragraphs:
        return []
    chunks: list[dict] = []
    current_parts: list[str] = []
    current_len = 0
    page_start = paragraphs[0]["page"]
    page_end = paragraphs[0]["page"]

    for para in paragraphs:
        text = str(para["text"]).strip()
        if not text:
            continue
        add_len = len(text) + (2 if current_parts else 0)
        if current_parts and current_len + add_len > chunk_chars:
            chunk_text = "\n\n".join(current_parts).strip()
            if chunk_text:
                chunks.append({"text": chunk_text, "page_start": page_start, "page_end": page_end})
            overlap_parts = _tail_overlap(current_parts, overlap)
            current_parts = overlap_parts.copy()
            current_len = sum(len(x) for x in current_parts) + max(0, len(current_parts) - 1) * 2
            page_start = para["page"] if not current_parts else page_start
        if not current_parts:
            page_start = para["page"]
        current_parts.append(text)
        current_len += add_len
        page_end = para["page"]

    if current_parts:
        chunk_text = "\n\n".join(current_parts).strip()
        if chunk_text:
            chunks.append({"text": chunk_text, "page_start": page_start, "page_end": page_end})
    return _dedupe_chunks(chunks)


def _tail_overlap(parts: Iterable[str], overlap: int) -> list[str]:
    if overlap <= 0:
        return []
    tail: list[str] = []
    total = 0
    for item in reversed(list(parts)):
        tail.insert(0, item)
        total += len(item)
        if total >= overlap:
            break
    return tail


def _dedupe_chunks(chunks: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for item in chunks:
        normalized = re.sub(r"\s+", " ", item["text"]).strip()
        if normalized in seen:
            continue
        seen.add(normalized)
        out.append(item)
    return out


def _cache_path(path: Path, *, chunk_chars: int, overlap: int) -> Path:
    stat = path.stat()
    key = f"{path.resolve()}::{stat.st_size}::{stat.st_mtime_ns}::{chunk_chars}::{overlap}"
    digest = sha1(key.encode("utf-8")).hexdigest()
    base = Path(".rag") / "pdf_cache"
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{digest}.json"


def _load_cached_chunks(path: Path, *, chunk_chars: int, overlap: int) -> list[ExtractedChunk] | None:
    cache_path = _cache_path(path, chunk_chars=chunk_chars, overlap=overlap)
    if not cache_path.exists():
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, list):
        return None
    out: list[ExtractedChunk] = []
    for item in payload:
        try:
            out.append(
                ExtractedChunk(
                    source=str(item["source"]),
                    chunk_index=int(item["chunk_index"]),
                    text=str(item["text"]),
                    meta=dict(item["meta"]),
                )
            )
        except Exception:
            return None
    return out


def _save_cached_chunks(path: Path, *, chunk_chars: int, overlap: int, chunks: list[ExtractedChunk]) -> None:
    cache_path = _cache_path(path, chunk_chars=chunk_chars, overlap=overlap)
    try:
        payload = [
            {
                "source": item.source,
                "chunk_index": item.chunk_index,
                "text": item.text,
                "meta": item.meta,
            }
            for item in chunks
        ]
        cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except Exception:
        return
