from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
import math
import json
import re

from ..config import DashScopeConfig, RagConfig
from .store import RagStore, RagChunk


class RagRetrievalError(RuntimeError):
    pass


@dataclass(frozen=True)
class RagHit:
    chunk: RagChunk
    score: float


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    denom = math.sqrt(na) * math.sqrt(nb)
    return float(dot / denom) if denom > 0 else 0.0


class RagRetriever:
    def __init__(self, *, dashscope: DashScopeConfig, rag: RagConfig, store: RagStore):
        self._dashscope_cfg = dashscope
        self._rag_cfg = rag
        self._store = store

    def build_snippets(self, query: str) -> str | None:
        snippets, _ = self.build_snippets_with_hits(query)
        return snippets

    def build_snippets_with_hits(self, query: str) -> tuple[str | None, list[RagHit]]:
        hits = self.retrieve(query)
        if not hits:
            return None, []
        lines: list[str] = []
        for i, h in enumerate(hits, start=1):
            src = h.chunk.meta.get("source_name") or h.chunk.source
            p0 = h.chunk.meta.get("page_start")
            p1 = h.chunk.meta.get("page_end")
            page_tag = f" p.{p0}" if p0 == p1 and p0 else f" p.{p0}-{p1}" if p0 and p1 else ""
            lines.append(f"{i}. ({src}{page_tag} / chunk {h.chunk.chunk_index}) {h.chunk.text}")
        return "\n".join(lines), hits

    def should_retrieve(self, query: str) -> bool:
        if self._store.count() <= 0:
            return False
        heuristic = self._keyword_route(query)
        if heuristic is not None:
            return heuristic
        return self._llm_route(query)

    def retrieve(self, query: str) -> list[RagHit]:
        if self._store.count() <= 0:
            return []
        query_emb = self._embed_query(query)
        chunks = self._store.iter_chunks()

        scored = [
            RagHit(chunk=c, score=_cosine_similarity(query_emb, c.embedding)) for c in chunks
        ]
        scored.sort(key=lambda x: x.score, reverse=True)
        # 先召回一批，再 rerank
        candidates = scored[: max(self._rag_cfg.top_k * 3, 12)]
        if not candidates:
            return []

        reranked = self._rerank(query, candidates)
        return reranked[: self._rag_cfg.top_k]

    def _keyword_route(self, query: str) -> bool | None:
        text = (query or "").strip().lower()
        if not text:
            return False
        strong = [
            "who", "mhgap", "精神卫生", "心理", "抑郁", "焦虑", "自杀", "自伤", "失眠", "精神病",
            "药物", "用药", "症状", "诊断", "评估", "干预", "指南", "治疗", "应激", "创伤",
            "双相", "躁狂", "物质", "成瘾", "儿童", "青少年", "痴呆", "癫痫",
        ]
        if any(token in text for token in strong):
            return True
        casual = ["你好", "在吗", "谢谢", "讲个笑话", "天气", "吃饭", "随便聊聊"]
        if any(token in text for token in casual):
            return False
        return None

    def _llm_route(self, query: str) -> bool:
        try:
            import dashscope  # type: ignore
            from dashscope import Generation  # type: ignore
        except Exception:
            return False

        dashscope.api_key = self._dashscope_cfg.api_key
        if self._dashscope_cfg.base_http_api_url:
            dashscope.base_http_api_url = self._dashscope_cfg.base_http_api_url

        messages = [
            {
                "role": "system",
                "content": (
                    "你是一个检索路由器。判断用户问题是否需要调用专业精神卫生知识库。"
                    "只输出 JSON：{\"retrieve\": true/false, \"reason\": \"...\"}。"
                    "当问题涉及精神卫生、心理健康、WHO/mhGAP 指南、症状、评估、治疗、干预、药物、风险管理时返回 true。"
                ),
            },
            {"role": "user", "content": query},
        ]
        try:
            resp = Generation.call(  # type: ignore[misc]
                model=self._dashscope_cfg.model,
                messages=messages,
                result_format="message",
                temperature=0,
                top_p=0.1,
                max_tokens=80,
                stream=False,
            )
        except Exception:
            return False

        if getattr(resp, "status_code", None) != HTTPStatus.OK:
            return False

        try:
            content = resp.output.choices[0].message.content
            if isinstance(content, list):
                content = "".join(str(x.get("text", "")) if isinstance(x, dict) else str(x) for x in content)
            if not isinstance(content, str):
                return False
            match = re.search(r"\{.*\}", content, re.S)
            payload = json.loads(match.group(0) if match else content)
            return bool(payload.get("retrieve"))
        except Exception:
            return False

    def _embed_query(self, query: str) -> list[float]:
        try:
            import dashscope  # type: ignore
            from dashscope import TextEmbedding  # type: ignore
        except ImportError as exc:
            raise RagRetrievalError("dashscope 未安装。请先执行: pip install dashscope") from exc

        dashscope.api_key = self._dashscope_cfg.api_key
        if self._dashscope_cfg.base_http_api_url:
            dashscope.base_http_api_url = self._dashscope_cfg.base_http_api_url

        resp = TextEmbedding.call(
            model=self._rag_cfg.embedding_model,
            input=query,
            dimension=self._rag_cfg.embedding_dimension,
            text_type="query",
        )
        if resp.status_code != HTTPStatus.OK:
            raise RagRetrievalError(
                f"Embedding 调用失败: request_id={resp.request_id} code={resp.code} message={resp.message}"
            )
        return list(resp.output["embeddings"][0]["embedding"])

    def _rerank(self, query: str, candidates: list[RagHit]) -> list[RagHit]:
        try:
            import dashscope  # type: ignore
        except ImportError:
            # 没有 dashscope 时，退化为向量召回排序
            return candidates

        dashscope.api_key = self._dashscope_cfg.api_key
        if self._dashscope_cfg.base_http_api_url:
            dashscope.base_http_api_url = self._dashscope_cfg.base_http_api_url

        docs = [c.chunk.text for c in candidates]
        try:
            resp = dashscope.TextReRank.call(  # type: ignore[attr-defined]
                model=self._rag_cfg.rerank_model,
                query=query,
                documents=docs,
                top_n=min(len(docs), self._rag_cfg.top_k),
                return_documents=True,
                instruct="Given a web search query, retrieve relevant passages that answer the query.",
            )
        except Exception:
            return candidates

        if getattr(resp, "status_code", None) != HTTPStatus.OK:
            return candidates

        results = getattr(resp, "output", None)
        if isinstance(results, dict):
            items = results.get("results") or []
        else:
            items = []

        index_to_score: dict[int, float] = {}
        for it in items:
            try:
                index_to_score[int(it["index"])] = float(it["relevance_score"])
            except Exception:
                continue

        enumerated = list(enumerate(candidates))
        enumerated.sort(key=lambda p: index_to_score.get(p[0], p[1].score), reverse=True)
        return [h for _, h in enumerated]
