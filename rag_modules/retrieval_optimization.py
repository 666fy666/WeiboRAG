"""检索优化模块。

封装向量检索与简易重排序策略。
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List

import faiss  # type: ignore
import numpy as np
from rank_bm25 import BM25Okapi

from .index_construction import BGESentenceEncoder

logger = logging.getLogger(__name__)

# 常量定义
TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]+|\w+")
LEXICAL_BOOST_BASE = 0.05
MIN_LEXICAL_COVERAGE = 1e-9


@dataclass
class RetrievalResult:
    """封装检索结果。"""

    chunk_id: str
    score: float
    text: str
    metadata: Dict[str, Any]
    raw_score: float = 0.0
    lexical_boost: float = 0.0
    bm25_score: float = 0.0
    vector_rank: int | None = None
    bm25_rank: int | None = None


class Retriever:
    """将向量检索与 BM25 检索融合，返回更稳健的结果列表。"""

    def __init__(
        self,
        index: faiss.Index,
        payloads: List[Dict[str, Any]],
        encoder: BGESentenceEncoder,
        top_k: int,
        rerank_top_k: int,
        bm25_top_k: int,
        rrf_k: float,
    ) -> None:
        self.index = index
        self.payloads = payloads
        self.encoder = encoder
        self.top_k = top_k
        self.rerank_top_k = max(rerank_top_k, top_k)
        self.bm25_top_k = max(bm25_top_k, top_k)
        self.rrf_k = max(rrf_k, 1.0)
        self.logger = logging.getLogger(f"{__name__}.Retriever")

        self._bm25_corpus_tokens = [
            self._tokenize_list(item.get("text", "")) for item in self.payloads
        ]
        self._bm25_index = self._build_bm25_index(self._bm25_corpus_tokens)

    def search(self, query: str) -> List[RetrievalResult]:
        """执行双路检索，并通过 RRF 融合返回结果。

        Args:
            query: 用户查询文本

        Returns:
            检索结果列表，按相关性排序

        Raises:
            ValueError: 如果查询为空
            RuntimeError: 如果编码失败
        """
        if not query or not query.strip():
            raise ValueError("查询语句不能为空")

        try:
            query_vec = self.encoder.embed([query])
        except Exception as exc:
            self.logger.error("编码查询失败: %s", exc)
            raise RuntimeError("查询编码失败") from exc

        if query_vec.size == 0:
            self.logger.warning("编码查询返回空向量，返回空结果")
            return []

        vector_hits = self._vector_search(query, query_vec)
        bm25_hits = self._bm25_search(query)
        fused_candidates = self._rrf_fusion(vector_hits, bm25_hits)

        results: List[RetrievalResult] = []
        for idx, fused_score, detail in fused_candidates[: self.top_k]:
            if idx < 0 or idx >= len(self.payloads):
                continue
            payload = self.payloads[idx]
            results.append(
                RetrievalResult(
                    chunk_id=str(payload.get("id", idx)),
                    score=float(fused_score),
                    text=payload.get("text", ""),
                    metadata=payload.get("metadata", {}),
                    raw_score=float(detail.get("raw_score", 0.0)),
                    lexical_boost=float(detail.get("lexical_boost", 0.0)),
                    bm25_score=float(detail.get("bm25_score", 0.0)),
                    vector_rank=detail.get("vector_rank"),
                    bm25_rank=detail.get("bm25_rank"),
                )
            )

        self.logger.debug(
            "向量候选 %d 条，BM25 候选 %d 条，融合后返回 %d 条",
            len(vector_hits),
            len(bm25_hits),
            len(results),
        )
        return results

    def _vector_search(self, query: str, query_vec: np.ndarray) -> List[Dict[str, Any]]:
        scores, indices = self.index.search(query_vec, self.rerank_top_k)
        hits: List[Dict[str, Any]] = []
        for raw_score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.payloads):
                continue
            payload = self.payloads[idx]
            text = payload.get("text", "")
            lexical = self._lexical_boost(query, text)
            combined = float(raw_score + lexical)
            hits.append(
                {
                    "index": int(idx),
                    "raw_score": float(raw_score),
                    "lexical_boost": float(lexical),
                    "combined_score": combined,
                }
            )
        hits.sort(key=lambda item: item["combined_score"], reverse=True)
        return hits

    def _bm25_search(self, query: str) -> List[Dict[str, Any]]:
        if self._bm25_index is None:
            return []

        query_tokens = self._tokenize_list(query)
        if not query_tokens:
            return []

        scores = self._bm25_index.get_scores(query_tokens)
        if scores.size == 0:
            return []

        top_n = min(self.bm25_top_k, scores.shape[0])
        candidate_indices = np.argsort(scores)[::-1][:top_n]

        hits: List[Dict[str, Any]] = []
        for idx in candidate_indices:
            score = float(scores[int(idx)])
            hits.append({"index": int(idx), "bm25_score": score})
        return hits

    def _rrf_fusion(
        self,
        vector_hits: List[Dict[str, Any]],
        bm25_hits: List[Dict[str, Any]],
    ) -> List[tuple[int, float, Dict[str, Any]]]:
        fused_scores: Dict[int, float] = {}
        details: Dict[int, Dict[str, Any]] = {}

        for rank, item in enumerate(vector_hits):
            idx = item["index"]
            fused_scores[idx] = fused_scores.get(idx, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            info = details.setdefault(idx, {})
            info.setdefault("raw_score", item.get("raw_score", 0.0))
            info.setdefault("lexical_boost", item.get("lexical_boost", 0.0))
            info.setdefault("vector_score", item.get("combined_score", 0.0))
            info["vector_rank"] = rank + 1

        for rank, item in enumerate(bm25_hits):
            idx = item["index"]
            fused_scores[idx] = fused_scores.get(idx, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            info = details.setdefault(idx, {})
            info.setdefault("bm25_score", item.get("bm25_score", 0.0))
            info["bm25_rank"] = rank + 1

        fused = sorted(fused_scores.items(), key=lambda pair: pair[1], reverse=True)
        return [(idx, score, details[idx]) for idx, score in fused]

    def _build_bm25_index(self, corpus_tokens: List[List[str]]) -> BM25Okapi | None:
        """构建 BM25 索引。

        Args:
            corpus_tokens: 语料 token 列表

        Returns:
            BM25 索引对象，如果无法构建则返回 None
        """
        if not corpus_tokens:
            self.logger.warning("语料为空，跳过构建 BM25 索引")
            return None

        if all(len(tokens) == 0 for tokens in corpus_tokens):
            self.logger.warning("所有文本块都为空，跳过构建 BM25 索引")
            return None

        try:
            return BM25Okapi(corpus_tokens)
        except Exception as exc:
            self.logger.error("构建 BM25 索引失败: %s", exc)
            return None

    def _lexical_boost(self, query: str, text: str) -> float:
        """基于字符级 token 重合率的得分修正。

        Args:
            query: 查询文本
            text: 候选文本

        Returns:
            词汇提升分数
        """
        if not query or not text:
            return 0.0

        query_tokens = self._tokenize_set(query)
        text_tokens = self._tokenize_set(text)
        if not query_tokens or not text_tokens:
            return 0.0

        overlaps = query_tokens.intersection(text_tokens)
        if not overlaps:
            return 0.0

        coverage = len(overlaps) / max(len(query_tokens), 1)
        if coverage < MIN_LEXICAL_COVERAGE:
            return 0.0

        return float(LEXICAL_BOOST_BASE * math.log1p(coverage * len(text_tokens)))

    @classmethod
    def _tokenize_set(cls, text: str) -> set[str]:
        return set(cls._tokenize_list(text))

    @staticmethod
    def _tokenize_list(text: str) -> List[str]:
        """将文本分词为 token 列表。

        Args:
            text: 输入文本

        Returns:
            token 列表（小写）
        """
        if not text:
            return []
        return [
            match.group(0).lower()
            for match in TOKEN_PATTERN.finditer(text)
            if match.group(0)
        ]


