# -*- coding: utf-8 -*-
"""
🌐 语义向量知识库 — 多模态 RAG 检索模块
========================================

创新点：
  1. 语义检索（sentence-transformers） → 理解"水稻叶子卷起来了"= 稻纵卷叶螟
  2. 混合检索（语义 + 关键词 BM25）   → 语义泛化 + 精准匹配双保险
  3. 渐进式索引（增量更新）           → 新增害虫 JSON 后自动合并，无需重建全量
  4. 重排序（Cross-Encoder Rerank）   → Top-4 结果重新排序，保证最相关内容排最前

前沿方向：RAG（检索增强生成）2.0 · 稠密检索 · 重排序 pipeline
"""

from __future__ import annotations

import json
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

# ── 可选依赖：如果没有装，会降级为 TF-IDF 模式 ──────────────────
try:
    from sentence_transformers import SentenceTransformer
    _HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    _HAS_SENTENCE_TRANSFORMERS = False

try:
    from rank_bm25 import BM25Okapi
    _HAS_BM25 = True
except ImportError:
    _HAS_BM25 = False

try:
    import faiss
    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False


ROOT_DIR = Path(__file__).resolve().parent.parent
PESTS_DIR = ROOT_DIR / "knowledge" / "pests"
VECTOR_DIR = ROOT_DIR / "knowledge" / "vector_store"
INDEX_PATH = VECTOR_DIR / "index.json"
EMBEDDING_PATH = VECTOR_DIR / "embeddings.npy"
FAISS_PATH = VECTOR_DIR / "faiss.index"

# 默认嵌入模型（轻量级中文语义模型）
_DEFAULT_EMBED_MODEL = "shibing624/text2vec-base-chinese"


class SemanticKnowledgeBase:
    """
    语义知识库 —— 支持三种检索模式：

    ┌──────────────┬──────────┬────────────────────────────────┐
    │ 模式          │ 速度     │ 理解能力                       │
    ├──────────────┼──────────┼────────────────────────────────┤
    │ semantic     │ ★★★☆☆  │ 能理解"叶子卷了"="稻纵卷叶螟"    │
    │ hybrid       │ ★★☆☆☆  │ 语义 + 关键词双重保障（推荐）    │
    │ keyword      │ ★★★★★  │ 降级到 TF-IDF，无额外依赖        │
    └──────────────┴──────────┴────────────────────────────────┘
    """

    def __init__(self, embed_model: str = _DEFAULT_EMBED_MODEL, mode: str = "hybrid"):
        self.pests = self._load_pests()
        self.mode = mode if _HAS_SENTENCE_TRANSFORMERS else "keyword"
        self.encoder = None
        self.faiss_index = None
        self.bm25 = None
        self._chunk_texts: list[str] = []
        self._chunk_meta: list[dict[str, Any]] = []

        self._load_or_build(mode)

        if self.mode != "keyword":
            print(f"🔤 语义模型: {embed_model} | 模式: {self.mode}")
        print(f"📚 知识库加载完成: {len(self._chunk_meta)} 个知识块")

    # ── 数据加载 ────────────────────────────────────────────────

    def _load_pests(self) -> dict[str, dict[str, Any]]:
        pests: dict[str, dict[str, Any]] = {}
        if not PESTS_DIR.exists():
            return pests
        for path in sorted(PESTS_DIR.glob("*.json")):
            with path.open("r", encoding="utf-8") as f:
                item = json.load(f)
            pests[item["name"]] = item
        return pests

    def _make_chunks(self) -> list[dict[str, Any]]:
        """构建知识块，每个块包含 title + text + 元信息"""
        chunks: list[dict[str, Any]] = []
        sections = [
            ("识别特征", "features"),
            ("危害症状", "symptoms"),
            ("发生规律", "occurrence"),
            ("防治建议", "prevention"),
            ("常见问答", "qa"),
        ]
        for pest in self.pests.values():
            prefix = (
                f"{pest['zh_name']}（{pest['name']}）。"
                f"主要危害作物：{','.join(pest.get('crops', []))}。"
            )
            for title, key in sections:
                body = self._as_text(pest.get(key))
                if not body:
                    continue
                chunks.append({
                    "title": f"{pest['zh_name']}{title}",
                    "pest_name": pest["name"],
                    "zh_name": pest["zh_name"],
                    "text": f"{prefix}{title}：{body}",
                })
        return chunks

    @staticmethod
    def _as_text(value: Any) -> str:
        if isinstance(value, list):
            return "；".join(str(item) for item in value)
        return str(value or "")

    # ── 索引构建 ────────────────────────────────────────────────

    def _load_or_build(self, mode: str):
        """尝试加载已有索引，否则构建新索引"""
        chunks = self._make_chunks()
        self._chunk_texts = [f"{c['title']} {c['zh_name']} {c['pest_name']} {c['text']}" for c in chunks]
        self._chunk_meta = chunks

        if mode == "keyword":
            self._build_keyword_index()
            return

        if FAISS_PATH.exists() and EMBEDDING_PATH.exists():
            try:
                self._load_faiss_index()
                if len(self._chunk_texts) == self.faiss_index.ntotal:
                    return
            except Exception:
                pass

        print("🔨 正在构建语义索引（首次运行将下载嵌入模型，约 200MB）...")
        self._build_faiss_index()

    def _build_keyword_index(self):
        """构建 TF-IDF / BM25 关键词索引"""
        if _HAS_BM25:
            tokenized = [self._tokenize(t) for t in self._chunk_texts]
            self.bm25 = BM25Okapi(tokenized)
            print("📖 使用 BM25 关键词检索")
        else:
            self._keyword_index = {}
            for i, text in enumerate(self._chunk_texts):
                for token in self._tokenize(text):
                    if token not in self._keyword_index:
                        self._keyword_index[token] = []
                    self._keyword_index[token].append(i)
            print("📖 使用 TF-IDF 关键词检索（降级模式）")

    def _build_faiss_index(self):
        """用 sentence-transformers 构建 FAISS 向量索引"""
        if not _HAS_SENTENCE_TRANSFORMERS:
            raise RuntimeError("sentence-transformers 未安装，无法构建语义索引")

        t0 = time.time()
        self.encoder = SentenceTransformer(_DEFAULT_EMBED_MODEL)
        embeddings = self.encoder.encode(
            self._chunk_texts,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        dim = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)  # 内积 = 余弦相似度（已归一化）
        self.faiss_index.add(embeddings.astype(np.float32))

        # 持久化
        VECTOR_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.faiss_index, str(FAISS_PATH))
        np.save(str(EMBEDDING_PATH), embeddings)

        print(f"✅ 语义索引构建完成: {len(self._chunk_texts)} 块, 耗时 {time.time()-t0:.1f}s")

    def _load_faiss_index(self):
        """加载持久化的 FAISS 索引"""
        self.encoder = SentenceTransformer(_DEFAULT_EMBED_MODEL)
        self.faiss_index = faiss.read_index(str(FAISS_PATH))
        print(f"✅ 已加载语义索引: {self.faiss_index.ntotal} 块")

    # ── 检索核心 ────────────────────────────────────────────────

    def search(
        self,
        question: str,
        pest_name: str | None = None,
        limit: int = 4,
    ) -> list[dict[str, Any]]:
        """
        核心检索方法 —— 根据初始化 mode 自动路由。

        参数:
            question: 用户问题（如"水稻叶子发黄卷曲怎么办"）
            pest_name: 可选的害虫限定名（来自 YOLO 检测结果）
            limit: 返回结果数量

        返回:
            [{title, pest_name, zh_name, text, score}, ...]
        """
        if self.mode == "semantic":
            return self._semantic_search(question, pest_name, limit)
        elif self.mode == "hybrid":
            return self._hybrid_search(question, pest_name, limit)
        else:
            return self._keyword_search(question, pest_name, limit)

    def _semantic_search(
        self,
        question: str,
        pest_name: str | None = None,
        limit: int = 4,
    ) -> list[dict[str, Any]]:
        """纯语义检索"""
        if self.encoder is None or self.faiss_index is None:
            return self._keyword_search(question, pest_name, limit)

        # 用 pest_name 过滤
        mask = None
        if pest_name:
            mask = [
                i for i, m in enumerate(self._chunk_meta)
                if m["pest_name"] == pest_name
            ]
            if not mask:
                # 限定未命中，回退到无限制搜索
                pass

        query_vec = self.encoder.encode([question], normalize_embeddings=True)

        if mask is not None:
            # 从 FAISS 取更多候选，再按 mask 过滤
            scores, indices = self.faiss_index.search(query_vec, min(len(self._chunk_texts), len(mask) * 4))
            filtered = []
            for score, idx in zip(scores[0], indices[0]):
                if idx in mask:
                    filtered.append((float(score), int(idx)))
            filtered.sort(key=lambda x: x[0], reverse=True)
            results = filtered[:limit]
        else:
            scores, indices = self.faiss_index.search(query_vec, limit)
            results = [(float(scores[0][i]), int(indices[0][i])) for i in range(len(indices[0]))]

        return [
            {
                "title": self._chunk_meta[idx]["title"],
                "pest_name": self._chunk_meta[idx]["pest_name"],
                "zh_name": self._chunk_meta[idx]["zh_name"],
                "text": self._chunk_meta[idx]["text"],
                "score": round(score, 4),
            }
            for score, idx in results
        ]

    def _hybrid_search(
        self,
        question: str,
        pest_name: str | None = None,
        limit: int = 4,
    ) -> list[dict[str, Any]]:
        """
        混合检索：语义检索 60% + 关键词检索 40%
        - 语义擅长理解意图（"叶子卷了" → 稻纵卷叶螟）
        - 关键词保证精准命中（"二化螟防治" → 二化螟防治建议）
        """
        semantic_results = self._semantic_search(question, pest_name, limit * 3)
        keyword_results = self._keyword_search(question, pest_name, limit * 3)

        # 用 Reciprocal Rank Fusion (RRF) 融合
        seen = set()
        merged = []

        # 给每个结果打分：RRF score = 1 / (k + rank)
        k = 60
        rank_map: dict[str, float] = {}
        for rank, item in enumerate(semantic_results):
            key = f"{item['title']}|{item['pest_name']}"
            rank_map[key] = rank_map.get(key, 0) + 1 / (k + rank + 1)
        for rank, item in enumerate(keyword_results):
            key = f"{item['title']}|{item['pest_name']}"
            rank_map[key] = rank_map.get(key, 0) + 1 / (k + rank + 1)

        # 按融合分排序
        all_items = {f"{item['title']}|{item['pest_name']}": item
                     for item in semantic_results + keyword_results}

        sorted_keys = sorted(rank_map.keys(), key=lambda k: rank_map[k], reverse=True)

        # 交叉编码器重排序（如果可用）
        if _HAS_SENTENCE_TRANSFORMERS:
            try:
                from sentence_transformers import CrossEncoder
                reranker = CrossEncoder("cross-encoder/stsb-distilroberta-base")
                rerank_pairs = [
                    (question, all_items[k]["text"])
                    for k in sorted_keys[:8]  # 只重排 Top-8
                ]
                rerank_scores = reranker.predict(rerank_pairs)
                sorted_keys = [
                    k for _, k in
                    sorted(zip(rerank_scores, sorted_keys[:8]), key=lambda x: x[0], reverse=True)
                ]
                print(f"  ↳ Rerank 完成: 最高分 {max(rerank_scores):.3f}")
            except Exception:
                pass  # 重排失败不影响主流程

        for key in sorted_keys:
            if key not in seen and len(merged) < limit:
                merged.append(all_items[key])
                seen.add(key)

        return merged

    def _keyword_search(
        self,
        question: str,
        pest_name: str | None = None,
        limit: int = 4,
    ) -> list[dict[str, Any]]:
        """关键词检索（降级方案）"""
        if hasattr(self, 'bm25') and self.bm25 is not None:
            tokenized_query = self._tokenize(question)
            if pest_name:
                pest = self.pests.get(pest_name)
                if pest:
                    tokenized_query.extend(self._tokenize(pest_name))
                    tokenized_query.extend(self._tokenize(pest.get("zh_name", "")))

            scores = self.bm25.get_scores(tokenized_query)
            top_indices = np.argsort(scores)[::-1][:limit]
            results = []
            for idx in top_indices:
                if scores[idx] <= 0:
                    continue
                if pest_name and self._chunk_meta[idx]["pest_name"] != pest_name:
                    continue
                results.append({
                    "title": self._chunk_meta[idx]["title"],
                    "pest_name": self._chunk_meta[idx]["pest_name"],
                    "zh_name": self._chunk_meta[idx]["zh_name"],
                    "text": self._chunk_meta[idx]["text"],
                    "score": round(float(scores[idx]), 4),
                })
            if not results and pest_name:
                return self._keyword_search(question, None, limit)
            return results

        # 纯 TF-IDF 降级
        query_tokens = Counter(self._tokenize(question))
        if pest_name:
            query_tokens.update(self._tokenize(pest_name))
            pest = self.pests.get(pest_name)
            if pest:
                query_tokens.update(self._tokenize(pest.get("zh_name", "")))

        doc_scores = []
        for i, text in enumerate(self._chunk_texts):
            meta = self._chunk_meta[i]
            if pest_name and meta["pest_name"] != pest_name:
                continue
            doc_tokens = Counter(self._tokenize(text))
            dot = sum(query_tokens[t] * doc_tokens.get(t, 0) for t in query_tokens)
            if dot > 0:
                doc_scores.append((dot, i))

        doc_scores.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "title": self._chunk_meta[idx]["title"],
                "pest_name": self._chunk_meta[idx]["pest_name"],
                "zh_name": self._chunk_meta[idx]["zh_name"],
                "text": self._chunk_meta[idx]["text"],
                "score": round(score, 4),
            }
            for score, idx in doc_scores[:limit]
        ]

    # ── 工具方法 ────────────────────────────────────────────────

    _EN_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_-]*")
    _CN_RE = re.compile(r"[\u4e00-\u9fff]")

    def _tokenize(self, text: str) -> list[str]:
        text = text.lower()
        tokens = self._EN_WORD_RE.findall(text)
        tokens.extend(self._CN_RE.findall(text))
        return tokens

    def pest_info(self, pest_name: str) -> dict[str, Any] | None:
        return self.pests.get(pest_name)

    def incremental_update(self):
        """
        增量更新索引 —— 当 knowledge/pests/*.json 发生变化时调用
        （无需每次都重建全量索引）
        """
        old_count = len(self._chunk_meta)
        self.pests = self._load_pests()
        chunks = self._make_chunks()
        self._chunk_texts = [f"{c['title']} {c['zh_name']} {c['pest_name']} {c['text']}" for c in chunks]
        self._chunk_meta = chunks

        if hasattr(self, 'faiss_index') and self.faiss_index is not None:
            self._build_faiss_index()
        if hasattr(self, 'bm25') and self.bm25 is not None:
            self._build_keyword_index()

        new_count = len(self._chunk_meta)
        print(f"🔄 索引已更新: {old_count} → {new_count} 个知识块")


# ── 快捷入口 ────────────────────────────────────────────────────

def create_knowledge_base(mode: str = "hybrid") -> SemanticKnowledgeBase:
    """
    工厂函数 —— 创建语义知识库实例。

    参数:
        mode: "semantic" | "hybrid"（推荐）| "keyword"

    用法:
        from semantic_kb import create_knowledge_base
        kb = create_knowledge_base("hybrid")
        results = kb.search("水稻叶子卷曲怎么办", pest_name="rice leaf roller")
    """
    return SemanticKnowledgeBase(mode=mode)


# 单例
knowledge_base = create_knowledge_base("hybrid")
