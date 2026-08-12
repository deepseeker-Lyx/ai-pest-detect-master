# -*- coding: utf-8 -*-
"""
问答路由 — /qa/*
================
包含：知识库问答（普通 + 流式）、状态、常见问题。
支持单害虫（pest_name）与多害虫（pest_names）分别检索、分节回答。
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from pathlib import Path

# 确保 backend 包可导入（兼容任意 cwd 启动）
if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from backend.auth import require_login
from backend.knowledge_base import answer_with_llm, answer_with_llm_stream, knowledge_base
from backend.schemas import QARequest, QAResponse, QASource
from backend.storage import save_qa

router = APIRouter()


# ── 工具：同步生成器 → 异步生成器桥接 ────────────────────────────
async def _aiter_sync(gen):
    """
    将同步生成器转为异步生成器（在后台线程中迭代，不阻塞事件循环）。
    用于流式 LLM 响应等同步阻塞式生成器。
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue(maxsize=16)
    _END = object()

    def _producer():
        try:
            for item in gen:
                try:
                    loop.call_soon_threadsafe(queue.put_nowait, item)
                except Exception:
                    break
        except BaseException as e:
            try:
                loop.call_soon_threadsafe(queue.put_nowait, e)
            except Exception:
                pass
        finally:
            try:
                loop.call_soon_threadsafe(queue.put_nowait, _END)
            except Exception:
                pass

    threading.Thread(target=_producer, name="sync-gen-bridge", daemon=True).start()

    while True:
        item = await queue.get()
        if item is _END:
            break
        if isinstance(item, BaseException):
            raise item
        yield item


# ── 工具：多害虫解析与检索 ───────────────────────────────────────
def _resolve_pests(payload: QARequest) -> list[str]:
    """从请求解析去重后的害虫列表（兼容 pest_name 与 pest_names）"""
    pests = [p.strip() for p in (payload.pest_names or []) if p and p.strip()]
    if payload.pest_name and payload.pest_name.strip():
        pests.append(payload.pest_name.strip())
    seen: set[str] = set()
    result: list[str] = []
    for p in pests:
        if p not in seen:
            seen.add(p)
            result.append(p)
    return result


def _search_contexts(question: str, pests: list[str]) -> list[dict]:
    """多害虫时对每种害虫分别检索（每种 top-2），否则原逻辑 top-4"""
    if len(pests) > 1:
        contexts: list[dict] = []
        for pn in pests:
            contexts.extend(knowledge_base.search(question, pn, limit=2))
        return contexts
    return knowledge_base.search(question, pests[0] if pests else None, limit=4)


def _pest_label(pests: list[str]) -> str:
    """历史记录用的害虫标签"""
    return "、".join(pests) if pests else ""


# ── 路由 ─────────────────────────────────────────────────────────
@router.get("/qa/status")
async def qa_status():
    return {
        "status": "ok",
        "pest_count": len(knowledge_base.pests),
        "chunk_count": knowledge_base.index.get("chunk_count", 0),
        "llm_configured": all([
            os.getenv("LLM_API_KEY", "").strip(),
            os.getenv("LLM_BASE_URL", "").strip(),
            os.getenv("LLM_MODEL", "").strip(),
        ]),
    }


@router.get("/qa/common-questions")
async def get_common_questions():
    """返回通用问题列表"""
    return {
        "questions": [
            "水稻常见害虫有哪些？",
            "如何预防稻纵卷叶螟？",
            "害虫防治的最佳时期是什么时候？",
            "有机防治方法有哪些？",
            "如何区分不同类型的害虫？",
            "农药使用的注意事项有哪些？",
        ]
    }


@router.post("/qa/ask", response_model=QAResponse)
async def ask_knowledge_base(payload: QARequest, user: dict = Depends(require_login)):
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")

    pest_names = _resolve_pests(payload)
    contexts = _search_contexts(question, pest_names)
    if not contexts:
        return QAResponse(
            answer="知识库中没有检索到足够相关的资料，暂时无法给出可靠回答。请先补充该害虫的识别特征、危害症状和防治建议。",
            sources=[],
            used_llm=False,
            message="未检索到相关知识库内容",
        )

    try:
        # LLM 调用为网络请求（超时最长 60s），放入线程池避免阻塞事件循环
        result = await asyncio.to_thread(
            answer_with_llm, question, contexts, payload.history,
            pest_names[0] if len(pest_names) == 1 else "",
            pest_names if len(pest_names) > 1 else None,
        )
    except Exception as exc:
        result = {
            "answer": "云端大模型调用失败，已降级返回本地知识库摘要。\n\n" + "\n".join(
                f"- {item['title']}：{item['text']}" for item in contexts[:3]
            ),
            "used_llm": False,
            "message": f"大模型调用失败: {exc}",
        }

    seen = set()
    sources = []
    for item in contexts:
        key = (item["title"], item["pest_name"])
        if key in seen:
            continue
        seen.add(key)
        sources.append(QASource(
            title=item["title"],
            pest_name=item["pest_name"],
            zh_name=item.get("zh_name", ""),
        ))

    # 记录问答历史（SQLite）
    try:
        save_qa(question, result["answer"], _pest_label(pest_names), result.get("used_llm", False), username=user["username"], is_internal=(user.get("role", "user") != "user"))
    except Exception:
        pass

    return QAResponse(
        answer=result["answer"],
        sources=sources,
        used_llm=result.get("used_llm", False),
        message=result.get("message", ""),
    )


@router.post("/qa/ask-stream")
async def ask_knowledge_base_stream(payload: QARequest, user: dict = Depends(require_login)):
    """流式问答接口（需登录），支持打字机效果"""
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")

    pest_names = _resolve_pests(payload)

    async def generate():
        contexts = await asyncio.to_thread(_search_contexts, question, pest_names)

        if not contexts:
            yield f"data: {json.dumps({'text': '知识库中没有检索到相关资料，暂时无法给出可靠回答。', 'done': True}, ensure_ascii=False)}\n\n"
            return

        try:
            # 流式调用大模型（在后台线程迭代，不阻塞事件循环）
            _collected = []
            async for chunk in _aiter_sync(answer_with_llm_stream(
                question, contexts, payload.history,
                pest_names[0] if len(pest_names) == 1 else "",
                pest_names if len(pest_names) > 1 else None,
            )):
                _collected.append(chunk)
                yield f"data: {json.dumps({'text': chunk, 'done': False}, ensure_ascii=False)}\n\n"

            # 记录问答历史（SQLite）
            try:
                save_qa(question, "".join(_collected)[:2000], _pest_label(pest_names), True, username=user["username"], is_internal=(user.get("role", "user") != "user"))
            except Exception:
                pass

            # 发送资料来源
            sources = []
            seen = set()
            for item in contexts:
                key = (item["title"], item["pest_name"])
                if key in seen:
                    continue
                seen.add(key)
                sources.append({
                    "title": item["title"],
                    "pest_name": item["pest_name"],
                    "zh_name": item.get("zh_name", ""),
                })

            yield f"data: {json.dumps({'sources': sources, 'done': True}, ensure_ascii=False)}\n\n"

        except Exception as exc:
            error_msg = f"\n\n[错误：{str(exc)}]"
            yield f"data: {json.dumps({'text': error_msg, 'done': True}, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
