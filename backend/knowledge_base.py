# -*- coding: utf-8 -*-
"""
Local RAG-style knowledge base for pest Q&A.

The retrieval layer is intentionally dependency-light for the prototype:
it builds a persisted token index from curated JSON documents under
knowledge/pests/. A future ChromaDB implementation can replace this module
without changing the /qa/ask API.
"""
from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

import requests


ROOT_DIR = Path(__file__).resolve().parent.parent
PESTS_DIR = ROOT_DIR / "knowledge" / "pests"
VECTOR_DIR = ROOT_DIR / "knowledge" / "vector_store"
INDEX_PATH = VECTOR_DIR / "index.json"


_EN_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_-]*")
_CN_RE = re.compile(r"[\u4e00-\u9fff]")


def _tokenize(text: str) -> list[str]:
    text = text.lower()
    tokens = _EN_WORD_RE.findall(text)
    tokens.extend(_CN_RE.findall(text))
    return tokens


def _as_text(value: Any) -> str:
    if isinstance(value, list):
        return "；".join(str(item) for item in value)
    return str(value or "")


class KnowledgeBase:
    def __init__(self) -> None:
        self.pests = self._load_pests()
        self.index = self._load_or_build_index()

    def _load_pests(self) -> dict[str, dict[str, Any]]:
        pests: dict[str, dict[str, Any]] = {}
        if not PESTS_DIR.exists():
            return pests

        for path in sorted(PESTS_DIR.glob("*.json")):
            with path.open("r", encoding="utf-8") as f:
                item = json.load(f)
            pest_name = item["name"]
            pests[pest_name] = item
        return pests

    def _make_chunks(self) -> list[dict[str, Any]]:
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
                f"主要危害作物：{_as_text(pest.get('crops'))}。"
            )
            for title, key in sections:
                body = _as_text(pest.get(key))
                if not body:
                    continue
                chunks.append({
                    "title": f"{pest['zh_name']}{title}",
                    "pest_name": pest["name"],
                    "zh_name": pest["zh_name"],
                    "text": f"{prefix}{title}：{body}",
                })
        return chunks

    def _build_index(self) -> dict[str, Any]:
        chunks = self._make_chunks()
        indexed_chunks = []
        for chunk in chunks:
            tokens = Counter(_tokenize(" ".join([
                chunk["title"],
                chunk["pest_name"],
                chunk["zh_name"],
                chunk["text"],
            ])))
            norm = math.sqrt(sum(count * count for count in tokens.values())) or 1.0
            indexed_chunks.append({
                **chunk,
                "tokens": dict(tokens),
                "norm": norm,
            })

        index = {
            "version": 1,
            "chunk_count": len(indexed_chunks),
            "chunks": indexed_chunks,
        }
        VECTOR_DIR.mkdir(parents=True, exist_ok=True)
        with INDEX_PATH.open("w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)
        return index

    def _load_or_build_index(self) -> dict[str, Any]:
        if INDEX_PATH.exists():
            with INDEX_PATH.open("r", encoding="utf-8") as f:
                return json.load(f)
        return self._build_index()

    def pest_info(self, pest_name: str) -> dict[str, Any] | None:
        return self.pests.get(pest_name)

    def search(self, question: str, pest_name: str | None = None, limit: int = 4) -> list[dict[str, Any]]:
        if not self.index.get("chunks"):
            return []

        query_tokens = Counter(_tokenize(question))
        if pest_name:
            pest = self.pests.get(pest_name)
            query_tokens.update(_tokenize(pest_name))
            if pest:
                query_tokens.update(_tokenize(pest.get("zh_name", "")))

        query_norm = math.sqrt(sum(count * count for count in query_tokens.values())) or 1.0
        scored = []
        for chunk in self.index["chunks"]:
            if pest_name and chunk["pest_name"] != pest_name:
                continue
            chunk_tokens = chunk.get("tokens", {})
            dot = sum(query_tokens[token] * chunk_tokens.get(token, 0) for token in query_tokens)
            score = dot / (query_norm * (chunk.get("norm") or 1.0))
            title = chunk.get("title", "")
            if any(word in question for word in ["防治", "治理", "用药", "药剂", "方法", "怎么办"]):
                if "防治建议" in title:
                    score += 0.45
            if any(word in question for word in ["危害", "症状", "影响", "哪里"]):
                if "危害症状" in title:
                    score += 0.35
            if any(word in question for word in ["识别", "判断", "特征", "是不是"]):
                if "识别特征" in title:
                    score += 0.35
            if pest_name and chunk["pest_name"] == pest_name:
                score += 0.25
            if score > 0:
                scored.append((score, chunk))

        if not scored and pest_name:
            return self.search(question, None, limit)

        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            {
                "title": chunk["title"],
                "pest_name": chunk["pest_name"],
                "zh_name": chunk["zh_name"],
                "text": chunk["text"],
                "score": round(score, 4),
            }
            for score, chunk in scored[:limit]
        ]


def _fallback_answer(question: str, contexts: list[dict[str, Any]]) -> str:
    if not contexts:
        return "知识库中没有检索到足够相关的资料，暂时无法给出可靠回答。建议补充该害虫的识别特征、危害症状和防治资料。"

    lines = ["当前未配置云端大模型，以下为基于本地知识库检索到的资料摘要："]
    for item in contexts[:3]:
        lines.append(f"- {item['title']}：{item['text']}")
    lines.append("如需生成更自然的问答回复，请配置 LLM_API_KEY、LLM_BASE_URL 和 LLM_MODEL。")
    return "\n".join(lines)


def _response_text(data: dict[str, Any]) -> str:
    if data.get("output_text"):
        return str(data["output_text"]).strip()

    parts: list[str] = []
    for item in data.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"} and content.get("text"):
                parts.append(str(content["text"]))
    return "\n".join(parts).strip()


def _call_responses_api(
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    messages: list[dict[str, str]],
) -> str:
    input_messages: list[dict[str, Any]] = [{
        "role": "system",
        "content": [{"type": "input_text", "text": system_prompt}],
    }]
    for msg in messages:
        input_messages.append({
            "role": msg["role"],
            "content": [{"type": "input_text", "text": msg["content"]}],
        })

    payload: dict[str, Any] = {
        "model": model,
        "input": input_messages,
        "store": False,
    }

    reasoning_effort = os.getenv("LLM_REASONING_EFFORT", "").strip()
    if reasoning_effort:
        payload["reasoning"] = {"effort": reasoning_effort}

    url = base_url if os.getenv("LLM_FULL_URL", "").strip().lower() in {"1", "true", "yes"} else f"{base_url}/responses"
    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    answer = _response_text(response.json())
    if not answer:
        raise ValueError("Responses API 未返回可解析的文本内容")
    return answer


def _call_chat_completions_api(
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    messages: list[dict[str, str]],
) -> str:
    url = base_url if os.getenv("LLM_FULL_URL", "").strip().lower() in {"1", "true", "yes"} else f"{base_url}/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "system", "content": system_prompt}, *messages],
        "temperature": 0.2,
    }
    thinking = os.getenv("LLM_THINKING", "").strip().lower()
    if thinking in {"enabled", "disabled"}:
        payload["thinking"] = {"type": thinking}
    reasoning_effort = os.getenv("LLM_REASONING_EFFORT", "").strip()
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort

    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=45,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def answer_with_llm(question: str, contexts: list[dict[str, Any]],
                    history: list[dict[str, str]] | None = None,
                    pest_name: str = "", pest_names: list[str] | None = None) -> dict[str, Any]:
    api_key = os.getenv("LLM_API_KEY", "").strip()
    base_url = os.getenv("LLM_BASE_URL", "").strip().rstrip("/")
    model = os.getenv("LLM_MODEL", "").strip()
    wire_api = os.getenv("LLM_WIRE_API", "chat").strip().lower()

    if not api_key or not base_url or not model:
        return {
            "answer": _fallback_answer(question, contexts),
            "used_llm": False,
            "message": "未配置云端大模型环境变量，已返回本地知识库摘要。",
        }

    context_text = "\n\n".join(
        f"[{i + 1}] {item['title']}（{item['pest_name']}）\n{item['text']}"
        for i, item in enumerate(contexts)
    )
    if not context_text:
        context_text = "未检索到相关知识库内容。"

    system_prompt = build_enhanced_system_prompt(pest_name, pest_names)
    messages: list[dict[str, str]] = []
    for msg in (history or [])[-6:]:
        role = msg.get("role")
        content = msg.get("content")
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content})
    messages.append({
        "role": "user",
        "content": f"知识库内容：\n{context_text}\n\n用户问题：{question}",
    })

    if wire_api == "responses":
        answer = _call_responses_api(base_url, api_key, model, system_prompt, messages)
    else:
        answer = _call_chat_completions_api(base_url, api_key, model, system_prompt, messages)
    return {"answer": answer, "used_llm": True, "message": ""}


def build_enhanced_system_prompt(pest_name: str = "", pest_names: list[str] | None = None) -> str:
    """构建增强的系统提示词。pest_names 多于 1 个时，要求分害虫分别回答。"""
    prompt = """你是专业的农业害虫识别与防治助手，基于 YOLOv11 深度学习模型提供技术支持。

## 回答要求
1. 必须基于给定的知识库内容回答，不要编造信息
2. 针对具体害虫给出精准的防治建议
3. 回答结构完整，包含以下层次：
   - 🔍 问题分析：先理解用户问题，说明问题的关键点
   - 📋 详细解答：分点详述，每个要点有具体说明
   - 🎯 实操建议：给出农户可立即执行的行动方案
   - ⚠️ 注意事项：安全用药提示、易混淆点等
4. 涉及防治措施时，按优先级推荐综合防治方案：
   - 🌱 农业防治（轮作、清除病残体、合理密植等）
   - 🐞 生物防治（天敌昆虫、微生物农药、植物源农药）
   - 🧪 化学防治（必要时使用，注明用药种类、时机和安全间隔期）
5. 提及农药时必须包含安全用药提示
6. 语言亲切自然、通俗易懂，面向农户和农技人员
7. 如果用户问的是症状判断，尽可能列出该害虫与其他类似害虫的区分要点
8. 如果知识库资料不足，明确说明不确定，不要猜测
9. 适当使用emoji和分段标题让回答更易读
"""

    if pest_names and len(pest_names) > 1:
        names = "、".join(pest_names)
        prompt += (
            f"\n\n当前图片中检测到多种害虫：{names}。\n"
            "⚠️ 请**分别**针对每一种害虫给出防治方法，"
            "每种害虫用清晰的编号标题分隔（如 1️⃣ 稻纵卷叶螟、2️⃣ 褐飞虱），"
            "并分别包含：危害特点、防治建议、安全用药提示。"
            "不要只回答第一种，必须覆盖全部检测到的害虫。"
        )
    elif pest_name:
        prompt += f"\n\n当前识别的害虫：{pest_name}\n请针对该害虫提供具体的防治指导。"

    return prompt


def answer_with_llm_stream(question: str, contexts: list[dict[str, Any]],
                           history: list[dict[str, str]] | None = None,
                           pest_name: str = "", pest_names: list[str] | None = None):
    """流式调用大模型（生成器函数）"""
    api_key = os.getenv("LLM_API_KEY", "").strip()
    base_url = os.getenv("LLM_BASE_URL", "").strip().rstrip("/")
    model = os.getenv("LLM_MODEL", "").strip()
    wire_api = os.getenv("LLM_WIRE_API", "chat").strip().lower()

    if not api_key or not base_url or not model:
        # 未配置大模型，返回本地知识库摘要（逐字输出模拟流式）
        fallback_text = _fallback_answer(question, contexts)
        for char in fallback_text:
            yield char
        return

    context_text = "\n\n".join(
        f"[{i + 1}] {item['title']}（{item['pest_name']}）\n{item['text']}"
        for i, item in enumerate(contexts)
    )
    if not context_text:
        context_text = "未检索到相关知识库内容。"

    system_prompt = build_enhanced_system_prompt(pest_name, pest_names)
    messages: list[dict[str, str]] = []
    for msg in (history or [])[-6:]:
        role = msg.get("role")
        content = msg.get("content")
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content})
    messages.append({
        "role": "user",
        "content": f"知识库内容：\n{context_text}\n\n用户问题：{question}",
    })

    # 流式调用
    url = base_url if os.getenv("LLM_FULL_URL", "").strip().lower() in {"1", "true", "yes"} else f"{base_url}/chat/completions"

    try:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "system", "content": system_prompt}, *messages],
            "temperature": 0.2,
            "stream": True,
        }
        thinking = os.getenv("LLM_THINKING", "").strip().lower()
        if thinking in {"enabled", "disabled"}:
            payload["thinking"] = {"type": thinking}
        reasoning_effort = os.getenv("LLM_REASONING_EFFORT", "").strip()
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort

        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            stream=True,
            timeout=60,
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if not line:
                continue

            line_text = line.decode('utf-8')
            if not line_text.startswith('data: '):
                continue

            data_str = line_text[6:]  # 移除 "data: " 前缀

            if data_str.strip() == '[DONE]':
                break

            try:
                data = json.loads(data_str)
                choices = data.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    yield content
            except json.JSONDecodeError:
                continue
            except (KeyError, IndexError) as e:
                continue

    except Exception as e:
        error_msg = f"\n\n[错误：大模型调用失败 - {str(e)}]\n\n使用本地知识库回答：\n"
        for char in error_msg:
            yield char
        # 降级到本地知识库
        fallback_text = _fallback_answer(question, contexts)
        for char in fallback_text:
            yield char


knowledge_base = KnowledgeBase()
