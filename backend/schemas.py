# -*- coding: utf-8 -*-
"""
Pydantic 数据模型 — API 请求/响应结构
"""
from __future__ import annotations

from pydantic import BaseModel, Field


# ── 检测 ─────────────────────────────────────────────────────────
class Detection(BaseModel):
    id:          int
    name:        str
    zh_name:     str = ""
    confidence:  float
    bbox:        list[int]   # [x1, y1, x2, y2]
    symptoms:    list[str] = Field(default_factory=list)
    prevention:  list[str] = Field(default_factory=list)
    alternatives: list[dict] = Field(default_factory=list)  # 不确定时列出多种可能


class DetectResponse(BaseModel):
    success:        bool
    result_image:   str        # URL path
    detections:     list[Detection]
    total:          int
    elapsed_ms:     float
    message:        str = ""
    # ⭐ 开放集识别字段
    is_unknown:     bool = False       # 是否疑似未知虫种
    unknown_type:   str = ""           # 未知类型：unknown_pest | low_confidence | ...
    unknown_suggestion: str = ""       # 给用户的提示文字


# ── 问答 ─────────────────────────────────────────────────────────
class QARequest(BaseModel):
    question:  str
    pest_name: str = ""
    pest_names: list[str] = Field(default_factory=list)  # 多害虫：图片含多种害虫时传入
    history:   list[dict[str, str]] = Field(default_factory=list)


class QASource(BaseModel):
    title:     str
    pest_name: str
    zh_name:   str = ""


class QAResponse(BaseModel):
    answer:    str
    sources:   list[QASource]
    used_llm:  bool = False
    message:   str = ""