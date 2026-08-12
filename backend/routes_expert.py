# -*- coding: utf-8 -*-
"""
专家路由 — /expert/*
====================
专家工作台（识别质量检验官）：
  - 查看用户使用的具体细节（检测原图 + 结果图 + 问答记录）
  - 对识别结果打标：识别成功 / 识别模糊（置信度低）/ 识别失败（无虫或未提取）
  - 形成模型质量反馈闭环，供整体问题定位与优化
仅 专家(expert) 与 管理员(admin) 可访问。
"""
from __future__ import annotations

import sys
from pathlib import Path

# 确保 backend 包可导入
if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.auth import require_expert
from backend.storage import (
    get_expert_detections, get_expert_stats, get_qa, mark_detection,
)

router = APIRouter()

# 标记选项说明（供前端展示）
MARK_LABELS = {
    "success": "识别成功",
    "ambiguous": "识别模糊（置信度低）",
    "failed": "识别失败（无虫或未提取）",
}


class MarkRequest(BaseModel):
    mark: str  # success / ambiguous / failed / ''（清空）
    note: str = ""


@router.get("/expert/overview")
def expert_overview(user: dict = Depends(require_expert)):
    """概览：识别记录总数 / 待审核数 / 各类标记分布（实时反馈）"""
    stats = get_expert_stats()
    return {"success": True, "current_user": user.get("username"), "stats": stats}


@router.get("/expert/records/detections")
def expert_detections(limit: int = 50, only_unmarked: bool = False,
                      user: dict = Depends(require_expert)):
    """专家视角的检测记录：原图 + 结果图 + 识别明细 + 标记状态"""
    try:
        records = get_expert_detections(limit, only_unmarked=only_unmarked)
        # 附加自动预判标记建议（模型侧，专家可确认/修改）
        for r in records:
            r["auto_suggest"] = _suggest_mark(r)
        return {"success": True, "records": records}
    except Exception as e:
        return {"success": False, "message": str(e), "records": []}


@router.get("/expert/records/qa")
def expert_qa(limit: int = 50, user: dict = Depends(require_expert)):
    """专家视角的问答记录：问题 + 回答 + 关联害虫"""
    try:
        return {"success": True, "records": get_qa(limit, username=None)}
    except Exception as e:
        return {"success": False, "message": str(e), "records": []}


@router.post("/expert/mark/{record_id}")
def expert_mark(record_id: int, body: MarkRequest,
                user: dict = Depends(require_expert)):
    """专家打标识别结果：success / ambiguous / failed；空串=清除标记"""
    if body.mark not in ("success", "ambiguous", "failed", ""):
        raise HTTPException(status_code=400, detail="非法的标记类型")
    if not mark_detection(record_id, body.mark, body.note.strip(), user.get("username", "")):
        raise HTTPException(status_code=404, detail="记录不存在")
    return {"success": True, "marked": body.mark, "by": user.get("username")}


def _suggest_mark(record: dict) -> str:
    """根据模型识别结果给出默认打标建议（供专家参考）"""
    if not record.get("is_unknown"):
        return "success"
    ut = record.get("unknown_type") or ""
    if ut == "no_detection":
        return "failed"
    return "ambiguous"
