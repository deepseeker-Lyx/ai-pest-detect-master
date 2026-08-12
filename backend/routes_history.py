# -*- coding: utf-8 -*-
"""
历史记录与统计路由 — /history/*
===============================
检测历史、问答历史、使用统计（SQLite）。
"""
from __future__ import annotations

import sys
from pathlib import Path

# 确保 backend 包可导入（兼容任意 cwd 启动）
if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import APIRouter, Depends

from backend.auth import require_login
from backend.storage import get_detections, get_qa, get_stats

router = APIRouter()


@router.get("/history/detections")
async def history_detections(limit: int = 20, user: dict = Depends(require_login)):
    """检测历史记录（按当前用户隔离，需登录）"""
    try:
        return {"success": True, "records": get_detections(limit, username=user["username"])}
    except Exception as e:
        return {"success": False, "message": str(e), "records": []}


@router.get("/history/qa")
async def history_qa(limit: int = 20, user: dict = Depends(require_login)):
    """问答历史记录（按当前用户隔离，需登录）"""
    try:
        return {"success": True, "records": get_qa(limit, username=user["username"])}
    except Exception as e:
        return {"success": False, "message": str(e), "records": []}


@router.get("/history/stats")
async def history_stats(user: dict = Depends(require_login)):
    """使用统计（按当前用户隔离，需登录）"""
    try:
        return {"success": True, **get_stats(username=user["username"])}
    except Exception as e:
        return {"success": False, "message": str(e)}
