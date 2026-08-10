# -*- coding: utf-8 -*-
"""
增强模块路由 — /enhanced/*
==========================
语义知识库 / AI Agent / 推理引擎 的状态与性能查询。
"""
from __future__ import annotations

import sys
from pathlib import Path

# 确保 backend 包可导入（兼容任意 cwd 启动）
if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import APIRouter

from backend.state import state

router = APIRouter()


@router.get("/enhanced/status")
async def enhanced_status():
    """查看增强模块状态"""
    return {
        "semantic_kb": state.semantic_kb is not None,
        "agent": state.agent is not None,
        "inference_engine": state.inference_engine is not None,
        "cache_stats": state.inference_engine.cache.stats() if state.inference_engine else None,
        "perf_report": state.inference_engine.get_perf_report() if state.inference_engine else None,
    }


@router.get("/enhanced/perf")
async def enhanced_perf():
    """查看推理引擎性能报告"""
    if state.inference_engine is None:
        return {"status": "unavailable", "message": "推理引擎未加载"}
    return state.inference_engine.get_perf_report()


@router.get("/enhanced/cache/clear")
async def enhanced_cache_clear():
    """清空推理缓存"""
    if state.inference_engine is None:
        return {"status": "unavailable"}
    state.inference_engine.cache.clear()
    return {"status": "ok", "message": "缓存已清空"}
