# -*- coding: utf-8 -*-
"""
管理路由 — /admin/*
===================
管理员专属接口：仪表盘概览、提交信息、用户管理（增删改查）、检测/问答记录、系统状态。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# 确保 backend 包可导入
if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from backend import config
from backend.auth import (
    create_user, delete_user, get_user_by_id, list_users,
    require_admin, update_user,
)
from backend.state import state
from backend.storage import (
    get_detections, get_expert_feedback_stats, get_hard_examples,
    get_qa, get_stats,
)

router = APIRouter()

# 项目根目录（Git 仓库根）
REPO_DIR = config._ROOT


# ── 请求体 ───────────────────────────────────────────────────
class AdminCreateUser(BaseModel):
    username: str = Field(min_length=2, max_length=20)
    password: str = Field(min_length=6, max_length=64)
    role: str = "user"
    display: str = ""


class AdminUpdateUser(BaseModel):
    display: str | None = None
    role: str | None = None
    password: str | None = None


@router.get("/admin/overview")
def admin_overview(user: dict = Depends(require_admin)):
    """仪表盘概览：用户/检测/问答统计 + 害虫频率 + 趋势 + 系统状态"""
    stats = get_stats()
    users = list_users()
    return {
        "success": True,
        "user_count": len(users),
        "total_detections": stats.get("total_detections", 0),
        "total_qa": stats.get("total_qa", 0),
        "unknown_count": stats.get("unknown_count", 0),
        "internal_detections": stats.get("internal_detections", 0),
        "internal_qa": stats.get("internal_qa", 0),
        "top_pests": stats.get("top_pests", []),
        "daily_trend": stats.get("daily_trend", []),
        "models_ready": state.models_ready,
        "enhanced": {
            "semantic_kb": state.semantic_kb is not None,
            "agent": state.agent is not None,
            "inference_engine": state.inference_engine is not None,
        },
    }


# ── 用户管理（增删改查） ──────────────────────────────────────
@router.get("/admin/users")
def admin_users(user: dict = Depends(require_admin)):
    """用户列表（不含密码哈希）"""
    return {"success": True, "users": list_users()}


@router.post("/admin/users")
def admin_create_user(body: AdminCreateUser, user: dict = Depends(require_admin)):
    """新增用户（管理员）"""
    created = create_user(body.username, body.password, body.role, body.display)
    if not created:
        raise HTTPException(status_code=400, detail="创建失败：用户名已存在或参数不合法（密码需≥6位）")
    return {"success": True, "user": created}


@router.put("/admin/users/{user_id}")
def admin_update_user(user_id: int, body: AdminUpdateUser,
                      user: dict = Depends(require_admin)):
    """更新用户：昵称 / 角色 / 密码"""
    target = get_user_by_id(user_id)
    if not target:
        raise HTTPException(status_code=404, detail="用户不存在")
    # 禁止通过接口把角色改到 admin 之外的最后一名 admin（保护）
    if body.role is not None and body.role != "admin" and target["role"] == "admin":
        admins = [u for u in list_users() if u["role"] == "admin"]
        if len(admins) <= 1:
            raise HTTPException(status_code=400, detail="系统至少保留一名管理员")
    if not update_user(user_id, display=body.display, role=body.role, password=body.password):
        raise HTTPException(status_code=400, detail="更新失败：参数不合法")
    return {"success": True, "user": get_user_by_id(user_id)}


@router.delete("/admin/users/{user_id}")
def admin_delete_user(user_id: int, user: dict = Depends(require_admin)):
    """删除用户（不能删除自己，不能删除最后一名管理员）"""
    target = get_user_by_id(user_id)
    if not target:
        raise HTTPException(status_code=404, detail="用户不存在")
    if target["username"] == user["username"]:
        raise HTTPException(status_code=400, detail="不能删除当前登录账号")
    if target["role"] == "admin":
        admins = [u for u in list_users() if u["role"] == "admin"]
        if len(admins) <= 1:
            raise HTTPException(status_code=400, detail="系统至少保留一名管理员")
    if not delete_user(user_id):
        raise HTTPException(status_code=500, detail="删除失败")
    return {"success": True, "deleted": user_id}


# ── 全量记录（管理员视角，不受用户隔离影响） ──────────────────
@router.get("/admin/records/detections")
def admin_records_detections(limit: int = 100, user: dict = Depends(require_admin)):
    """全部检测记录（管理员）"""
    return {"success": True, "records": get_detections(limit, username=None)}


@router.get("/admin/records/qa")
def admin_records_qa(limit: int = 100, user: dict = Depends(require_admin)):
    """全部问答记录（管理员）"""
    return {"success": True, "records": get_qa(limit, username=None)}


# ── 专家反馈闭环 ───────────────────────────────────────────────
@router.get("/admin/expert-feedback")
def admin_expert_feedback(user: dict = Depends(require_admin)):
    """专家反馈汇总：标记分布 / 薄弱害虫 TOP / 每日打标趋势（供管理台看大方向）"""
    try:
        return {"success": True, **get_expert_feedback_stats()}
    except Exception as e:
        return {"success": False, "message": str(e)}


@router.get("/admin/exports/hard-examples")
def admin_export_hard_examples(limit: int = 200, user: dict = Depends(require_admin)):
    """导出难例集：被标记为「识别失败」的原图+备注+预测，供模型补数据/难例挖掘"""
    try:
        return {"success": True, "total": 0, "records": get_hard_examples(limit)}
    except Exception as e:
        return {"success": False, "message": str(e), "records": []}


@router.get("/admin/commits")
def admin_commits(user: dict = Depends(require_admin), limit: int = 50):
    """获取系统提交信息（Git 提交历史），仅管理员可访问"""
    try:
        result = subprocess.run(
            ["git", "log", f"-n", str(limit),
             "--pretty=format:%h|%ad|%s", "--date=format:%Y-%m-%d %H:%M"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            cwd=str(REPO_DIR), timeout=5,
        )
        commits = []
        for line in (result.stdout or "").strip().splitlines():
            if "|" in line:
                hash_, date, subject = line.split("|", 2)
                commits.append({"hash": hash_, "date": date, "subject": subject})
        return {
            "success": True,
            "total": len(commits),
            "commits": commits,
            "current_user": user.get("username"),
        }
    except Exception as e:
        return {"success": False, "commits": [], "error": str(e)}
