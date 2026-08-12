# -*- coding: utf-8 -*-
"""
认证路由 — /auth/*
==================
登录 / 注册 / 登出 / 当前用户信息（JWT 认证）。
"""
from __future__ import annotations

import sys
from pathlib import Path

# 确保 backend 包可导入
if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
import requests
import urllib.parse

from backend.auth import (
    WECHAT_APPID, WECHAT_SECRET, login, login_by_phone, login_by_wechat,
    register, require_login, send_sms_code, verify_sms_code,
)

router = APIRouter()


class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=2, max_length=20, description="用户名")
    password: str = Field(..., min_length=6, description="密码（至少 6 位）")
    display: str = Field("", max_length=20, description="昵称")


class SmsSendRequest(BaseModel):
    phone: str


class SmsLoginRequest(BaseModel):
    phone: str
    code: str


class WechatLoginRequest(BaseModel):
    code: str


@router.post("/auth/login")
def auth_login(payload: LoginRequest):
    """登录：返回 JWT 与角色"""
    result = login(payload.username.strip(), payload.password)
    if result is None:
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    token, user = result
    return {"token": token, **user}


@router.post("/auth/register")
def auth_register(payload: RegisterRequest):
    """注册普通用户"""
    user = register(payload.username.strip(), payload.password, payload.display.strip())
    if user is None:
        raise HTTPException(status_code=400, detail="注册失败：用户名已存在或密码过短（至少 6 位）")
    return {"success": True, **user}


@router.post("/auth/logout")
def auth_logout():
    """登出：JWT 无状态，前端清除本地 token 即可"""
    return {"success": True}


# ── 手机号验证码登录 ────────────────────────────────────────────
@router.post("/auth/sms/send")
def auth_sms_send(payload: SmsSendRequest):
    """发送短信验证码（含 60s 频率限制；未接短信服务时返回调试码便于测试）"""
    result = send_sms_code(payload.phone)
    if not result["ok"]:
        raise HTTPException(status_code=400, detail=result["msg"])
    resp = {"success": True, "msg": result["msg"]}
    if result.get("debug_code"):
        resp["debug_code"] = result["debug_code"]   # 仅调试模式返回，生产环境应移除
    return resp


@router.post("/auth/sms/login")
def auth_sms_login(payload: SmsLoginRequest):
    """手机号 + 验证码登录：校验通过即登录（无账号自动注册）"""
    if not verify_sms_code(payload.phone, payload.code):
        raise HTTPException(status_code=401, detail="验证码错误或已过期")
    result = login_by_phone(payload.phone)
    if result is None:
        raise HTTPException(status_code=500, detail="登录失败，请重试")
    token, user = result
    return {"token": token, **user}


# ── 微信登录（OAuth2 授权码模式） ───────────────────────────────
@router.get("/auth/wechat/auth-url")
def auth_wechat_url(redirect_uri: str = ""):
    """返回微信扫码授权链接（前端跳转用）；未配置 AppID 时降级提示"""
    if not WECHAT_APPID:
        raise HTTPException(
            status_code=503,
            detail="微信登录尚未配置：请在环境变量设置 WECHAT_APPID / WECHAT_SECRET",
        )
    redirect = redirect_uri or "http://127.0.0.1:8000/login"  # 生产应配置为公网回调地址
    url = (
        "https://open.weixin.qq.com/connect/qrconnect"
        f"?appid={WECHAT_APPID}"
        f"&redirect_uri={urllib.parse.quote(redirect)}"
        "&response_type=code&scope=snsapi_login&state=wechat_login#wechat_redirect"
    )
    return {"success": True, "url": url}


@router.post("/auth/wechat/login")
def auth_wechat_login(payload: WechatLoginRequest):
    """微信授权登录：用前端拿到的 code 换取 openid，再按 openid 登录/注册"""
    if not WECHAT_APPID or not WECHAT_SECRET:
        raise HTTPException(
            status_code=503,
            detail="微信登录尚未配置：请在环境变量设置 WECHAT_APPID / WECHAT_SECRET",
        )
    try:
        resp = requests.get(
            "https://api.weixin.qq.com/sns/oauth2/access_token",
            params={
                "appid": WECHAT_APPID,
                "secret": WECHAT_SECRET,
                "code": payload.code,
                "grant_type": "authorization_code",
            },
            timeout=10,
        ).json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"微信服务请求失败: {e}")
    if resp.get("errcode"):
        raise HTTPException(status_code=400, detail=f"微信授权失败: {resp.get('errmsg', '未知错误')}")
    result = login_by_wechat(resp["openid"])
    if result is None:
        raise HTTPException(status_code=500, detail="登录失败，请重试")
    token, user = result
    return {"token": token, **user}


@router.get("/auth/me")
def auth_me(user: dict = Depends(require_login)):
    """返回当前登录用户信息"""
    return user
