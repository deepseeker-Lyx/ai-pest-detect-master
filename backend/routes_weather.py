# -*- coding: utf-8 -*-
"""
天气与虫害预警路由 — /weather/*
================================
实时天气、基于天气的害虫风险评估、IP 定位。
"""
from __future__ import annotations

import sys
from pathlib import Path

# 确保 backend 包可导入（兼容任意 cwd 启动）
if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import APIRouter, Request

from backend.weather_service import assess_pest_risk, get_weather, locate_by_ip

router = APIRouter()


@router.get("/weather/current")
async def weather_current(lat: float = 0, lng: float = 0):
    """获取指定坐标的实时天气"""
    if not lat or not lng:
        return {"error": "请提供 lat 和 lng 参数"}
    weather = get_weather(lat, lng)
    if weather is None:
        return {"error": "天气数据获取失败"}
    return weather


@router.get("/weather/pest-risk")
async def pest_risk(
    pest: str = "",
    lat: float = 0,
    lng: float = 0,
):
    """基于天气评估指定害虫的风险等级"""
    weather = get_weather(lat, lng) if lat and lng else None
    if weather is None:
        return {"error": "无法获取天气数据", "alerts": []}

    alerts = assess_pest_risk(
        pest_name=pest or None,
        temperature=weather["temperature"],
        humidity=weather["humidity"],
        precipitation=weather.get("precipitation", 0),
    )
    return {
        "weather": weather,
        "pest_name": pest,
        "alerts": alerts,
    }


@router.get("/weather/locate")
async def weather_locate(request: Request):
    """通过客户端 IP 获取位置并返回天气"""
    client_ip = request.client.host if request.client else None
    location = locate_by_ip(client_ip)
    if not location:
        return {"error": "无法定位"}

    weather = get_weather(location["lat"], location["lng"])
    return {
        "location": location,
        "weather": weather,
    }
