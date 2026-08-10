# -*- coding: utf-8 -*-
"""
🌤 天气与虫害预警服务
====================
使用 Open-Meteo 免费 API（无需 Key），结合天气数据做虫害风险评估。
"""

from __future__ import annotations

from typing import Any

import requests

# ── WMO 天气代码 → 中文描述 ─────────────────────────────────────
WMO_CODES: dict[int, str] = {
    0: "☀️ 晴天",
    1: "🌤 少云",
    2: "⛅ 多云",
    3: "☁️ 阴天",
    45: "🌫 雾",
    48: "🌫 雾凇",
    51: "🌦 小毛毛雨",
    53: "🌦 毛毛雨",
    55: "🌧 大毛毛雨",
    56: "🌧 冻毛毛雨",
    57: "🌧 冻毛毛雨",
    61: "🌦 小雨",
    63: "🌧 中雨",
    65: "🌧 大雨",
    66: "🌧 冻雨",
    67: "🌧 冻雨",
    71: "🌨 小雪",
    73: "🌨 中雪",
    75: "❄️ 大雪",
    77: "❄️ 雪粒",
    80: "🌦 阵雨",
    81: "🌧 中阵雨",
    82: "🌧 大阵雨",
    85: "🌨 小阵雪",
    86: "❄️ 大阵雪",
    95: "⛈ 雷暴",
    96: "⛈ 雷暴加冰雹",
    99: "⛈ 雷暴加冰雹",
}

# ── 害虫适生条件（温度/湿度范围） ──────────────────────────────
PEST_CONDITIONS: dict[str, dict[str, Any]] = {
    "rice leaf roller": {
        "zh_name": "稻纵卷叶螟",
        "temp_min": 22, "temp_max": 32,
        "humidity_min": 75,
        "risk_high_temp": 28, "risk_high_humidity": 85,
    },
    "rice leaf caterpillar": {
        "zh_name": "稻毛虫",
        "temp_min": 20, "temp_max": 30,
        "humidity_min": 70,
        "risk_high_temp": 25, "risk_high_humidity": 80,
    },
    "paddy stem maggot": {
        "zh_name": "稻茎蛆",
        "temp_min": 18, "temp_max": 28,
        "humidity_min": 70,
        "risk_high_temp": 24, "risk_high_humidity": 80,
    },
    "asiatic rice borer": {
        "zh_name": "二化螟",
        "temp_min": 20, "temp_max": 33,
        "humidity_min": 70,
        "risk_high_temp": 26, "risk_high_humidity": 85,
    },
    "yellow rice borer": {
        "zh_name": "三化螟",
        "temp_min": 20, "temp_max": 32,
        "humidity_min": 70,
        "risk_high_temp": 25, "risk_high_humidity": 85,
    },
    "rice gall midge": {
        "zh_name": "稻瘿蚊",
        "temp_min": 22, "temp_max": 32,
        "humidity_min": 75,
        "risk_high_temp": 27, "risk_high_humidity": 85,
    },
    "Rice Stemfly": {
        "zh_name": "稻茎蝇",
        "temp_min": 18, "temp_max": 28,
        "humidity_min": 65,
        "risk_high_temp": 24, "risk_high_humidity": 80,
    },
    "brown plant hopper": {
        "zh_name": "褐飞虱",
        "temp_min": 22, "temp_max": 34,
        "humidity_min": 75,
        "risk_high_temp": 28, "risk_high_humidity": 85,
    },
    "white backed plant hopper": {
        "zh_name": "白背飞虱",
        "temp_min": 20, "temp_max": 32,
        "humidity_min": 70,
        "risk_high_temp": 26, "risk_high_humidity": 85,
    },
    "small brown plant hopper": {
        "zh_name": "灰飞虱",
        "temp_min": 18, "temp_max": 30,
        "humidity_min": 65,
        "risk_high_temp": 25, "risk_high_humidity": 80,
    },
    "rice water weevil": {
        "zh_name": "水稻象甲",
        "temp_min": 18, "temp_max": 28,
        "humidity_min": 60,
        "risk_high_temp": 24, "risk_high_humidity": 75,
    },
    "rice leafhopper": {
        "zh_name": "稻叶蝉",
        "temp_min": 20, "temp_max": 32,
        "humidity_min": 65,
        "risk_high_temp": 26, "risk_high_humidity": 80,
    },
    "grain spreader thrips": {
        "zh_name": "稻蓟马",
        "temp_min": 22, "temp_max": 33,
        "humidity_min": 60,
        "risk_high_temp": 28, "risk_high_humidity": 75,
    },
    "rice shell pest": {
        "zh_name": "稻螟蛉",
        "temp_min": 20, "temp_max": 30,
        "humidity_min": 70,
        "risk_high_temp": 26, "risk_high_humidity": 80,
    },
    "grub": {
        "zh_name": "蛴螬",
        "temp_min": 15, "temp_max": 28,
        "humidity_min": 55,
        "risk_high_temp": 22, "risk_high_humidity": 70,
    },
    "mole cricket": {
        "zh_name": "蝼蛄",
        "temp_min": 15, "temp_max": 30,
        "humidity_min": 55,
        "risk_high_temp": 22, "risk_high_humidity": 70,
    },
}

# ── 通用预警 ────────────────────────────────────────────────────
GENERAL_RISK_RULES = [
    ("持续降雨后高温高湿", lambda t, h, r: h > 80 and t > 25 and r > 5, "⚠️ 持续降雨后高温高湿，多种病虫害易高发，建议加强田间巡查"),
    ("连续高温干旱", lambda t, h, r: t > 33 and h < 50, "☀️ 连续高温干旱，注意飞虱、蓟马等耐旱害虫"),
    ("适温高湿", lambda t, h, r: 22 <= t <= 30 and h >= 70, "🌱 当前温湿度适宜害虫繁殖，请密切监测"),
]


def get_weather(lat: float, lng: float) -> dict[str, Any] | None:
    """
    获取指定坐标的实时天气。

    使用 Open-Meteo API（免费，无需 API Key）。
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lng,
        "current": "temperature_2m,relative_humidity_2m,weather_code,precipitation",
        "timezone": "auto",
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json().get("current", {})

        code = data.get("weather_code", 0)
        return {
            "temperature": data.get("temperature_2m"),
            "humidity": data.get("relative_humidity_2m"),
            "weather": WMO_CODES.get(code, f"🌤 {code}"),
            "weather_code": code,
            "precipitation": data.get("precipitation", 0),
        }
    except Exception as e:
        print(f"⚠️ 天气API请求失败: {e}")
        return None


def assess_pest_risk(
    pest_name: str | None,
    temperature: float,
    humidity: float,
    precipitation: float = 0,
) -> list[dict[str, str]]:
    """
    基于当前天气评估虫害风险。

    返回风险提示列表（可能有多条）。
    """
    alerts: list[dict[str, str]] = []
    pest = PEST_CONDITIONS.get(pest_name) if pest_name else None

    if pest:
        # 专项虫害风险评估
        in_temp_range = pest["temp_min"] <= temperature <= pest["temp_max"]
        in_humidity_range = humidity >= pest["humidity_min"]

        if in_temp_range and in_humidity_range:
            # 判断风险等级
            if temperature >= pest["risk_high_temp"] and humidity >= pest["risk_high_humidity"]:
                level = "🔴 高风险"
                detail = f"当前温湿度非常适宜{pest['zh_name']}爆发，建议立即检查田间并准备防治"
            elif temperature >= pest["risk_high_temp"] or humidity >= pest["risk_high_humidity"]:
                level = "🟡 中风险"
                detail = f"条件接近{pest['zh_name']}高发阈值，建议加强监测"
            else:
                level = "🟢 低风险"
                detail = f"当前条件一般，常规监测即可"
            alerts.append({"level": level, "detail": detail})
        elif in_temp_range and not in_humidity_range:
            alerts.append({"level": "🟢 低风险", "detail": f"温度适宜但湿度偏低，{pest['zh_name']}扩散风险不高"})
        else:
            alerts.append({"level": "🟢 低风险", "detail": f"当前温度不在{pest['zh_name']}适生范围内，风险较低"})

    # 通用预警
    for name, rule, msg in GENERAL_RISK_RULES:
        if rule(temperature, humidity, precipitation):
            alerts.append({"level": "💡 提醒", "detail": msg})

    return alerts if alerts else [{"level": "🟢 低风险", "detail": "当前天气条件对虫害无明显影响"}]


def _is_private_ip(ip: str) -> bool:
    """判断是否为私有/本地 IP"""
    return (
        ip.startswith("192.168.") or
        ip.startswith("10.") or
        ip.startswith("127.") or
        ip.startswith("::1") or
        ip == "localhost" or
        (ip.startswith("172.") and 16 <= int(ip.split(".")[1]) <= 31)
    )


def locate_by_ip(client_ip: str | None = None) -> dict[str, Any] | None:
    """
    通过客户端 IP 获取大致位置坐标（降级方案）。

    使用 ip-api.com（免费，无需 API Key，每分钟限制 45 次）。
    """
    if not client_ip or _is_private_ip(client_ip):
        # 私有 IP（局域网/WiFi）→ 使用默认坐标（中国中部）
        return {"lat": 30.5, "lng": 114.3, "city": "网络定位"}

    try:
        url = f"http://ip-api.com/json/{client_ip}?fields=status,lat,lon,city"
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") == "success":
            return {
                "lat": data["lat"],
                "lng": data["lon"],
                "city": data.get("city", ""),
            }
    except Exception:
        pass
    return None
