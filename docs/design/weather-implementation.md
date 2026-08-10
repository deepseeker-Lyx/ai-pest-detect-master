# 🌤 天气与虫害预警模块 — 实现笔记

> 日期：2026-07-29  
> 涉及文件：`backend/weather_service.py` · `backend/main.py` · `frontend/static/js/app.js` · `frontend/static/css/style.css`

---

## 📋 目录

1. [需求分析](#一需求分析)
2. [技术选型](#二技术选型)
3. [后端实现](#三后端实现)
4. [前端实现](#四前端实现)
5. [定位问题与解决方案](#五定位问题与解决方案)
6. [完整数据流](#六完整数据流)
7. [API 接口文档](#七-api-接口文档)
8. [常见问题](#八常见问题)

---

## 一、需求分析

### 用户需求
在害虫检测结果下方，展示基于用户位置的**实时天气**和**虫害风险预警**，帮助农户判断当前环境是否适合害虫爆发。

### 核心功能
```
用户上传图片检测
    │
    ├─ 获取用户位置（浏览器定位 / IP 降级）
    │
    ├─ ① 天气显示
    │   ├─ 天气状况（晴/雨/多云）
    │   ├─ 温度（°C）
    │   └─ 湿度（%）
    │
    └─ ② 虫害预警
        ├─ 基于当前温湿度评估该害虫风险等级
        └─ 通用天气预警（高温干旱/高湿适生等）
```

### 设计原则
- **免费**：不引入需要付费 API Key 的服务
- **零配置**：用户不需要申请任何东西，开箱即用
- **优雅降级**：定位失败时不报错，不显示即可
- **轻量紧凑**：手机端不占太多空间

---

## 二、技术选型

| 需求 | 选型 | 原因 |
|------|------|------|
| **天气数据** | [Open-Meteo](https://open-meteo.com/) | 免费、无需 API Key、支持坐标查询、数据准确 |
| **IP 定位** | [ip-api.com](http://ip-api.com/) | 免费、无需 API Key、返回经纬度 |
| **浏览器定位** | `navigator.geolocation` | 原生 API，精确度高 |
| **数据格式** | JSON | 前后端通用 |

### 为什么选 Open-Meteo？

| 对比项 | Open-Meteo | 和风天气 | OpenWeatherMap |
|--------|-----------|---------|---------------|
| API Key | ❌ 不需要 | ✅ 需要注册 | ✅ 需要注册 |
| 免费额度 | 无限制 | 1000次/天 | 60次/分钟 |
| 坐标查询 | ✅ 支持 | ✅ 支持 | ✅ 支持 |
| 国内速度 | ✅ 较快 | ✅ 快 | ❌ 慢 |

---

## 三、后端实现

### 文件结构
```
backend/weather_service.py    ← 核心逻辑（天气API + 风险评估）
backend/main.py               ← API 路由
```

### 3.1 天气数据获取

```python
def get_weather(lat: float, lng: float) -> dict | None:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lng,
        "current": "temperature_2m,relative_humidity_2m,weather_code,precipitation",
        "timezone": "auto",
    }
    resp = requests.get(url, params=params, timeout=10)
    data = resp.json().get("current", {})
    return {
        "temperature": data["temperature_2m"],
        "humidity": data["relative_humidity_2m"],
        "weather": WMO_CODES.get(data["weather_code"], "未知"),
        "precipitation": data.get("precipitation", 0),
    }
```

**关键点**：
- `current` 参数获取实时数据（非预报）
- `weather_code` 是 WMO 标准代码，需要映射为中文描述
- 设置 `timeout=10` 防止网络问题卡死

### 3.2 WMO 天气代码映射

Open-Meteo 返回的天气代码是国际标准，需要转为中文可读描述：

```python
WMO_CODES = {
    0: "☀️ 晴天",   1: "🌤 少云",   2: "⛅ 多云",   3: "☁️ 阴天",
    45: "🌫 雾",     51: "🌦 毛毛雨", 61: "🌦 小雨",   63: "🌧 中雨",
    65: "🌧 大雨",   71: "🌨 小雪",   75: "❄️ 大雪",
    80: "🌦 阵雨",   95: "⛈ 雷暴",
    # ... 共 20+ 种
}
```

### 3.3 虫害风险评估（规则引擎）

基于**昆虫生态学**原理，每种害虫都有其适宜的温度和湿度范围。

```python
PEST_CONDITIONS = {
    "rice leafhopper": {           # 稻叶蝉
        "zh_name": "稻叶蝉",
        "temp_min": 20, "temp_max": 32,      # 适温范围
        "humidity_min": 65,                   # 最低湿度要求
        "risk_high_temp": 26,                 # 高风险温度阈值
        "risk_high_humidity": 80,             # 高风险湿度阈值
    },
    "brown plant hopper": {        # 褐飞虱
        "zh_name": "褐飞虱",
        "temp_min": 22, "temp_max": 34,
        "humidity_min": 75,
        "risk_high_temp": 28,
        "risk_high_humidity": 85,
    },
    # ... 共 16 种害虫
}
```

**风险评估逻辑**：
```
当前温度在适生范围内 + 湿度达标
  ├─ 温度 >= 高风险阈值 AND 湿度 >= 高风险阈值 → 🔴 高风险
  ├─ 温度 >= 高风险阈值 OR  湿度 >= 高风险阈值 → 🟡 中风险
  └─ 其他                                         → 🟢 低风险

温度不在适生范围内 → 🟢 低风险（条件不适宜）
```

### 3.4 通用天气预警

不依赖特定害虫，基于天气条件给出通用提醒：

```python
GENERAL_RISK_RULES = [
    ("持续降雨后高温高湿",
     lambda t, h, r: h > 80 and t > 25 and r > 5,
     "⚠️ 持续降雨后高温高湿，多种病虫害易高发"),
    ("连续高温干旱",
     lambda t, h, r: t > 33 and h < 50,
     "☀️ 连续高温干旱，注意飞虱、蓟马等耐旱害虫"),
    ("适温高湿",
     lambda t, h, r: 22 <= t <= 30 and h >= 70,
     "🌱 当前温湿度适宜害虫繁殖，请密切监测"),
]
```

---

## 四、前端实现

### 4.1 定位流程（双重降级）

```
开始
  │
  ├─ 浏览器支持 Geolocation？
  │     ├─ 是 → 请求定位
  │     │        ├─ 成功 → 用精确坐标获取天气
  │     │        └─ 失败 → 走 IP 定位
  │     └─ 否 → 走 IP 定位
  │
  └─ IP 定位
         ├─ 成功 → 用 IP 坐标获取天气
         └─ 失败 → 不显示（静默）
```

```javascript
// 前端定位核心逻辑
function fetchWeatherAndPestRisk(detectData) {
  if (navigator.geolocation) {
    navigator.geolocation.getCurrentPosition(
      (pos) => fetchByCoord(pos.coords.latitude, pos.coords.longitude, detectData),
      () => fetchByIP(detectData),          // ← 浏览器定位失败，降级
      { timeout: 5000 }
    );
  } else {
    fetchByIP(detectData);                   // ← 不支持定位，降级
  }
}
```

### 4.2 后端 IP 定位（私有 IP 处理）

```
手机请求 → 后端收到 client IP = 192.168.114.156
    │
    ├─ 是私有 IP（192.168.x.x / 10.x.x.x / 172.16-31.x.x）？
    │     └─ 是 → 返回默认坐标（中国中部 30.5, 114.3）
    │
    └─ 是公网 IP？
          └─ 调用 ip-api.com 查询真实位置
```

**关键代码**：
```python
def _is_private_ip(ip: str) -> bool:
    return (
        ip.startswith("192.168.") or
        ip.startswith("10.") or
        ip.startswith("127.") or
        ip == "localhost" or
        (ip.startswith("172.") and 16 <= int(ip.split(".")[1]) <= 31)
    )
```

### 4.3 信息栏渲染（紧凑卡片）

检测结果卡片下方追加一个轻量信息栏：

```
┌─────────────────────────────────────┐
│ 🌤 ⛅ 多云 33.2°C  💧 60%          │ ← 天气（一行）
│ 🟢 当前温度不在稻叶蝉适生范围内...   │ ← 预警（最多2行）
│ 📍 基于您的位置 · 数据仅供参考      │ ← 脚注
└─────────────────────────────────────┘
```

**CSS 设计要点**：
- `flex` 布局，图标固定宽度 20px
- 文字自动换行（`word-break: break-word`）
- 预警行字体稍小（12px），与天气行区分
- 超过 2 条预警可点击"查看更多"展开

---

## 五、定位问题与解决方案

### 问题 1：手机端不显示天气

| 现象 | 原因 | 解决 |
|------|------|------|
| 手机访问 HTTP 页面，定位不弹窗 | 浏览器 Geolocation API 在 HTTP 下不工作（需要 HTTPS） | 添加 IP 定位降级方案 |
| `/weather/locate` 返回 `"error": "无法定位"` | 手机 IP `192.168.114.x` 是私有地址，ip-api.com 查不到 | 识别私有 IP 返回默认坐标 |

### 问题 2：调试经验

```
使用命令行测试 API（不需打开浏览器）：
$ python -c "import requests; r=requests.get('http://localhost:8000/weather/current?lat=30&lng=120'); print(r.json())"
{'temperature': 33.2, 'humidity': 60, 'weather': '⛅ 多云'}
```

---

## 六、完整数据流

```
手机浏览器                               后端 FastAPI                    Open-Meteo
    │                                       │                              │
    ├─ POST /detect/image ─────────────────→│                              │
    │                                       │  YOLO 检测                   │
    │←──────────────────── 检测结果 ────────┤                              │
    │                                       │                              │
    ├─ GET /weather/locate ───────────────→│                              │
    │                                       ├─ client IP = 192.168.x.x    │
    │                                       ├─ 私有IP → 默认坐标          │
    │                                       ├─ GET /forecast?lat=30&lng=114
    │                                       │  ←──────────────────────────┤
    │                                       │  ←── 天气数据 ─────────────┤
    │←──────── {weather, location} ────────┤                              │
    │                                       │                              │
    ├─ GET /weather/pest-risk?pest=... ───→│                              │
    │                                       ├─ 规则引擎评估风险           │
    │←─────────── {alerts} ────────────────┤                              │
    │                                       │                              │
    └─ renderWeatherBar() ─── 显示信息栏    │                              │
```

---

## 七、API 接口文档

| 方法 | 路径 | 参数 | 返回 |
|------|------|------|------|
| GET | `/weather/current` | `lat`, `lng` | `{temperature, humidity, weather, precipitation}` |
| GET | `/weather/pest-risk` | `pest`, `lat`, `lng` | `{weather, pest_name, alerts[]}` |
| GET | `/weather/locate` | 无（自动获取客户端 IP） | `{location: {lat, lng, city}, weather}` |

### 示例响应

**GET /weather/current?lat=30&lng=120**
```json
{
  "temperature": 33.2,
  "humidity": 60,
  "weather": "⛅ 多云",
  "weather_code": 2,
  "precipitation": 0.0
}
```

**GET /weather/pest-risk?pest=rice+leafhopper&lat=30&lng=120**
```json
{
  "weather": { "temperature": 33.2, "humidity": 60, "weather": "⛅ 多云" },
  "pest_name": "rice leafhopper",
  "alerts": [
    { "level": "🟢 低风险", "detail": "当前温度不在稻叶蝉适生范围内，风险较低" }
  ]
}
```

---

## 八、常见问题

### Q1：天气数据更新频率？
Open-Meteo 实时数据每 **15 分钟**更新一次，每次检测都会重新请求，保证数据新鲜。

### Q2：IP 定位准确吗？
公网 IP 定位精度一般在**城市级别**（10-50km），对天气来说完全够用。局域网环境下使用默认坐标。

### Q3：ip-api.com 有速率限制吗？
免费版每分钟 45 次请求，对个人项目完全够用。如果超限会自动回退到默认坐标。

### Q4：用户拒绝定位会怎样？
不会报错，只是不显示天气信息栏（静默失败）。体验上完全无影响。

### Q5：能离线使用吗？
不能，天气和 IP 定位都需要网络。但**害虫检测本身**不依赖网络（YOLO 模型在本地运行）。

---

## 📌 经验总结

> **做功能时先想好"用户如果定位失败怎么办"**——好的体验不是功能有多强，而是出问题时不会让用户困惑。三层降级（浏览器定位 → IP 定位 → 不显示）确保了零报错。
