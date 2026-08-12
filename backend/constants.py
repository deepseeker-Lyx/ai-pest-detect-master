# -*- coding: utf-8 -*-
"""
共享常量模块 — 统一管理害虫类别名单与置信度阈值
================================================
避免 main.py / semantic_kb / 训练脚本 等多处重复定义 16 类名单，
减少不一致风险（类别顺序必须与训练 data.yaml 完全一致）。
"""
from __future__ import annotations

# ── 16 类水稻害虫（YOLO 类别顺序，与训练 data.yaml 保持一致） ──
PEST_NAMES: list[str] = [
    'rice leaf roller', 'rice leaf caterpillar', 'paddy stem maggot',
    'asiatic rice borer', 'yellow rice borer', 'rice gall midge',
    'Rice Stemfly', 'brown plant hopper', 'white backed plant hopper',
    'small brown plant hopper', 'rice water weevil', 'rice leafhopper',
    'grain spreader thrips', 'rice shell pest', 'grub', 'mole cricket',
]

PEST_NAMES_ZH: list[str] = [
    '稻纵卷叶螟', '稻毛虫', '稻茎蛆', '二化螟', '三化螟', '稻瘿蚊',
    '稻茎蝇', '褐飞虱', '白背飞虱', '灰飞虱', '水稻象甲', '稻叶蝉',
    '稻蓟马', '稻螟蛉', '蛴螬', '蝼蛄',
]

# 类别索引 → 中文名 映射（便捷）
PEST_NAME_TO_ZH: dict[str, str] = dict(zip(PEST_NAMES, PEST_NAMES_ZH))

# ── 类别自适应置信度阈值 ─────────────────────────────────────────
# 特征明显的害虫用高阈值，易混淆的用低阈值避免漏检
CLASS_CONF_THRESHOLDS: dict[str, float] = {
    "asiatic rice borer": 0.75,       # 二化螟 — 特征明显
    "yellow rice borer": 0.70,        # 三化螟
    "brown plant hopper": 0.65,       # 褐飞虱
    "rice leafhopper": 0.55,          # 稻叶蝉 — 易混淆
    "rice leaf roller": 0.45,         # 稻纵卷叶螟 — 易混淆
    "rice shell pest": 0.50,          # 稻螟蛉 — 易混淆
    "paddy stem maggot": 0.45,        # 稻茎蛆
    "rice gall midge": 0.45,          # 稻瘿蚊
    "Rice Stemfly": 0.40,             # 稻茎蝇
    "white backed plant hopper": 0.50,
    "small brown plant hopper": 0.45,
    "rice water weevil": 0.55,
    "grain spreader thrips": 0.40,
    "rice leaf caterpillar": 0.45,
    "grub": 0.60,
    "mole cricket": 0.65,
}

# ── 上传 / 存储 限制 ────────────────────────────────────────────
MAX_UPLOAD_BYTES: int = 10 * 1024 * 1024      # 单张图片上限 10MB
RESULT_RETENTION_HOURS: float = 72.0          # 检测结果图保留 3 天
UPLOAD_RETENTION_HOURS: float = 24.0          # 上传临时文件保留 1 天
CLEANUP_INTERVAL_HOURS: float = 6.0           # 清理任务执行间隔
HISTORY_RETENTION_DAYS: int = 365             # 检测/问答历史记录保留天数（控制数据增长，可调）
