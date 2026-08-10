# -*- coding: utf-8 -*-
"""
轻量 SQLite 存储层 — 记录检测与问答历史，支持统计
零依赖：使用标准库 sqlite3，无需额外安装
"""
import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

_DB_PATH = Path(__file__).parent / "pest_data.db"
_LOCK = threading.Lock()

# ── 全局单例连接（WAL 模式，显著降低频繁建连开销） ─────────────
_conn = None


def _get_conn():
    """惰性创建全局单例连接。调用方需持有 _LOCK 保证线程安全。"""
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(str(_DB_PATH), timeout=15, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.execute("PRAGMA synchronous=NORMAL")
        _conn.execute("PRAGMA busy_timeout=10000")
    return _conn


def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def init_db():
    """初始化数据库表（幂等，可重复调用）"""
    with _LOCK:
        conn = _get_conn()
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS detection_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT,
            detections_json TEXT DEFAULT '[]',
            total INTEGER DEFAULT 0,
            elapsed_ms REAL DEFAULT 0,
            is_unknown INTEGER DEFAULT 0,
            unknown_type TEXT DEFAULT '',
            created_at TEXT
        );
        CREATE TABLE IF NOT EXISTS qa_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT,
            answer TEXT,
            pest_name TEXT DEFAULT '',
            used_llm INTEGER DEFAULT 0,
            created_at TEXT
        );
        """)
        conn.commit()


def save_detection(result_image: str, detections, total: int,
                   elapsed_ms: float, is_unknown: bool, unknown_type: str):
    """保存一条检测记录。detections 为 Detection 对象列表（兼容 dict）"""
    dets = []
    for d in detections:
        if isinstance(d, dict):
            dets.append({
                "name": d.get("name", ""),
                "zh_name": d.get("zh_name", ""),
                "confidence": d.get("confidence", 0),
                "bbox": d.get("bbox", []),
            })
        else:
            dets.append({
                "name": getattr(d, "name", ""),
                "zh_name": getattr(d, "zh_name", ""),
                "confidence": getattr(d, "confidence", 0),
                "bbox": getattr(d, "bbox", []),
            })
    with _LOCK:
        conn = _get_conn()
        conn.execute(
            "INSERT INTO detection_records "
            "(image_path, detections_json, total, elapsed_ms, is_unknown, unknown_type, created_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (result_image, json.dumps(dets, ensure_ascii=False), total,
             elapsed_ms, 1 if is_unknown else 0, unknown_type, _now()),
        )
        conn.commit()


def save_qa(question: str, answer: str, pest_name: str, used_llm: bool):
    """保存一条问答记录"""
    with _LOCK:
        conn = _get_conn()
        conn.execute(
            "INSERT INTO qa_records (question, answer, pest_name, used_llm, created_at) "
            "VALUES (?,?,?,?,?)",
            (question, answer, pest_name, 1 if used_llm else 0, _now()),
        )
        conn.commit()


def get_detections(limit: int = 20):
    with _LOCK:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT * FROM detection_records ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_qa(limit: int = 20):
    with _LOCK:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT * FROM qa_records ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_stats():
    """统计：总数、害虫频率 TOP10、每日趋势、未知数"""
    with _LOCK:
        conn = _get_conn()
        total_det = conn.execute(
            "SELECT COUNT(*) c FROM detection_records"
        ).fetchone()["c"]
        total_qa = conn.execute(
            "SELECT COUNT(*) c FROM qa_records"
        ).fetchone()["c"]
        unknown = conn.execute(
            "SELECT COUNT(*) c FROM detection_records WHERE is_unknown=1"
        ).fetchone()["c"]

        # 害虫出现频率（解析 detections_json）
        pest_freq: dict[str, int] = {}
        rows = conn.execute("SELECT detections_json FROM detection_records").fetchall()
        for r in rows:
            try:
                dets = json.loads(r["detections_json"] or "[]")
                for d in dets:
                    name = d.get("zh_name") or d.get("name") or "未知"
                    pest_freq[name] = pest_freq.get(name, 0) + 1
            except Exception:
                pass
        top_pests = sorted(pest_freq.items(), key=lambda x: x[1], reverse=True)[:10]

        # 每日检测趋势（最近 7 天）
        rows = conn.execute(
            "SELECT substr(created_at,1,10) d, COUNT(*) c FROM detection_records "
            "GROUP BY d ORDER BY d DESC LIMIT 7"
        ).fetchall()
        trend = [{"date": r["d"], "count": r["c"]} for r in rows][::-1]

        return {
            "total_detections": total_det,
            "total_qa": total_qa,
            "unknown_count": unknown,
            "top_pests": [{"name": n, "count": c} for n, c in top_pests],
            "daily_trend": trend,
        }
