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
            original_image TEXT DEFAULT '',
            detections_json TEXT DEFAULT '[]',
            total INTEGER DEFAULT 0,
            elapsed_ms REAL DEFAULT 0,
            is_unknown INTEGER DEFAULT 0,
            unknown_type TEXT DEFAULT '',
            username TEXT DEFAULT '',
            is_internal INTEGER DEFAULT 0,
            expert_mark TEXT DEFAULT '',
            expert_note TEXT DEFAULT '',
            marked_by TEXT DEFAULT '',
            marked_at TEXT,
            created_at TEXT
        );
        CREATE TABLE IF NOT EXISTS qa_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT,
            answer TEXT,
            pest_name TEXT DEFAULT '',
            used_llm INTEGER DEFAULT 0,
            username TEXT DEFAULT '',
            is_internal INTEGER DEFAULT 0,
            created_at TEXT
        );
        """)
        # 旧库迁移：表已存在但缺新列时补充（幂等）
        dcols = {r["name"] for r in conn.execute("PRAGMA table_info(detection_records)").fetchall()}
        for col in ("username", "original_image", "expert_mark", "expert_note", "marked_by", "marked_at"):
            if col not in dcols:
                conn.execute(f"ALTER TABLE detection_records ADD COLUMN {col} TEXT DEFAULT ''")
        if "is_internal" not in dcols:
            conn.execute("ALTER TABLE detection_records ADD COLUMN is_internal INTEGER DEFAULT 0")
        qcols = {r["name"] for r in conn.execute("PRAGMA table_info(qa_records)").fetchall()}
        if "username" not in qcols:
            conn.execute("ALTER TABLE qa_records ADD COLUMN username TEXT DEFAULT ''")
        if "is_internal" not in qcols:
            conn.execute("ALTER TABLE qa_records ADD COLUMN is_internal INTEGER DEFAULT 0")
        # 常用查询索引（按用户 / 按时间，随数据量增长保证查询性能）
        conn.execute("CREATE INDEX IF NOT EXISTS idx_det_user ON detection_records(username)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_det_created ON detection_records(created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_qa_user ON qa_records(username)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_qa_created ON qa_records(created_at)")
        conn.commit()


def save_detection(result_image: str, detections, total: int,
                   elapsed_ms: float, is_unknown: bool, unknown_type: str,
                   username: str = "", original_image: str = "", is_internal: bool = False):
    """保存一条检测记录。detections 为 Detection 对象列表（兼容 dict）
    is_internal=True 表示内部/测试数据（专家/管理员操作），不进用户质量统计"""
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
            "(image_path, original_image, detections_json, total, elapsed_ms, is_unknown, unknown_type, username, is_internal, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (result_image, original_image, json.dumps(dets, ensure_ascii=False), total,
             elapsed_ms, 1 if is_unknown else 0, unknown_type, username,
             1 if is_internal else 0, _now()),
        )
        conn.commit()


def save_qa(question: str, answer: str, pest_name: str, used_llm: bool,
            username: str = "", is_internal: bool = False):
    """保存一条问答记录；is_internal=True 表示内部/测试数据"""
    with _LOCK:
        conn = _get_conn()
        conn.execute(
            "INSERT INTO qa_records (question, answer, pest_name, used_llm, username, is_internal, created_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (question, answer, pest_name, 1 if used_llm else 0, username,
             1 if is_internal else 0, _now()),
        )
        conn.commit()


def get_detections(limit: int = 20, username: str | None = None):
    """检测记录；username=None 返回全部（管理员），否则只看该用户名（空串=游客）"""
    with _LOCK:
        conn = _get_conn()
        if username is None:
            rows = conn.execute(
                "SELECT * FROM detection_records ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM detection_records WHERE username=? ORDER BY id DESC LIMIT ?",
                (username, limit),
            ).fetchall()
        return [dict(r) for r in rows]


# ── 专家工作台：识别质量审核 ──────────────────────────────────
def get_expert_detections(limit: int = 50, only_unmarked: bool = False):
    """专家视角的检测记录（含原图/结果图/标记状态）；排除内部测试数据；only_unmarked=只取未标记"""
    with _LOCK:
        conn = _get_conn()
        if only_unmarked:
            rows = conn.execute(
                "SELECT * FROM detection_records WHERE is_internal=0 AND expert_mark='' ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM detection_records WHERE is_internal=0 ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]


def get_expert_stats():
    """专家工作台概览：总数 / 待审数 / 各类标记数（排除内部测试数据）"""
    with _LOCK:
        conn = _get_conn()
        total = conn.execute(
            "SELECT COUNT(*) c FROM detection_records WHERE is_internal=0"
        ).fetchone()["c"]
        unmarked = conn.execute(
            "SELECT COUNT(*) c FROM detection_records WHERE is_internal=0 AND expert_mark=''"
        ).fetchone()["c"]
        marked = {}
        for m in ("success", "ambiguous", "failed"):
            marked[m] = conn.execute(
                "SELECT COUNT(*) c FROM detection_records WHERE is_internal=0 AND expert_mark=?", (m,)
            ).fetchone()["c"]
        return {"total": total, "unmarked": unmarked, "marked": marked}


def mark_detection(record_id: int, mark: str, note: str = "", marked_by: str = "") -> bool:
    """专家打标：success / ambiguous / failed"""
    if mark not in ("success", "ambiguous", "failed", ""):
        return False
    with _LOCK:
        conn = _get_conn()
        cur = conn.execute(
            "UPDATE detection_records SET expert_mark=?, expert_note=?, marked_by=?, marked_at=? WHERE id=?",
            (mark, note, marked_by, _now() if mark else "", record_id),
        )
        conn.commit()
        return cur.rowcount > 0


# ── 专家反馈闭环：管理台汇总 + 难例导出 ───────────────────────
def get_expert_feedback_stats():
    """管理员视角的专家反馈汇总：标记分布 / 薄弱害虫 TOP / 每日打标趋势（排除内部测试数据）"""
    with _LOCK:
        conn = _get_conn()
        marked = {}
        for m in ("success", "ambiguous", "failed"):
            marked[m] = conn.execute(
                "SELECT COUNT(*) c FROM detection_records WHERE is_internal=0 AND expert_mark=?", (m,)
            ).fetchone()["c"]
        unmarked = conn.execute(
            "SELECT COUNT(*) c FROM detection_records WHERE is_internal=0 AND expert_mark=''"
        ).fetchone()["c"]

        # 被标 失败/模糊 的记录中，出现最多的害虫类别（模型薄弱点）
        weak: dict[str, int] = {}
        rows = conn.execute(
            "SELECT detections_json FROM detection_records WHERE is_internal=0 AND expert_mark IN ('failed','ambiguous')"
        ).fetchall()
        for r in rows:
            try:
                dets = json.loads(r["detections_json"] or "[]")
                for d in dets:
                    name = d.get("zh_name") or d.get("name") or "未知"
                    weak[name] = weak.get(name, 0) + 1
            except Exception:
                pass
        top_weak = sorted(weak.items(), key=lambda x: x[1], reverse=True)[:10]

        # 每日打标趋势（近 7 天，按标记类型拆分）
        trend_rows = conn.execute(
            "SELECT substr(marked_at,1,10) d, expert_mark m, COUNT(*) c "
            "FROM detection_records WHERE is_internal=0 AND expert_mark!='' AND marked_at!='' "
            "GROUP BY d, m ORDER BY d DESC LIMIT 21"
        ).fetchall()
        grouped: dict[str, dict] = {}
        for r in trend_rows:
            grouped.setdefault(r["d"], {})[r["m"]] = r["c"]
        daily_trend = [
            {"date": d, "success": g.get("success", 0),
             "ambiguous": g.get("ambiguous", 0), "failed": g.get("failed", 0)}
            for d, g in sorted(grouped.items())[-7:]
        ]

        return {
            "marked": marked,
            "unmarked": unmarked,
            "top_weak_pests": [{"name": n, "count": c} for n, c in top_weak],
            "daily_trend": daily_trend,
        }


def get_hard_examples(limit: int = 200):
    """难例集：被标记为「识别失败」的记录（原图/结果图/备注/预测），供数据回流"""
    with _LOCK:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT id, image_path, original_image, detections_json, expert_note, "
            "marked_by, marked_at, username, created_at "
            "FROM detection_records WHERE is_internal=0 AND expert_mark='failed' "
            "ORDER BY marked_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            try:
                d["detections_json"] = json.loads(d.get("detections_json") or "[]")
            except Exception:
                d["detections_json"] = []
            out.append(d)
        return out


def cleanup_old_records(keep_days: int = 365) -> int:
    """保留策略：删除超过 keep_days 天的检测/问答记录，控制数据膨胀。
    返回删除条数。调用方（后台调度器）周期执行。"""
    from datetime import datetime, timedelta
    cutoff = (datetime.now() - timedelta(days=keep_days)).strftime("%Y-%m-%d %H:%M:%S")
    removed = 0
    with _LOCK:
        conn = _get_conn()
        c1 = conn.execute("DELETE FROM detection_records WHERE created_at < ?", (cutoff,))
        c2 = conn.execute("DELETE FROM qa_records WHERE created_at < ?", (cutoff,))
        conn.commit()
        removed = int(c1.rowcount) + int(c2.rowcount)
    if removed:
        print(f"🧹 数据保留策略: 清理 {removed} 条超期记录(>{keep_days}天)")
    return removed


def get_qa(limit: int = 20, username: str | None = None):
    """问答记录；username=None 返回全部，否则只看该用户名"""
    with _LOCK:
        conn = _get_conn()
        if username is None:
            rows = conn.execute(
                "SELECT * FROM qa_records ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM qa_records WHERE username=? ORDER BY id DESC LIMIT ?",
                (username, limit),
            ).fetchall()
        return [dict(r) for r in rows]


def get_stats(username: str | None = None):
    """统计：总数、害虫频率 TOP10、每日趋势、未知数
    username=None 全量（管理台，默认排除内部测试数据，另返回 internal_* 计数）；
    否则只看该用户名（个人视角，含其内部记录）"""
    with _LOCK:
        conn = _get_conn()
        if username is not None:
            total_det = conn.execute(
                "SELECT COUNT(*) c FROM detection_records WHERE username=?", (username,)
            ).fetchone()["c"]
            total_qa = conn.execute(
                "SELECT COUNT(*) c FROM qa_records WHERE username=?", (username,)
            ).fetchone()["c"]
            unknown = conn.execute(
                "SELECT COUNT(*) c FROM detection_records WHERE username=? AND is_unknown=1",
                (username,),
            ).fetchone()["c"]
            internal_det = 0
            internal_qa = 0
            det_where = " WHERE username=?"
            det_args = (username,)
            trend_where = " WHERE username=?"
            trend_args = (username,)
        else:
            total_det = conn.execute(
                "SELECT COUNT(*) c FROM detection_records WHERE is_internal=0"
            ).fetchone()["c"]
            total_qa = conn.execute(
                "SELECT COUNT(*) c FROM qa_records WHERE is_internal=0"
            ).fetchone()["c"]
            unknown = conn.execute(
                "SELECT COUNT(*) c FROM detection_records WHERE is_internal=0 AND is_unknown=1"
            ).fetchone()["c"]
            internal_det = conn.execute(
                "SELECT COUNT(*) c FROM detection_records WHERE is_internal=1"
            ).fetchone()["c"]
            internal_qa = conn.execute(
                "SELECT COUNT(*) c FROM qa_records WHERE is_internal=1"
            ).fetchone()["c"]
            det_where = " WHERE is_internal=0"
            det_args = ()
            trend_where = " WHERE is_internal=0"
            trend_args = ()

        # 害虫出现频率（解析 detections_json）
        pest_freq: dict[str, int] = {}
        rows = conn.execute(
            f"SELECT detections_json FROM detection_records{det_where}", det_args
        ).fetchall()
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
            f"SELECT substr(created_at,1,10) d, COUNT(*) c FROM detection_records{trend_where} "
            "GROUP BY d ORDER BY d DESC LIMIT 7", trend_args
        ).fetchall()
        trend = [{"date": r["d"], "count": r["c"]} for r in rows][::-1]

        return {
            "total_detections": total_det,
            "total_qa": total_qa,
            "unknown_count": unknown,
            "top_pests": [{"name": n, "count": c} for n, c in top_pests],
            "daily_trend": trend,
            "internal_detections": internal_det,
            "internal_qa": internal_qa,
        }
