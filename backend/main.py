# -*- coding: utf-8 -*-
"""
农作物害虫智能检测系统 — Web 后端入口
======================================
FastAPI + YOLO v11

职责：
  1. 创建应用、中间件、静态资源挂载
  2. 后台线程加载模型与增强模块（写入门 backend/state.py）
  3. 挂载各业务路由（/detect /qa /history /weather /enhanced）
  4. 提供根页面与健康检查

业务逻辑按职责拆分到 backend/routes_*.py，本文件保持精简。
"""
import logging
import os
import sys
import threading
import time
import uuid
from pathlib import Path

# 确保 backend 包可导入（兼容任意 cwd 启动）
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

from backend import routes_detect, routes_enhanced, routes_history, routes_qa, routes_weather
from backend.state import state
from backend.config import FRONTEND_DIR, MODEL_PATH, RESULT_DIR, UPLOAD_DIR
from backend.constants import CLEANUP_INTERVAL_HOURS, RESULT_RETENTION_HOURS, UPLOAD_RETENTION_HOURS
from backend.storage import init_db as storage_init

# ── 自动加载项目根目录的 .env 文件 ──────────────────────────────
_env_file = _ROOT / ".env"
if _env_file.exists():
    for _line in _env_file.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            _k = _k.strip()
            if not os.environ.get(_k, "").strip():
                os.environ[_k] = _v.strip()

# ── 新模块集成（带优雅降级） ─────────────────────────────────────
_HAS_NEW_MODULES = False
try:
    from backend.semantic_kb import create_knowledge_base as create_semantic_kb
    from backend.pest_agent import create_agent
    from backend.inference_engine import create_engine
    _HAS_NEW_MODULES = True
except ImportError as e:
    print(f"ℹ️ 新模块未加载（{e}），使用原始版本")

# ── 应用初始化 ────────────────────────────────────────────────────
app = FastAPI(
    title="害虫检测 API",
    description="基于 YOLO v11 的农作物害虫检测后端",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 结构化访问日志 ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
_log = logging.getLogger("pest-api")


@app.middleware("http")
async def http_access_log(request: Request, call_next):
    """为每个请求生成唯一 ID，并记录方法/路径/状态码/耗时"""
    request_id = uuid.uuid4().hex[:8]
    t0 = time.time()
    try:
        response = await call_next(request)
    except Exception:
        _log.exception("[%s] %s %s 处理异常", request_id, request.method, request.url.path)
        raise
    elapsed_ms = (time.time() - t0) * 1000
    response.headers["X-Request-ID"] = request_id
    _log.info(
        "[%s] %s %s -> %s (%.0fms)",
        request_id, request.method, request.url.path, response.status_code, elapsed_ms,
    )
    return response


# ── 挂载静态文件 ──────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR / "static")), name="static")
app.mount("/results", StaticFiles(directory=str(RESULT_DIR)), name="results")

# ── 模型加载（后台线程，不阻塞启动） ─────────────────────────────
def _load_models_background():
    """在后台线程中加载模型与增强模块，结果写入 state"""
    try:
        print("⏳ [后台] 正在加载 YOLO 模型...")
        state.model = YOLO(str(MODEL_PATH), task='detect')
        state.model(np_zeros())   # 预热
        print("✅ [后台] YOLO 模型加载完成")

        if _HAS_NEW_MODULES:
            try:
                local_kb = create_semantic_kb("hybrid")
                local_agent = create_agent(knowledge_base=local_kb)
                local_engine = create_engine(yolo_model=state.model, enable_clahe=True)
                state.semantic_kb = local_kb
                state.agent = local_agent
                state.inference_engine = local_engine
                print("✅ [后台] 增强模块加载完成：语义知识库 + AI Agent + 推理引擎")
            except Exception as e:
                print(f"ℹ️ [后台] 增强模块初始化失败（{e}），使用原始版本")

        state.models_ready = True
    except Exception as e:
        state.loading_error = str(e)
        print(f"❌ [后台] 模型加载失败: {e}")


def np_zeros(shape=(48, 48, 3)):
    """预热用占位数组（避免在入口引入 numpy）"""
    import numpy as np
    return np.zeros(shape, dtype=np.uint8)


_threading_started = False


def _ensure_loading_started():
    global _threading_started
    if not _threading_started:
        _threading_started = True
        thread = threading.Thread(target=_load_models_background, daemon=True)
        thread.start()


_ensure_loading_started()

# 初始化历史记录数据库（幂等）
storage_init()


# ── 运行时文件清理（防止结果/上传目录无限增长） ──────────────────
def _cleanup_old_files(directory: Path, max_hours: float) -> int:
    """删除目录中超过指定时长的文件，返回删除数量"""
    if not directory.exists():
        return 0
    cutoff = time.time() - max_hours * 3600
    removed = 0
    try:
        for p in directory.iterdir():
            if not p.is_file():
                continue
            try:
                if p.stat().st_mtime < cutoff:
                    p.unlink(missing_ok=True)
                    removed += 1
            except OSError:
                pass
    except OSError:
        pass
    if removed:
        print(f"🧹 清理 {directory.name}: 删除 {removed} 个过期文件")
    return removed


def _start_cleanup_scheduler() -> None:
    """后台线程周期性清理过期文件"""
    def _loop():
        while True:
            try:
                _cleanup_old_files(RESULT_DIR, RESULT_RETENTION_HOURS)
                _cleanup_old_files(UPLOAD_DIR, UPLOAD_RETENTION_HOURS)
            except Exception:
                pass
            time.sleep(CLEANUP_INTERVAL_HOURS * 3600)

    threading.Thread(target=_loop, name="file-cleanup", daemon=True).start()


_cleanup_old_files(RESULT_DIR, RESULT_RETENTION_HOURS)
_cleanup_old_files(UPLOAD_DIR, UPLOAD_RETENTION_HOURS)
_start_cleanup_scheduler()


# ── 挂载业务路由 ──────────────────────────────────────────────────
app.include_router(routes_detect.router)
app.include_router(routes_qa.router)
app.include_router(routes_history.router)
app.include_router(routes_weather.router)
app.include_router(routes_enhanced.router)


# ── 根路由与健康检查 ─────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def index():
    """品牌加载页（入场动画）"""
    return FileResponse(str(FRONTEND_DIR / "templates" / "loading.html"))


@app.get("/app")
async def app_index():
    """主应用页面（由加载页跳转过来）"""
    return FileResponse(str(FRONTEND_DIR / "templates" / "index.html"))


@app.get("/health")
async def health():
    if state.loading_error:
        return {"status": "error", "message": state.loading_error}
    if not state.models_ready:
        return {"status": "loading", "message": "模型加载中"}
    return {"status": "ok", "model": str(MODEL_PATH.name)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
