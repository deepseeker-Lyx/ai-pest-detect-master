# -*- coding: utf-8 -*-
"""
路径与运行配置 — 集中管理目录常量
================================
确保从任何工作目录都能以 `backend.xxx` 导入（兼容 `uvicorn backend.main:app`
与 `cd backend && python main.py` 两种启动方式）。
"""
from __future__ import annotations

import sys
from pathlib import Path

# 项目根目录加入 sys.path，保证 `backend.*` 导入在任何 cwd 下可用
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── 路径常量 ──────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent          # backend/
MODEL_PATH   = BASE_DIR / "models" / "best.pt"
UPLOAD_DIR   = BASE_DIR / "uploads"
RESULT_DIR   = BASE_DIR / "results"
FRONTEND_DIR = _ROOT / "frontend"

UPLOAD_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)
