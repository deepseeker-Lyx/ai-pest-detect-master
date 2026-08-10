# -*- coding: utf-8 -*-
"""
全局运行时状态 — 由 main.py 在后台线程初始化，各路由模块读取
============================================================
避免循环导入：main.py 写入，路由模块只读。
"""
from __future__ import annotations


class _RuntimeState:
    """持有模型、推理引擎、Agent、知识库等运行时对象"""

    def __init__(self) -> None:
        self.model = None               # YOLO 模型
        self.inference_engine = None    # 推理加速引擎（可选）
        self.agent = None               # AI 防治助手（可选）
        self.semantic_kb = None         # 语义知识库（可选）
        self.models_ready = False       # 模型是否就绪
        self.loading_error: str | None = None  # 加载失败原因


state = _RuntimeState()
