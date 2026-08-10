# -*- coding: utf-8 -*-
"""
检测路由 — /detect/*
====================
包含：图片上传检测、Base64 检测、增强检测（+Agent 分析）
以及检测相关的工具函数（文件校验 / IoU / 合并 / 开放集识别）。
"""
from __future__ import annotations

import asyncio
import base64
import sys
import time
import uuid
from pathlib import Path

# 确保 backend 包可导入（兼容任意 cwd 启动）
if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from backend.state import state
from backend.config import RESULT_DIR, UPLOAD_DIR
from backend.constants import CLASS_CONF_THRESHOLDS, MAX_UPLOAD_BYTES, PEST_NAMES as NAMES
from backend.knowledge_base import knowledge_base
from backend.pest_agent import PestDetection, detection_to_report
from backend.schemas import Detection, DetectResponse
from backend.storage import save_detection

router = APIRouter()


# ── 文件校验工具 ─────────────────────────────────────────────────
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

# 常见图片格式文件头魔数（防止伪装成图片的恶意文件）
_IMAGE_MAGIC: list[tuple[bytes, str]] = [
    (b"\xff\xd8\xff", "jpeg"),
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"BM", "bmp"),
    (b"GIF87a", "gif"),
    (b"GIF89a", "gif"),
    (b"RIFF", "webp"),
]


def _check_ext(filename: str):
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型: {ext}. 允许: {', '.join(ALLOWED_EXTENSIONS)}"
        )


def _validate_image_bytes(content: bytes) -> None:
    """校验上传内容：大小上限 + 文件头魔数，拒绝伪装/超大文件"""
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"文件过大，最大允许 {MAX_UPLOAD_BYTES // (1024 * 1024)}MB",
        )
    if not content:
        raise HTTPException(status_code=400, detail="空文件")
    if not any(content.startswith(magic) for magic, _ in _IMAGE_MAGIC):
        raise HTTPException(
            status_code=400,
            detail="文件不是有效的图片（文件头校验失败）",
        )


def _safe_filename(original: str) -> str:
    uid = uuid.uuid4().hex[:12]
    ext = Path(original).suffix.lower()
    return f"{uid}{ext}"


# ── 检测结果处理 ─────────────────────────────────────────────────
def _make_detection(index: int, box: list[float], cls: float, conf: float) -> Detection:
    name = NAMES[int(cls)] if int(cls) < len(NAMES) else f"class_{int(cls)}"
    info = knowledge_base.pest_info(name) or {}
    return Detection(
        id=index + 1,
        name=name,
        zh_name=info.get("zh_name", ""),
        confidence=round(float(conf) * 100, 2),
        bbox=[int(x) for x in box],
        symptoms=info.get("symptoms", []),
        prevention=info.get("prevention", []),
    )


def _iou(box1: list[int], box2: list[int]) -> float:
    """计算两个边界框的 IoU（交并比）"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def _merge_uncertain_detections(detections: list[Detection]) -> list[Detection]:
    """
    合并重叠度高且置信度接近的检测框。

    规则：
    1. 主框置信度 >= 75% → 不显示备选项，直接抑制重叠框
    2. 主框置信度 < 75% 且重叠度高、置信度接近 → 显示备选项
    3. 同一名称去重
    """
    if len(detections) < 2:
        return detections

    merged = []
    used = set()

    for i, det in enumerate(detections):
        if i in used:
            continue
        alternatives = []
        for j, other in enumerate(detections):
            if i == j or j in used:
                continue
            iou = _iou(det.bbox, other.bbox)
            conf_diff = abs(det.confidence - other.confidence)
            if iou > 0.5 and conf_diff < 20:
                alternatives.append(other)
                used.add(j)

        if alternatives:
            all_dets = [det] + alternatives
            all_dets.sort(key=lambda d: d.confidence, reverse=True)
            primary = all_dets[0]

            if primary.confidence >= 75:
                merged.append(primary)
            else:
                seen_names = {primary.name}
                unique_alts = []
                for d in all_dets[1:]:
                    if d.name not in seen_names:
                        seen_names.add(d.name)
                        unique_alts.append({
                            "name": d.name,
                            "zh_name": d.zh_name,
                            "confidence": d.confidence,
                        })
                primary.alternatives = unique_alts
                merged.append(primary)
        else:
            merged.append(det)
        used.add(i)

    return merged


def _run_detect(img_input, result_name: str) -> DetectResponse:
    """执行一次完整检测：推理 → 阈值过滤 → 合并 → 开放集识别 → 存历史"""
    t0 = time.time()

    # 推理引擎仅用于缓存加速（输入需为 numpy 数组）
    if state.inference_engine is not None and isinstance(img_input, np.ndarray):
        result, timeline = state.inference_engine.infer(img_input)
        elapsed = timeline.total_ms
        results = result
    else:
        results = state.model(img_input, conf=0.25, augment=True)[0]
        elapsed = (time.time() - t0) * 1000

    result_img = results.plot()
    result_path = RESULT_DIR / result_name
    cv2.imwrite(str(result_path), result_img)

    boxes = results.boxes
    # 用类别自适应阈值过滤检测框
    filtered_indices = []
    for i, (box, cls, conf) in enumerate(zip(
        boxes.xyxy.tolist(), boxes.cls.tolist(), boxes.conf.tolist()
    )):
        class_name = NAMES[int(cls)] if int(cls) < len(NAMES) else None
        threshold = CLASS_CONF_THRESHOLDS.get(class_name, 0.25) if class_name else 0.25
        if float(conf) >= threshold:
            filtered_indices.append(i)

    raw_detections = [
        _make_detection(i, boxes.xyxy.tolist()[idx], boxes.cls.tolist()[idx], boxes.conf.tolist()[idx])
        for i, idx in enumerate(filtered_indices)
    ]
    detections = _merge_uncertain_detections(raw_detections)

    # ═══════════════════════════════════════════════════════════
    # ⭐ 开放集识别（Open-Set Recognition）
    # 当模型对检测结果的置信度普遍偏低时，标记为"未知虫种"，避免强行归类。
    # ═══════════════════════════════════════════════════════════
    is_unknown = False
    unknown_type = ""
    unknown_suggestion = ""

    if detections:
        confs = [d.confidence for d in detections]
        max_conf = max(confs)

        if max_conf < 40.0:
            is_unknown = True
            unknown_type = "low_confidence"
            unknown_suggestion = (
                "当前检测到的目标置信度普遍较低，可能不在已知的 16 类水稻害虫范围内。"
                "建议咨询当地农技专家进行鉴定。"
            )
        elif max_conf < 55.0 and len(detections) > 1:
            is_unknown = True
            unknown_type = "uncertain_multiple"
            unknown_suggestion = (
                "检测到多个疑似目标但置信度均偏低，"
                "可能是未知害虫或非目标物体。建议上传更清晰的图片。"
            )
    else:
        is_unknown = True
        unknown_type = "no_detection"
        unknown_suggestion = (
            "未检测到已知的 16 类水稻害虫。"
            "可能是图片中不存在害虫，或害虫种类不在当前模型识别范围内。"
        )

    # 记录检测历史（SQLite，失败不影响主流程）
    try:
        save_detection(
            result_image=f"/results/{result_name}",
            detections=detections,
            total=len(detections),
            elapsed_ms=round(elapsed, 1),
            is_unknown=is_unknown,
            unknown_type=unknown_type,
        )
    except Exception:
        pass

    return DetectResponse(
        success=True,
        result_image=f"/results/{result_name}",
        detections=detections,
        total=len(detections),
        elapsed_ms=round(elapsed, 1),
        is_unknown=is_unknown,
        unknown_type=unknown_type,
        unknown_suggestion=unknown_suggestion,
    )


# ── 路由 ─────────────────────────────────────────────────────────
@router.post("/detect/image", response_model=DetectResponse)
async def detect_image(file: UploadFile = File(...)):
    """上传图片进行害虫检测"""
    if not state.models_ready:
        return JSONResponse(status_code=503, content={"detail": "模型加载中，请稍后重试"})
    _check_ext(file.filename)

    content = await file.read()
    _validate_image_bytes(content)

    save_name = _safe_filename(file.filename)
    upload_path = UPLOAD_DIR / save_name
    await asyncio.to_thread(upload_path.write_bytes, content)

    try:
        # YOLO 推理为 CPU/GPU 密集，放入线程池避免阻塞事件循环
        return await asyncio.to_thread(_run_detect, str(upload_path), f"result_{save_name}")
    finally:
        upload_path.unlink(missing_ok=True)


@router.post("/detect/base64", response_model=DetectResponse)
async def detect_base64(payload: dict):
    """接收 base64 编码的图片（移动端摄像头直接上传）"""
    data_url: str = payload.get("image", "")
    if not data_url:
        raise HTTPException(status_code=400, detail="缺少 image 字段")

    if "," in data_url:
        data_url = data_url.split(",", 1)[1]

    try:
        img_bytes = base64.b64decode(data_url)
        if len(img_bytes) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=400,
                detail=f"图片过大，最大允许 {MAX_UPLOAD_BYTES // (1024 * 1024)}MB",
            )
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("图片解码失败")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"图片解析错误: {e}")

    result_name = f"result_{uuid.uuid4().hex[:12]}.jpg"
    return await asyncio.to_thread(_run_detect, img, result_name)


@router.post("/detect/analyze")
async def detect_and_analyze(file: UploadFile = File(...)):
    """
    🔬 增强检测 —— 检测 + AI Agent 分析一步完成。

    返回结构化报告：诊断 → 分析 → 防治建议 → 预防措施
    """
    _check_ext(file.filename)

    content = await file.read()
    _validate_image_bytes(content)

    save_name = _safe_filename(file.filename)
    upload_path = UPLOAD_DIR / save_name
    await asyncio.to_thread(upload_path.write_bytes, content)

    try:
        # 1. 检测（CPU 密集，放入线程池避免阻塞事件循环）
        detect_result = await asyncio.to_thread(_run_detect, str(upload_path), f"result_{save_name}")

        # 2. 如果有 Agent，生成结构化分析报告
        if state.agent is not None and detect_result.detections:
            reports = []
            for det in detect_result.detections:
                pest_det = PestDetection(
                    name=det.name,
                    zh_name=det.zh_name,
                    confidence=det.confidence,
                    bbox=det.bbox,
                    symptoms=det.symptoms,
                    prevention=det.prevention,
                )
                report = await asyncio.to_thread(detection_to_report, state.agent, pest_det)
                reports.append(report)
            return {
                "success": True,
                "result_image": detect_result.result_image,
                "elapsed_ms": detect_result.elapsed_ms,
                "total": detect_result.total,
                "detections": [d.model_dump() for d in detect_result.detections],
                "analysis": reports,
            }

        # 3. 无 Agent，返回普通检测结果
        return {
            "success": True,
            "result_image": detect_result.result_image,
            "elapsed_ms": detect_result.elapsed_ms,
            "total": detect_result.total,
            "detections": [d.model_dump() for d in detect_result.detections],
            "message": "Agent 未加载，已返回基础检测结果",
        }
    finally:
        upload_path.unlink(missing_ok=True)
