# -*- coding: utf-8 -*-
"""
🔬 多尺度 / 滑窗推理 — 小目标检测增强（论文实验 / 离线高精度模式）
====================================================================
动机：小目标（如灰飞虱、稻蓟马）在 640×640 输入下像素占比极小，
     经过 YOLO 下采样后特征信息丢失。将图片放大后再推理，
     小目标特征更清晰，可显著提升召回。

三种模式：
  1. 单尺度（baseline）  : 与线上一致，作为对照
  2. 多尺度（multiscale）: 对 0.8/1.0/1.25/1.5 等多个尺度推理，坐标映射回原图后 NMS 融合
  3. 滑窗（sliding）     : 将大图切成重叠小块分别放大推理，再映射回原图融合

用法：
  # 单张图片
  python scripts/multiscale_detect.py --weights backend/models/best.pt \
      --source 某图片.jpg --mode multiscale --scales 0.8,1.0,1.25,1.5

  # 整个目录（输出对比汇总表 + 保存融合结果图）
  python scripts/multiscale_detect.py --source 图片目录 --mode multiscale \
      --scales 0.8,1.0,1.25,1.5 --save output_dir

  # 滑窗模式（大图）
  python scripts/multiscale_detect.py --source 大图.jpg --mode sliding --window 2

说明：
  - 多尺度会增加推理次数（每个尺度一次），CPU 上延迟成倍增加，
    适合"离线高精度"或论文实验；线上实时仍用单尺度。
  - 小目标定义：框面积 / 原图面积 < 1%（可 --small-ratio 调整）。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np


# ═══════════════════════════════════════════════════════════════
#  参数解析
# ═══════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="多尺度/滑窗小目标检测增强")
    p.add_argument("--weights", default="backend/models/best.pt", help="模型权重路径")
    p.add_argument("--source", required=True, help="图片路径或目录")
    p.add_argument("--mode", choices=["single", "multiscale", "sliding", "all"], default="all",
                   help="single=单尺度对照 / multiscale=多尺度 / sliding=滑窗 / all=三者都跑")
    p.add_argument("--scales", default="0.8,1.0,1.25,1.5", help="多尺度推理的缩放比例（逗号分隔）")
    p.add_argument("--window", type=int, default=2, help="滑窗切分数（N×N 块）")
    p.add_argument("--overlap", type=float, default=0.25, help="滑窗重叠比例（0-1）")
    p.add_argument("--imgsz", type=int, default=640, help="推理输入分辨率")
    p.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    p.add_argument("--iou", type=float, default=0.45, help="融合 NMS 的 IoU 阈值")
    p.add_argument("--small-ratio", type=float, default=0.01, help="小目标判定：框面积/原图面积 阈值")
    p.add_argument("--save", default=None, help="结果保存目录（默认不保存）")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════
#  检测工具
# ═══════════════════════════════════════════════════════════════

def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    """贪心 NMS：boxes(xyxy, N×4), scores(N)，返回保留的索引数组"""
    if len(boxes) == 0:
        return np.array([], dtype=int)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = np.argsort(-scores)
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / np.maximum(union, 1e-9)
        order = order[np.where(iou <= iou_thr)[0] + 1]
    return np.array(keep, dtype=int)


def _detect_on_scaled(model, img: np.ndarray, scale: float, imgsz: int, conf: float) -> np.ndarray:
    """将 img 缩放 scale 倍后推理，返回原图坐标的检测 [x1,y1,x2,y2,conf,cls]"""
    h, w = img.shape[:2]
    rw, rh = max(int(round(w * scale)), 16), max(int(round(h * scale)), 16)
    scaled = cv2.resize(img, (rw, rh))
    res = model.predict(scaled, imgsz=imgsz, conf=conf, verbose=False)[0]
    boxes = res.boxes
    if boxes is None or len(boxes) == 0:
        return np.zeros((0, 6), dtype=np.float32)
    xyxy = boxes.xyxy.cpu().numpy() / scale
    confs = boxes.conf.cpu().numpy().reshape(-1, 1)
    clss = boxes.cls.cpu().numpy().reshape(-1, 1)
    return np.hstack([xyxy, confs, clss])


def _fuse(det_list: list[np.ndarray], iou_thr: float) -> np.ndarray:
    """融合多个尺度的检测：按类别分别 NMS"""
    nonempty = [d for d in det_list if len(d) > 0]
    if not nonempty:
        return np.zeros((0, 6), dtype=np.float32)
    all_det = np.vstack(nonempty)
    if len(all_det) == 0:
        return all_det
    keep = []
    for cls in np.unique(all_det[:, 5]):
        m = all_det[:, 5] == cls
        idx = np.where(m)[0]
        sub = all_det[m]
        keep_idx = _nms(sub[:, :4], sub[:, 4], iou_thr)
        keep.extend(idx[keep_idx])
    return all_det[np.array(sorted(keep), dtype=int)]


def _count_small(dets: np.ndarray, img_area: float, ratio: float) -> int:
    """统计小目标数量（框面积 / 原图面积 < ratio）"""
    if len(dets) == 0:
        return 0
    areas = (dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1])
    return int(np.sum(areas / img_area < ratio))


# ═══════════════════════════════════════════════════════════════
#  三种模式
# ═══════════════════════════════════════════════════════════════

def run_single(model, img: np.ndarray, imgsz: int, conf: float) -> np.ndarray:
    res = model.predict(img, imgsz=imgsz, conf=conf, verbose=False)[0]
    boxes = res.boxes
    if boxes is None or len(boxes) == 0:
        return np.zeros((0, 6), dtype=np.float32)
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy().reshape(-1, 1)
    clss = boxes.cls.cpu().numpy().reshape(-1, 1)
    return np.hstack([xyxy, confs, clss])


def run_multiscale(model, img: np.ndarray, scales: list[float],
                   imgsz: int, conf: float, iou: float) -> np.ndarray:
    dets = [_detect_on_scaled(model, img, s, imgsz, conf) for s in scales]
    return _fuse(dets, iou)


def run_sliding(model, img: np.ndarray, window: int, overlap: float,
                imgsz: int, conf: float, iou: float) -> np.ndarray:
    """把图切成 window×window 块（带重叠），每块放大到 imgsz 推理后映射回原图融合"""
    h, w = img.shape[:2]
    stride_x = w / max(window, 1)
    stride_y = h / max(window, 1)
    dets = []
    for gy in range(window):
        for gx in range(window):
            x1 = max(0, int(gx * stride_x - overlap * stride_x))
            y1 = max(0, int(gy * stride_y - overlap * stride_y))
            x2 = min(w, int((gx + 1) * stride_x + overlap * stride_x))
            y2 = min(h, int((gy + 1) * stride_y + overlap * stride_y))
            if x2 - x1 < 16 or y2 - y1 < 16:
                continue
            patch = img[y1:y2, x1:x2]
            scale = imgsz / max(x2 - x1, y2 - y1)  # 让最长边放大到 imgsz
            d = _detect_on_scaled(model, patch, scale, imgsz, conf)
            if len(d) > 0:
                d[:, 0] += x1
                d[:, 1] += y1
                d[:, 2] += x1
                d[:, 3] += y1
                dets.append(d)
    return _fuse(dets, iou)


# ═══════════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════════

def process_one(model, img_path: Path, args: argparse.Namespace) -> dict:
    img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return {"file": str(img_path), "error": "图片解码失败"}
    h, w = img.shape[:2]
    img_area = float(h * w)
    scales = [float(x) for x in args.scales.split(",") if x.strip()]

    modes = ["single", "multiscale", "sliding"] if args.mode == "all" else [args.mode]
    result = {"file": str(img_path), "size": f"{w}x{h}", "modes": {}}

    for mode in modes:
        t0 = time.time()
        if mode == "single":
            dets = run_single(model, img, args.imgsz, args.conf)
        elif mode == "multiscale":
            dets = run_multiscale(model, img, scales, args.imgsz, args.conf, args.iou)
        else:
            dets = run_sliding(model, img, args.window, args.overlap, args.imgsz, args.conf, args.iou)
        elapsed = time.time() - t0
        result["modes"][mode] = {
            "detections": int(len(dets)),
            "small": _count_small(dets, img_area, args.small_ratio),
            "elapsed_ms": round(elapsed * 1000, 1),
        }
        if args.save:
            save_img = img.copy()
            for d in dets:
                x1, y1, x2, y2, c, cls = [float(v) for v in d]
                cv2.rectangle(save_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.putText(save_img, f"{int(cls)} {c:.2f}", (int(x1), max(15, int(y1) - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            out_dir = Path(args.save)
            out_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_dir / f"{img_path.stem}_{mode}.jpg"), save_img)
    return result


def main() -> int:
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ 请先安装 ultralytics")
        return 1

    print(f"📦 加载模型: {args.weights}")
    model = YOLO(args.weights)
    print("✅ 模型加载完成\n")

    src = Path(args.source)
    if src.is_file():
        paths = [src]
    elif src.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        paths = sorted([p for p in src.iterdir() if p.suffix.lower() in exts])
        if not paths:
            print(f"❌ 目录中没有图片: {src}")
            return 1
        print(f"📂 目录 {src} 共 {len(paths)} 张图片")
    else:
        print(f"❌ 路径不存在: {src}")
        return 1

    results = []
    for p in paths:
        r = process_one(model, p, args)
        results.append(r)
        modes = r.get("modes", {})
        brief = " | ".join(f"{k}: {v['detections']}个(小目标{v['small']})" for k, v in modes.items())
        print(f"  {p.name}: {brief}")

    # 汇总表
    if len(results) > 1:
        print("\n📊 汇总（总计）")
        header = f"{'模式':<12} | {'检测总数':>8} | {'小目标数':>8} | {'平均耗时ms':>10}"
        print(header)
        print("-" * len(header))
        for mode in ["single", "multiscale", "sliding"]:
            if any(mode in r["modes"] for r in results):
                total_d = sum(r["modes"][mode]["detections"] for r in results)
                total_s = sum(r["modes"][mode]["small"] for r in results)
                avg_ms = sum(r["modes"][mode]["elapsed_ms"] for r in results) / len(results)
                print(f"{mode:<12} | {total_d:>8} | {total_s:>8} | {avg_ms:>10.1f}")

    # 保存 JSON
    if args.save:
        out_dir = Path(args.save)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n💾 结果已保存: {out_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
