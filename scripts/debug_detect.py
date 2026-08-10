# -*- coding: utf-8 -*-
"""
🔍 检测诊断脚本 — 定位"为什么识别不出来"
=========================================
背景：给系统上传含多个害虫的照片时，前端提示"未检测到/未知虫种"。
可能原因有三：
  ① 模型原始置信度低（多害虫/密集/小目标场景天然置信度偏低）
  ② 类别自适应阈值过滤（CLASS_CONF_THRESHOLDS 0.40~0.75，可能过滤掉低置信度检测）
  ③ 开放集识别误判（多目标且置信度都不高时被标记为"未知虫种"）

本脚本对图片做低阈值推理，输出【全部原始检测】与【经过 main.py 阈值过滤后】
的对比，并逐条标注每个框被哪个阈值过滤，帮你精确定位问题出在哪一步。

用法：
  python scripts/debug_detect.py --source 识别不出的图片.jpg
  python scripts/debug_detect.py --source 识别不出的图片.jpg --weights backend/models/best.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def load_constants():
    """与线上 main.py 完全一致地加载类别与阈值"""
    try:
        from backend.constants import PEST_NAMES as NAMES, CLASS_CONF_THRESHOLDS
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from backend.constants import PEST_NAMES as NAMES, CLASS_CONF_THRESHOLDS
    return NAMES, CLASS_CONF_THRESHOLDS


def main() -> int:
    p = argparse.ArgumentParser(description="检测诊断")
    p.add_argument("--source", required=True, help="图片路径")
    p.add_argument("--weights", default="backend/models/best.pt", help="模型权重")
    p.add_argument("--low-conf", type=float, default=0.10,
                   help="诊断用低置信度阈值（尽量不丢候选，默认 0.10）")
    args = p.parse_args()

    NAMES, THRESHOLDS = load_constants()

    try:
        import cv2
        import numpy as np
        from ultralytics import YOLO
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        return 1

    img = cv2.imdecode(np.fromfile(args.source, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print(f"❌ 无法读取图片: {args.source}")
        return 1
    h, w = img.shape[:2]
    print(f"📷 图片: {args.source} ({w}x{h})\n")

    print(f"📦 加载模型: {args.weights}")
    model = YOLO(args.weights)

    # 用低阈值推理，拿到尽可能多的候选
    res = model.predict(img, conf=args.low_conf, imgsz=640, verbose=False)[0]
    boxes = res.boxes
    if boxes is None or len(boxes) == 0:
        print(f"❌ 模型在 conf={args.low_conf} 下仍然 0 个检测")
        print("   → 说明模型本身没识别出害虫（而非阈值过滤）")
        print("     可能：图片太模糊/虫太小/虫种不在 16 类内/角度遮挡严重")
        return 0

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    clss = boxes.cls.cpu().numpy().astype(int)

    print(f"✅ 模型原始检测 {len(confs)} 个（conf >= {args.low_conf}）\n")
    print(f"{'#':>3} | {'类别':<24} | {'置信度':>6} | {'线上阈值':>6} | {'判定':<12} | 框(面积占比)")
    print("-" * 90)

    kept = 0
    for i, (cls_id, conf) in enumerate(zip(clss, confs)):
        name = NAMES[cls_id] if cls_id < len(NAMES) else f"class_{cls_id}"
        thr = THRESHOLDS.get(name, 0.25)
        passed = float(conf) >= thr
        kept += int(passed)
        box = xyxy[i]
        area_ratio = (box[2] - box[0]) * (box[3] - box[1]) / (w * h) * 100
        judge = "✅ 保留" if passed else "❌ 被过滤"
        print(f"{i:>3} | {name:<24} | {float(conf):>6.2f} | {thr:>6.2f} | {judge:<12} | "
              f"[{int(box[0])},{int(box[1])},{int(box[2])},{int(box[3])}] ({area_ratio:.1f}%)")

    print("-" * 90)
    print(f"\n📊 汇总：原始 {len(confs)} 个 → 经线上阈值过滤后剩 {kept} 个")

    # 诊断结论
    print("\n🔎 诊断：")
    if kept == 0:
        print("  → 所有检测都被【类别自适应阈值】过滤掉了！")
        print("    原因：多害虫/密集/小目标场景置信度普遍偏低，而线上阈值 0.40~0.75 偏高。")
        print("    建议：a) 降低阈值（或对密集场景用动态阈值）b) 提高输入分辨率后重新检测")
    elif kept < len(confs):
        print(f"  → 有 {len(confs) - kept} 个被阈值过滤，保留 {kept} 个。")
        print("    若前端仍显示'未知'，还需检查开放集识别逻辑（max_conf<40 或 多目标置信度低）。")
    else:
        print("  → 阈值过滤没有损失任何检测。问题可能在开放集识别或前端展示。")

    # 开放集识别逻辑提示（与 main.py 一致）
    if kept > 0:
        kept_confs = []
        for cls_id, conf in zip(clss, confs):
            name = NAMES[cls_id] if cls_id < len(NAMES) else ""
            if float(conf) >= THRESHOLDS.get(name, 0.25):
                kept_confs.append(float(conf))
        max_conf = (max(kept_confs) if kept_confs else 0.0) * 100  # 转百分比，与 main.py 一致
        print(f"\n⚠️ 开放集识别检查（与线上一致）：")
        print(f"   最高置信度 = {max_conf:.2f}%")
        if max_conf < 40:
            print("   → 最高置信度 < 40% → 会被标记为 low_confidence 未知虫种！")
        elif max_conf < 55 and kept > 1:
            print("   → 多个目标且最高置信度 < 55% → 会被标记为 uncertain_multiple 未知虫种！")
        else:
            print("   → 不会被开放集识别误判，问题应在展示层。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
