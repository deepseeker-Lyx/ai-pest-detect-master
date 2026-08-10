"""
=============================================================
YOLO 模型对比训练脚本（论文实验 A：主检测器对比）
水稻害虫检测 — 农作物害虫智能检测系统
=============================================================

用途：补训论文实验 A 缺失的 4 个对照模型，并汇总 6 个模型对比表：
  - 蒸馏 v11n（本文主角，已训好 → 只评估）
  - YOLOv11m 教师（已训好 → 只评估）
  - YOLOv11n 无蒸馏（关键对照，需补训）
  - YOLOv8n / YOLOv8s / YOLOv11s（需补训）

使用方法（AutoDL 上）：
  1. 上传本脚本 + 数据集压缩包到服务器
  2. 确保环境就绪（RTX 5090 需 nightly cu128；A10/其他 Ampere 用稳定版 cu121）
  3. 一键补训 4 个模型 + 评估全部 6 个 + 生成对比表:
     python train_ablation.py --device 0

  4. 仅评估已有权重，生成对比表（模型已训好时用，秒出表）:
     python train_ablation.py --device 0 --eval-only

  5. 只补训部分模型（逗号分隔，keys 见 MODEL_PLAN）:
     python train_ablation.py --device 0 --models plain_v11n,yolov8n

  6. 本地 CPU 小规模验证流程:
     python train_ablation.py --device cpu --epochs 5 --batch 4

数据集路径: E:\\BaiduNetdiskDownload\\pest dataset(DST1794).zip
"""

import os
import sys
import zipfile
import argparse
import shutil
import time
from pathlib import Path

# ═══════════════════════════════════════════════════════════
#  配置区域 — 与 train_distill.py 保持一致
# ═══════════════════════════════════════════════════════════

# 数据集压缩包路径
DATASET_ZIP = r"E:\BaiduNetdiskDownload\pest dataset(DST1794).zip"

# 工作目录（训练产出存放位置）
WORK_DIR = Path(__file__).parent / "training_output"

# 数据集解压目录
DATASET_DIR = WORK_DIR / "dataset"

# 16 类水稻害虫（与项目保持一致）
CLASS_NAMES = [
    "rice leaf roller", "rice leaf caterpillar", "paddy stem maggot",
    "asiatic rice borer", "yellow rice borer", "rice gall midge",
    "Rice Stemfly", "brown plant hopper", "white backed plant hopper",
    "small brown plant hopper", "rice water weevil", "rice leafhopper",
    "grain spreader thrips", "rice shell pest", "grub", "mole cricket",
]

CLASS_NAMES_ZH = [
    "稻纵卷叶螟", "稻毛虫", "稻茎蛆", "二化螟", "三化螟", "稻瘿蚊",
    "稻茎蝇", "褐飞虱", "白背飞虱", "灰飞虱", "水稻象甲", "稻叶蝉",
    "稻蓟马", "稻螟蛉", "蛴螬", "蝼蛄",
]

# ═══════════════════════════════════════════════════════════
#  模型清单（论文实验 A 的 6 个模型）
#  weight_path 指向 run 目录下的 best.pt：
#    - 若存在 → 跳过训练，只评估
#    - 若不存在 → 需要训练（arch 指定预训练骨架，自动下载）
# ═══════════════════════════════════════════════════════════

def _w(key):
    """拼接 run 目录下的 best.pt 路径"""
    return str(WORK_DIR / "runs" / key / "weights" / "best.pt")

MODEL_PLAN = [
    {
        "key": "plain_v11n",
        "label": "YOLOv11n（无蒸馏）",
        "arch": "yolo11n.pt",            # 官方预训练权重（自动下载）
        "run_name": "ablation_plain_v11n",
        "weight_path": _w("ablation_plain_v11n"),
        "need_train": True,
        "note": "关键对照：证明蒸馏的贡献",
    },
    {
        "key": "yolov8n",
        "label": "YOLOv8n",
        "arch": "yolov8n.pt",
        "run_name": "ablation_yolov8n",
        "weight_path": _w("ablation_yolov8n"),
        "need_train": True,
        "note": "上代 nano 基线",
    },
    {
        "key": "yolov8s",
        "label": "YOLOv8s",
        "arch": "yolov8s.pt",
        "run_name": "ablation_yolov8s",
        "weight_path": _w("ablation_yolov8s"),
        "need_train": True,
        "note": "上代 small",
    },
    {
        "key": "yolov11s",
        "label": "YOLOv11s",
        "arch": "yolo11s.pt",
        "run_name": "ablation_yolov11s",
        "weight_path": _w("ablation_yolov11s"),
        "need_train": True,
        "note": "更大模型对照",
    },
    {
        "key": "distill_v11n",
        "label": "蒸馏 v11n（本文）",
        "arch": "yolo11n.pt",
        "run_name": "student_distill_v11n",     # 与 train_distill.py 一致
        "weight_path": None,                      # 由 --distill-weights 指定本地 best.pt
        "need_train": False,
        "note": "教师 v11m → 学生 v11n（复用本地已训权重）",
    },
    {
        "key": "teacher_v11m",
        "label": "YOLOv11m（教师）",
        "arch": "yolo11m.pt",
        "run_name": "teacher_v11m",             # 与 train_distill.py 一致
        "weight_path": _w("teacher_v11m"),
        "need_train": True,                      # 实例已释放，旧教师权重丢失 → 重训
        "note": "参考上限（20M）",
    },
]


# ═══════════════════════════════════════════════════════════
#  第一步：解压数据集 + 创建 data.yaml
# ═══════════════════════════════════════════════════════════

def extract_dataset():
    """解压 DST1794 数据集到工作目录"""
    if DATASET_DIR.exists():
        print(f"✅ 数据集已存在: {DATASET_DIR}")
        return True

    if not os.path.exists(DATASET_ZIP):
        print(f"❌ 数据集压缩包不存在: {DATASET_ZIP}")
        print("请修改脚本顶部的 DATASET_ZIP 路径")
        return False

    print(f"📦 正在解压数据集: {DATASET_ZIP}")
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(DATASET_ZIP, "r") as zf:
        zf.extractall(WORK_DIR)

    if DATASET_DIR.exists():
        for split in ["train", "valid", "test"]:
            img_dir = DATASET_DIR / "images" / split
            if img_dir.exists():
                print(f"   📸 {split}: {len(list(img_dir.glob('*.*')))} 张图片")
        print("✅ 解压完成！")
        return True
    else:
        print("⚠️ 未找到 dataset/ 目录，请检查压缩包结构")
        return False


def create_data_yaml():
    """创建 YOLO 训练所需的 data.yaml"""
    yaml_path = DATASET_DIR / "data.yaml"
    if yaml_path.exists():
        print(f"✅ data.yaml 已存在: {yaml_path}")
        return yaml_path

    img_dir = DATASET_DIR / "images"
    has_train = (img_dir / "train").exists()
    has_val = (img_dir / "valid").exists() or (img_dir / "val").exists()
    has_test = (img_dir / "test").exists()
    val_dir = "valid" if (img_dir / "valid").exists() else "val" if (img_dir / "val").exists() else "train"

    yaml_content = f"""# DST1794 — 16 类水稻害虫对比训练配置（实验 A）
train: {"images/train" if has_train else ""}
val: images/{val_dir}
{"test: images/test" if has_test else "# test: (无测试集)"}

nc: {len(CLASS_NAMES)}
names:
"""
    for i, (en, zh) in enumerate(zip(CLASS_NAMES, CLASS_NAMES_ZH)):
        yaml_content += f"  {i}: {en}  # {zh}\n"

    yaml_path.write_text(yaml_content, encoding="utf-8")
    print(f"✅ data.yaml 已创建: {yaml_path}")
    return yaml_path


# ═══════════════════════════════════════════════════════════
#  第二步：训练单个模型（统一超参数，保证公平对比）
# ═══════════════════════════════════════════════════════════

def train_one(cfg, args):
    """训练一个对照模型。cfg 来自 MODEL_PLAN。"""
    from ultralytics import YOLO

    print("\n" + "=" * 60)
    print(f"🏋️  训练: {cfg['label']}  (run: {cfg['run_name']})")
    print("=" * 60)

    model = YOLO(cfg["arch"])

    # ⚠️ 与蒸馏学生完全一致的超参数，保证公平
    results = model.train(
        data=str(DATASET_DIR / "data.yaml"),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
        workers=args.workers,
        project=str(WORK_DIR / "runs"),
        name=cfg["run_name"],
        exist_ok=True,
        verbose=True,
        close_mosaic=args.epochs // 2,   # 后半段关闭 Mosaic（与蒸馏一致）
        cos_lr=True,                      # 余弦学习率
        warmup_epochs=3,
        lr0=0.001,
        lrf=0.01,
    )

    return cfg["weight_path"] if os.path.exists(cfg["weight_path"]) else None


# ═══════════════════════════════════════════════════════════
#  第三步：评估模型（加载 best.pt → model.info() + val）
# ═══════════════════════════════════════════════════════════

def evaluate_one(cfg, args):
    """加载 best.pt，取参数量/FLOPs/体积 + 指标，返回结果 dict"""
    from ultralytics import YOLO

    wp = cfg["weight_path"]
    result = {
        "label": cfg["label"],
        "note": cfg["note"],
        "weight": wp,
        "exists": os.path.exists(wp),
    }

    if not result["exists"]:
        result["error"] = "权重不存在"
        return result

    model = YOLO(wp)

    # 参数量 / FLOPs / 模型大小
    try:
        info = model.info()  # (n_layers, n_params, n_gradients, GFLOPs)
        result["params"] = info[1]
        result["gflops"] = info[3]
    except Exception as e:
        result["params"] = "N/A"
        result["gflops"] = "N/A"

    result["size_mb"] = round(os.path.getsize(wp) / 1024 / 1024, 2)

    # 统一在验证集上评估（公平对比）
    try:
        val = model.val(
            data=str(DATASET_DIR / "data.yaml"),
            imgsz=args.imgsz,
            device=args.device,
            project=str(WORK_DIR / "runs"),
            name=f"eval_{cfg['key']}",
            exist_ok=True,
        )
        rd = val.results_dict
        result["mAP50"] = rd.get("metrics/mAP50(B)")
        result["mAP50-95"] = rd.get("metrics/mAP50-95(B)")
        result["P"] = rd.get("metrics/precision(B)")
        result["R"] = rd.get("metrics/recall(B)")
    except Exception as e:
        result["val_error"] = str(e)

    return result


# ═══════════════════════════════════════════════════════════
#  第四步：生成论文对比表
# ═══════════════════════════════════════════════════════════

def build_report(results, args):
    """把 6 个模型的评估结果汇总成 markdown 表格"""
    lines = []
    lines.append("# 📊 论文实验 A：主检测器对比结果\n")
    lines.append(f"> 数据集: DST1794（16 类水稻害虫）| 验证集统一评估 | imgsz={args.imgsz}\n")
    lines.append("| 模型 | 参数量 | FLOPs(G) | 体积(MB) | mAP@0.5 | mAP@0.5:0.95 | P | R | 备注 |")
    lines.append("|------|--------|----------|----------|---------|-------------|-----|-----|------|")

    for r in results:
        if not r.get("exists"):
            lines.append(f"| {r['label']} | ❌ 未训练 | - | - | - | - | - | - | {r['note']} |")
            continue

        pct = lambda v: f"{v:.2%}" if isinstance(v, (int, float)) else "N/A"
        params = r.get("params", "N/A")
        if isinstance(params, (int, float)):
            params = f"{params/1e6:.2f}M"
        gflops = r.get("gflops", "N/A")
        if isinstance(gflops, (int, float)):
            gflops = f"{gflops:.1f}"

        lines.append(
            f"| {r['label']} | {params} | {gflops} | {r.get('size_mb', 'N/A')} "
            f"| {pct(r.get('mAP50'))} | {pct(r.get('mAP50-95'))} "
            f"| {pct(r.get('P'))} | {pct(r.get('R'))} | {r['note']} |"
        )

    report = "\n".join(lines) + "\n"
    return report


# ═══════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="论文实验 A：主检测器对比训练")
    parser.add_argument("--device", default="0",
                        help="设备: '0'=GPU, 'cpu'=CPU")
    parser.add_argument("--epochs", type=int, default=100,
                        help="训练轮数 (默认: 100，与蒸馏一致)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="输入分辨率 (默认: 640)")
    parser.add_argument("--batch", default="auto",
                        help="Batch size (默认: auto)")
    parser.add_argument("--patience", type=int, default=20,
                        help="早停耐心值 (默认: 20)")
    parser.add_argument("--workers", type=int, default=4,
                        help="数据加载线程数 (默认: 4)")
    parser.add_argument("--models", default="",
                        help="只补训指定模型（逗号分隔 key，默认全部待训模型）")
    parser.add_argument("--eval-only", action="store_true",
                        help="仅评估已有权重生成对比表，不训练")
    parser.add_argument("--distill-weights", default="",
                        help="本地蒸馏 v11n 的 best.pt 路径（实例释放后蒸馏权重需手动指定）")
    args = parser.parse_args()

    print("=" * 60)
    print("🌾 论文实验 A：主检测器对比训练")
    print("=" * 60)
    print(f"   工作目录: {WORK_DIR}")
    print(f"   设备: {args.device} | Epochs: {args.epochs} | imgsz: {args.imgsz}")
    print("=" * 60)

    # 环境信息
    try:
        import torch
        print(f"   PyTorch: {torch.__version__} | CUDA可用: {torch.cuda.is_available()}"
              + (f" | 设备数: {torch.cuda.device_count()}" if torch.cuda.is_available() else ""))
    except ImportError:
        print("   ⚠️ 未安装 PyTorch，请先: pip install torch")

    # 数据集准备
    if not extract_dataset():
        sys.exit(1)
    create_data_yaml()

    # 蒸馏 v11n 权重：优先用 --distill-weights 指定的本地 best.pt
    if args.distill_weights:
        for cfg in MODEL_PLAN:
            if cfg["key"] == "distill_v11n":
                cfg["weight_path"] = os.path.abspath(args.distill_weights)
                print(f"🎯 蒸馏 v11n 使用指定权重: {cfg['weight_path']}")
                break

    # 选定要训练的模型
    to_train = [c for c in MODEL_PLAN if c["need_train"]]
    if args.models:
        selected = {k.strip() for k in args.models.split(",") if k.strip()}
        to_train = [c for c in to_train if c["key"] in selected]

    # 训练（跳过权重已存在的模型，支持断点续跑）
    if not args.eval_only:
        for cfg in to_train:
            if os.path.exists(cfg["weight_path"]):
                print(f"⏭️  已存在，跳过训练: {cfg['label']}")
                continue
            train_one(cfg, args)

    # 评估全部模型（存在的）
    print("\n" + "=" * 60)
    print("📊 统一评估全部模型（验证集）")
    print("=" * 60)
    results = []
    for cfg in MODEL_PLAN:
        r = evaluate_one(cfg, args)
        if r.get("exists"):
            print(f"   ✅ {r['label']}: mAP@0.5={r.get('mAP50')}")
        else:
            print(f"   ⬜ {r['label']}: 未训练，跳过")
        results.append(r)

    # 生成对比表
    report = build_report(results, args)
    print("\n" + "=" * 60)
    print(report)
    print("=" * 60)

    report_path = WORK_DIR / "ablation_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"✅ 对比表已保存: {report_path}")


if __name__ == "__main__":
    main()
