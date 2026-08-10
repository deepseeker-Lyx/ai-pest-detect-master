"""
=============================================================
🎯 小目标 / 易混淆类优化训练脚本 — 水稻害虫检测
=============================================================
面向问题：灰飞虱(AP 14.6%)、稻纵卷叶螟(23.5%)、稻蓟马等小目标/密集
场景检测率低。本脚本通过三类手段提升小目标性能，全部与论文基线
(train_ablation.py / train_distill.py) 同超参，保证公平对比：

  ① 提高输入分辨率 imgsz=896（小目标像素更多，特征更清晰）
  ② 小目标数据增强：mosaic + copy-paste（小目标复制粘贴）+ scale 随机缩放
  ③ 可选 P2 小目标检测头（--model yolo11n-p2.yaml，需 ultralytics 支持）

使用方法（GPU 服务器，如 AutoDL RTX 5090 / 学校 A10）：
  1. 上传本脚本 + 数据集压缩包
  2. 基线（公平对照，与消融一致）:
     python train_small_object.py --device 0 --imgsz 640 --name small_baseline_640
  3. 小目标优化（推荐）:
     python train_small_object.py --device 0 --imgsz 896 --name small_opt_896
  4. 更高分辨率（显存足够时）:
     python train_small_object.py --device 0 --imgsz 1024 --name small_opt_1024
  5. 尝试 P2 小目标头（若 ultralytics 支持）:
     python train_small_object.py --device 0 --imgsz 896 --model yolo11n-p2.yaml

  6. 本地 CPU 小规模验证流程:
     python train_small_object.py --device cpu --epochs 3 --batch 4 --imgsz 640

统一超参数（与论文基线一致，保证公平）：
  epochs=100 | cos_lr | warmup=3 | lr0=0.001 | lrf=0.01 | patience=20
  close_mosaic=epochs//2（后半段关闭 mosaic 防小目标遮挡）
数据集路径: E:\\BaiduNetdiskDownload\\pest dataset(DST1794).zip
"""
import argparse
import os
import sys
import time
import zipfile
from pathlib import Path

# ═══════════════════════════════════════════════════════════
#  配置
# ═══════════════════════════════════════════════════════════

DATASET_ZIP = r"E:\BaiduNetdiskDownload\pest dataset(DST1794).zip"
WORK_DIR = Path(__file__).parent / "training_output"
DATASET_DIR = WORK_DIR / "dataset"

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


def parse_args():
    p = argparse.ArgumentParser(description="小目标/易混淆类优化训练")
    p.add_argument("--model", default="yolo11n.pt",
                   help="模型结构：yolo11n.pt（默认）/ yolo11n-p2.yaml（P2 小目标头）")
    p.add_argument("--imgsz", type=int, default=896, help="输入分辨率（小目标推荐 896/1024）")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=-1, help="-1 自动")
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--copy-paste", type=float, default=0.3,
                   help="copy-paste 小目标复制增强强度 0-1（需 ultralytics 支持）")
    p.add_argument("--name", default=None, help="run 名称")
    p.add_argument("--eval", action="store_true", help="训练后自动在验证集评估")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════
#  数据准备
# ═══════════════════════════════════════════════════════════

def extract_dataset():
    if DATASET_DIR.exists():
        print(f"✅ 数据集已存在: {DATASET_DIR}")
        return True
    if not os.path.exists(DATASET_ZIP):
        print(f"❌ 数据集压缩包不存在: {DATASET_ZIP}")
        print("请修改脚本顶部 DATASET_ZIP 路径")
        return False
    print(f"📦 解压数据集: {DATASET_ZIP}")
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(DATASET_ZIP, "r") as zf:
        zf.extractall(WORK_DIR)
    print("✅ 解压完成")
    return DATASET_DIR.exists()


def create_data_yaml():
    yaml_path = DATASET_DIR / "data.yaml"
    if yaml_path.exists():
        print(f"✅ data.yaml 已存在")
        return yaml_path
    img_dir = DATASET_DIR / "images"
    has_train = (img_dir / "train").exists()
    has_test = (img_dir / "test").exists()
    val_dir = "valid" if (img_dir / "valid").exists() else "val" if (img_dir / "val").exists() else "train"
    yaml_content = f"""# DST1794 — 16 类水稻害虫小目标优化训练配置
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
#  训练
# ═══════════════════════════════════════════════════════════

def main():
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ 请先安装 ultralytics")
        sys.exit(1)

    if not extract_dataset():
        sys.exit(1)
    data_yaml = str(create_data_yaml())

    run_name = args.name or f"small_opt_imgsz{args.imgsz}"
    print("\n" + "=" * 60)
    print(f"🎯 小目标优化训练 | model={args.model} | imgsz={args.imgsz} | run={run_name}")
    print("=" * 60)

    model = YOLO(args.model)

    results = model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
        workers=args.workers,
        project=str(WORK_DIR / "runs"),
        name=run_name,
        exist_ok=True,
        verbose=True,
        # ── 与论文基线一致的超参（公平对比） ──
        close_mosaic=args.epochs // 2,
        cos_lr=True,
        warmup_epochs=3,
        lr0=0.001,
        lrf=0.01,
        # ── 小目标增强 ──
        copy_paste=args.copy_paste,
        scale=0.5,          # 随机缩放，模拟不同拍摄距离（小目标多尺度）
        mosaic=1.0,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        flipud=0.0,
    )
    print(f"✅ 训练完成: {results.save_dir}")

    # 可选：自动评估
    if args.eval:
        best = Path(results.save_dir) / "weights" / "best.pt"
        if best.exists():
            print(f"\n📊 评估: {best}")
            val = YOLO(str(best)).val(
                data=data_yaml,
                imgsz=args.imgsz,
                device=args.device,
                project=str(WORK_DIR / "runs"),
                name=f"eval_{run_name}",
                exist_ok=True,
            )
            rd = val.results_dict
            print(f"   mAP@0.5 = {rd.get('metrics/mAP50(B)')}")
            print(f"   mAP@0.5:0.95 = {rd.get('metrics/mAP50-95(B)')}")
            print(f"   Precision = {rd.get('metrics/precision(B)')}")
            print(f"   Recall = {rd.get('metrics/recall(B)')}")
        else:
            print("⚠️ 未找到 best.pt，跳过评估")

    print(f"\n💾 输出目录: {results.save_dir}")


if __name__ == "__main__":
    main()
