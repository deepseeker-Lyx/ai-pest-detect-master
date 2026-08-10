# AutoDL RTX 5090 环境搭建（Blackwell / sm_120）—— 排障结论与一键流程

> 日期: 2026-08-03 | 适用: AutoDL RTX 5090 实例（Python 基础镜像 3.12）
> 目的: 训练 `train_ablation.py` 论文实验 A 前，装好 PyTorch (cu128) + Ultralytics

## ⚠️ 踩过的坑（务必先读）

1. **AutoDL 基础镜像自带 torch 2.5.1+cu124（stable）不支持 RTX 5090 (sm_120)**
   - 报错: `NVIDIA GeForce RTX 5090 with CUDA capability sm_120 is not compatible`
2. **官方 nightly cu128 已移除 Python 3.12 (cp312) wheel**，只剩 cp313/cp314
   - `pip install torch --index-url .../cu128` 在 py312 下报 `from versions: none`
   - 必须用 **Python 3.13** 环境
3. **上交镜像 `mirror.sjtu.edu.cn` 有 cp312 的 torch**（历史残留）但**没有 cp312 的 torchvision**
   - 所以 py312 + nightly 方案不完整，别走这条路
4. **各镜像对 pip simple index 的支持情况**:
   - 上交 `https://mirror.sjtu.edu.cn/pytorch-wheels/nightly/cu128` ✅ 有 cp313 全套（torch/torchvision/torchaudio），约 1.27MB/s，**支持 Range 可多线程**
   - 华为云 `mirrors.huaweicloud.com/pytorch/wheels/...` ❌ 返回 SPA 门户页，非 pip 索引
   - 清华 `mirrors.tuna.tsinghua.edu.cn` ❌ 连接被阻（HTTP 返回空）
   - 阿里云 `mirrors.aliyun.com/pytorch-wheels/` ❌ 无 cu128 目录
   - 官方 `download.pytorch.org` ⚠️ 能连但慢（~200kB/s），且无 cp312
5. **AutoDL 默认阿里云 pypi 源没有 ultralytics**（`from versions: none`）
6. 实例可能有瞬时 DNS 故障，重试即可

## ✅ 下次开机一键流程（约 15-20 分钟）

```bash
# ① 创建 Python 3.13 环境
conda create -n py313 python=3.13 -y
conda activate py313

# ② 安装 PyTorch nightly cu128（从上交镜像，cp313 全套版本匹配）
pip install --pre torch torchvision torchaudio \
    --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/nightly/cu128

# 若上交 pip 索引解析失败(302问题)，退路：官方源（有 cp313 全套，但慢）
# pip install --pre torch torchvision torchaudio \
#     --index-url https://download.pytorch.org/whl/nightly/cu128

# ③ 验证 5090 可用
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# ④ 安装 ultralytics（阿里云源没有，用上交 pypi）
pip install ultralytics -i https://mirror.sjtu.edu.cn/pypi/web/simple
# 备选: -i https://pypi.org/simple  (官方源，慢但可用)

# ⑤ 运行论文实验 A（自动跳过已训模型）
python train_ablation.py --device 0 --distill-weights ~/autodl-tmp/distill_v11n.pt
# 先小规模验证: python train_ablation.py --device 0 --epochs 5 --distill-weights ~/autodl-tmp/distill_v11n.pt
```

## 📦 上传清单（scp）

```bash
# 本地 PowerShell:
scp train_ablation.py root@<地址>:~/autodl-tmp/
scp backend/models/best.pt root@<地址>:~/autodl-tmp/distill_v11n.pt
# 数据集压缩包（如需）:
scp "E:\BaiduNetdiskDownload\pest dataset(DST1794).zip" root@<地址>:~/autodl-tmp/
```

## 🧠 多线程下载加速（若 1.27MB/s 太慢）

上交支持 HTTP Range，可用 Python requests 8 线程分块下载 torch wheel：
- torch: `torch-2.12.0.dev20260408+cu128-cp313-cp313-manylinux_2_28_x86_64.whl` (~794MB)
- 版本号会随 nightly 更新，先 `curl -sL <index>/torch/ | grep cp313` 获取最新
