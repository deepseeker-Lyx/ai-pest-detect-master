# 🚀 GPU 训练环境配置指南

由于你的项目需要在 GPU 上训练模型，而 Colab 无法使用，以下是国内可用的 GPU 方案。

---

## 方案一：AutoDL（推荐，性价比最高）

> 国内最流行的 GPU 租用平台，**¥0.88 - ¥4/小时**，按量计费，随用随停。

### 操作步骤

#### 1. 注册 & 充值
- 官网：https://www.autodl.com
- 注册后充值 **¥10-20** 足够训练多次

#### 2. 创建实例
```
控制台 → 租用新实例
  ├─ 地区: 选择"北京/上海/杭州"（有货即可）
  ├─ GPU: 推荐 RTX 4090（¥2-3/小时）或 RTX 3090（¥1.5/小时）
  ├─ 镜像: 选择 PyTorch 2.x + CUDA 12.x 官方镜像
  └─ 数据盘: 20GB 足够
```

#### 3. 上传数据集 & 代码
实例创建后，使用 AutoDL 自带的 **JupyterLab** 或 **FileZilla (SFTP)** 上传：

```bash
# 在本地 PowerShell 中（用 SFTP 或 scp）
# 先上传数据集压缩包
# 再上传训练脚本
```

#### 4. 运行训练
在 AutoDL 的 JupyterLab 终端中执行：

```bash
# 解压数据集（如果上传了 ZIP）
unzip "pest dataset(DST1794).zip" -d dataset

# 修改 train_distill.py 中的 DATASET_ZIP 路径
# 然后直接运行
python train_distill.py --device 0 --teacher-epochs 100 --epochs 100
```

#### 5. 下载模型
训练完成后，在 JupyterLab 中下载 `training_output/runs/student_distill_v11n/weights/best.pt`

#### 6. 关机
**记得关机！** AutoDL 不使用时一定要"关机"（不是"停止"），否则继续计费。

---

## 方案二：阿里云 GPU 云服务器（如果你已有阿里云账号）

- 创建 ECS 实例：选择 **GPU 计算型**（如 ecs.gn6i-c4g1.xlarge，含 T4 显卡）
- 大约 **¥10-20/小时**
- 需要自己配置 PyTorch 环境，稍复杂

---

## 方案三：恒源云 / 极客云（国内其他平台）

- **恒源云** (gpuhub.com)：类似 AutoDL，RTX 3090 约 ¥1.5/小时
- **极客云** (jikecloud.net)：有 RTX 4090，约 ¥2.5/小时

---

## 📝 训练脚本使用说明

### 在 GPU 服务器上运行

```bash
# 完整训练（教师 + 蒸馏，约 6-10 小时）
python train_distill.py --device 0

# 使用预训练教师（跳过教师训练，节省 3-5 小时，效果略差）
python train_distill.py --device 0 --use-pretrained-teacher

# 自定义参数
python train_distill.py --device 0 --epochs 150 --teacher-epochs 80 --distill-alpha 0.3

# 仅评估已有模型
python train_distill.py --eval-only /path/to/best.pt
```

### 在本地 CPU 上测试流程（仅验证脚本是否正常）

```bash
# 用小 epoch 数验证流程
python train_distill.py --device cpu --epochs 3 --teacher-epochs 3
```

---

## 💡 推荐做法

| 步骤 | 操作 | 费用 |
|------|------|------|
| 1 | AutoDL 租用 RTX 4090（¥2-3/小时） | ¥2-3 |
| 2 | 上传数据集 + 代码（10 分钟） | — |
| 3 | 训练教师 v11m（约 3 小时） | ¥6-9 |
| 4 | 蒸馏学生 v11n（约 3 小时） | ¥6-9 |
| 5 | 下载 best.pt | — |
| 6 | **关机！** | |
| **总计** | | **约 ¥12-18** |

> 一顿饭的钱就能完成训练，比本地 CPU 跑几天的体验好太多了 🚀
