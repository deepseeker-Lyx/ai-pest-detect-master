# 📘 农作物害虫智能检测系统 — 实战学习笔记

> 日期：2026-07-29  
> 环境：Windows · Python 3.12 · YOLOv11 · FastAPI

---

## 📋 目录

1. [环境迁移与路径问题](#一环境迁移与路径问题)
2. [服务启动与端口冲突](#二服务启动与端口冲突)
3. [检测接口报错排查](#三检测接口报错排查)
4. [推理引擎 CLAHE 导致的精度问题](#四推理引擎-clahe-导致的精度问题)
5. [不确定检测合并逻辑](#五不确定检测合并逻辑)
6. [云端大模型配置](#六云端大模型配置)
7. [前端 Markdown 渲染优化](#七前端-markdown-渲染优化)
8. [局域网手机访问](#八局域网手机访问)
9. [模型精度优化思路](#九模型精度优化思路)
10. [常用命令速查](#十常用命令速查)

---

## 一、环境迁移与路径问题

### 问题描述
项目从 `C:\Users\陆彦旭\Desktop\post-detect-master` 复制到 `E:\post-detect-master` 后，FAISS 报 Unicode 错误。

### 根因
FAISS 的 C++ 底层库不支持中文路径（`陆彦旭` 含中文字符），在 `faiss.write_index()` 时崩溃。

### 排查手段
```powershell
# 搜索代码中是否有硬编码的 C 盘路径
grep -r "C:\\Users\\陆彦旭" --include="*.py"

# 结果：代码全部使用相对路径 Path(__file__).parent，无需修改
```

### 解决方案
将项目移到**不含中文的路径**（如 `E:\post-detect-master`）。代码中使用相对路径的项目不受影响。

### 关键命令
```powershell
# 复制项目到 E 盘
Copy-Item -Path "C:\Users\陆彦旭\Desktop\post-detect-master" -Destination "E:\" -Recurse
```

### 经验总结
> ✅ 项目代码尽量使用**相对路径**（`Path(__file__).parent`），避免硬编码绝对路径  
> ✅ Windows 开发路径避免中文字符，防止 C++ 底层库兼容问题

---

## 二、服务启动与端口冲突

### 问题描述
启动服务时报错：
```
ERROR: [Errno 10048] error while attempting to bind on address ('0.0.0.0', 8000)
通常每个套接字地址(协议/网络地址/端口)只允许使用一次。
```

### 根因
端口 8000 已被上一个服务进程占用，未正常关闭。

### 排查手段
```powershell
# 查找占用 8000 端口的进程 PID
netstat -ano | findstr :8000

# 输出示例：
# TCP    0.0.0.0:8000    0.0.0.0:0    LISTENING    25424
```

### 解决方案
```powershell
# 强制杀掉占用端口的进程（PID 替换为实际值）
taskkill /PID 25424 /F
```

### 经验总结
> ✅ 启动服务前先检查端口是否被占用  
> ✅ 使用 `netstat -ano | findstr :端口号` 定位占用进程  
> ✅ 使用 `taskkill /PID 进程号 /F` 强制终止

---

## 三、检测接口报错排查

### 问题描述
上传图片检测时返回 **500 Internal Server Error**，错误日志：
```
AttributeError: 'str' object has no attribute 'shape'
```

### 根因
`inference_engine.infer()` 函数类型标注为 `image: np.ndarray`，但 `detect_image` 路由传入了文件路径字符串（`str(upload_path)`）。推理引擎内部调用 `self.cache.get(image)` → `_compute_hash(image)` → `image.shape`，字符串没有 `.shape` 属性。

### 排查手段
1. **查看服务终端日志**：找到完整的 Traceback，定位到报错文件和行号
2. **分析调用链**：
   ```
   detect_image (main.py:270)
     → _run_detect(str(upload_path), ...) (main.py:192)
       → inference_engine.infer(img_input) (inference_engine.py:299)
         → self.cache.get(image) (inference_engine.py:138)
           → self._compute_hash(image) (inference_engine.py:124)
             → image.shape[0]  ← 报错！str 没有 shape
   ```
3. **对比两个路由的传参差异**：
   - `detect_image`：传 `str(upload_path)` ← ❌ 字符串
   - `detect_base64`：传 `img`（numpy 数组）← ✅ 正常

### 解决方案
```python
# 修复前：直接传字符串给推理引擎
if inference_engine is not None:
    result, timeline = inference_engine.infer(img_input)  # 崩溃

# 修复后：字符串走原始 YOLO 路径，numpy 数组走推理引擎
if inference_engine is not None and isinstance(img_input, np.ndarray):
    result, timeline = inference_engine.infer(img_input)
else:
    results = model(img_input)[0]
```

### 经验总结
> ✅ 函数参数类型不匹配是常见 bug，注意检查调用链  
> ✅ 对比不同路由的传参方式可快速定位问题  
> ✅ `isinstance()` 做类型判断可优雅处理多种输入类型

---

## 四、推理引擎 CLAHE 导致的精度问题

### 问题描述
启用推理引擎后，模型检测不到原本能识别的害虫。

### 根因
推理引擎的 `preprocess()` 方法对图片做了 **CLAHE 自适应直方图均衡**，改变了图像的色彩和对比度分布。YOLO 模型是在原始图像分布上训练的，CLAHE 处理后的图像特征偏移导致检测失效。

### 排查手段
```powershell
# 查看推理引擎的预处理代码
code backend\inference_engine.py
# → preprocess() 方法中调用了 cv2.createCLAHE()
```

### 解决方案
**绕过推理引擎，使用原始 YOLO 检测路径**。推理引擎只用于 `detect_base64`（移动端摄像头）场景的缓存加速。

```python
# _run_detect 中的检测逻辑
else:
    results = model(img_input, conf=0.15, augment=True)[0]  # 原始 YOLO 路径
```

同时添加了两个小优化：
- `conf=0.15`：降低置信度阈值，提高召回率
- `augment=True`：启用 TTA（Test-Time Augmentation），多尺度推理综合投票

### 经验总结
> ✅ 图像预处理增强并非万能，可能破坏模型原有的特征分布  
> ✅ 模型精度优化优先考虑**训练阶段**（数据增强、更大模型），而非推理阶段  
> ✅ 降级方案：新的增强模块出问题时能回退到原始逻辑

---

## 五、不确定检测合并逻辑

### 问题描述
当模型对同一目标输出两个不同类别的检测框（不确定是 A 还是 B），应显示"检测到1个目标，有两种可能性"，而不是显示两个独立目标。

### 解决方案

**后端**（`main.py`）：
```python
def _iou(box1, box2):
    """计算两个边界框的 IoU（交并比）"""
    # 重叠区域 / 合并区域

def _merge_uncertain_detections(detections):
    """合并重叠度高 + 置信度接近的检测框"""
    # IoU > 0.5 且 置信度差值 < 20% → 合并
```

**前端**（`app.js`）：
```javascript
function renderDetectionSummary(data) {
  const hasUncertain = data.detections.some(d => d.alternatives?.length > 0);
  if (hasUncertain) {
    return `检测到 ${totalTargets} 个目标，共 ${totalPossibilities} 种可能`;
  }
  return `检测到 ${data.total} 个害虫目标`;
}
```

### 经验总结
> ✅ 后处理逻辑（NMS 后的合并）可显著提升用户体验  
> ✅ 前端需要配合后端的数据结构变化做相应更新

---

## 六、云端大模型配置

### 问题描述
问答功能提示"当前未配置云端大模型"。

### 解决方案
在项目根目录创建 `.env` 文件：

```ini
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-chat
LLM_API_KEY=sk-你的Key
LLM_WIRE_API=chat
LLM_FULL_URL=false
LLM_THINKING=disabled
LLM_REASONING_EFFORT=
```

`.env` 文件被 `main.py` 自动读取并注入环境变量，`knowledge_base.py` 中的 `answer_with_llm()` 函数使用这些配置调用大模型 API。

### 关键命令
```powershell
# 检查 .env 是否存在
Test-Path "E:\post-detect-master\.env"

# 查看 .env 内容
Get-Content "E:\post-detect-master\.env" -Encoding utf8
```

### 经验总结
> ✅ API Key 属于敏感信息，**不要通过 AI 模型传输**，直接编辑本地文件  
> ✅ `.env` 文件已被 `.gitignore` 忽略，不会上传到 GitHub  
> ✅ 修改 `.env` 后需要**重启服务**才能生效

---

## 七、前端 Markdown 渲染优化

### 问题描述
LLM 返回的 Markdown 格式文本（`**粗体**`、`### 标题`、`---` 分隔线）在聊天框中显示为原始标记符号。

### 解决方案
增强 `renderMarkdown()` 函数，将 Markdown 语法转换为 HTML：

```javascript
function renderMarkdown(value) {
  const escaped = escapeHTML(value);
  return escaped
    .replace(/^###\s+(.+)$/gm, '<h4 class="qa-heading">$1</h4>')
    .replace(/^##\s+(.+)$/gm, '<h3 class="qa-heading">$1</h3>')
    .replace(/^#\s+(.+)$/gm, '<h2 class="qa-heading">$1</h2>')
    .replace(/^---+$/gm, '<hr class="qa-hr">')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/^\s*-\s+(.+)$/gm, '<div class="qa-list-item">• $1</div>')
    .replace(/\n/g, '<br>');
}
```

同时添加对应 CSS 样式。

### 缓存问题
修改 JS/CSS 后浏览器可能使用缓存版本，需要：
1. 更新 HTML 中的版本号：`app.js?v=mobile-ux-3` → `app.js?v=mobile-ux-4`
2. 浏览器按 **Ctrl + F5** 强制刷新（清除缓存）

### 经验总结
> ✅ 前端静态文件修改后需要更新版本参数避免缓存  
> ✅ `Ctrl + F5` = 强制刷新（清除缓存），`F5` = 普通刷新  
> ✅ Markdown 渲染可以用简单正则实现，不需要引入第三方库

---

## 八、局域网手机访问

### 步骤
1. **电脑开放端口**：服务已用 `--host 0.0.0.0` 启动，监听所有网络接口
2. **查找电脑 IP**：
   ```powershell
   Get-NetIPConfiguration | Where-Object { $_.IPv4DefaultGateway -ne $null } | Select-Object -ExpandProperty IPv4Address | Select-Object -ExpandProperty IPAddress
   ```
3. **手机浏览器访问**：`http://192.168.x.x:8000`

### 防火墙放行（如果连不上）
```powershell
# 需以管理员身份运行 PowerShell
New-NetFirewallRule -DisplayName "PestDetect 8000" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow
```

或者手动操作：Windows 安全中心 → 防火墙 → 允许应用通过防火墙 → 添加 `python.exe`

### 经验总结
> ✅ `--host 0.0.0.0` 让服务监听所有网卡，不仅是 localhost  
> ✅ 手机和电脑必须在**同一个局域网**（同一个 WiFi）  
> ✅ Windows 防火墙可能阻止外部访问，需要放行端口

---

## 九、模型精度优化思路

当前模型指标：mAP@0.5 = 49.4%，Precision = 41.1%，Recall = 57.5%

### 🟢 简单（不改代码也能试）

| 方案 | 操作 | 预期提升 |
|------|------|---------|
| 降低置信度阈值 | `model(img, conf=0.15)` | Recall↑，但误报↑ |
| TTA 增强推理 | `model(img, augment=True)` | mAP +2~3% |
| 调整 NMS 阈值 | `model(img, iou=0.5)` | 减少重叠框误删 |

### 🟡 中等（需重新训练）

| 方案 | 操作 | 预期提升 |
|------|------|---------|
| 更大模型 | YOLOv11n → v11s/v11m | mAP +8~15% |
| 更长训练 | epochs=300 + cosine LR | mAP +3~5% |
| 更强数据增强 | Mosaic + MixUp | 小目标检测提升 |

### 🔴 困难（需投入大量精力）

| 方案 | 说明 |
|------|------|
| 数据清洗 | 检查标注质量，统一标注标准 |
| 小目标优化 | 更大输入尺寸 imgsz=960，添加 SAHI |
| 类别平衡 | 对样本少的类别做过采样 |
| 换架构 | YOLOv12 / RT-DETR |

### 关键命令
```powershell
# 测试检测效果（带参数）
python -c "from ultralytics import YOLO; model=YOLO('backend/models/best.pt'); r=model('test.jpg', conf=0.15, augment=True)[0]; print(r.boxes)"
```

### 经验总结
> ✅ 模型精度瓶颈通常在**数据质量和模型架构**，而非推理代码  
> ✅ 先做最简单的改进（调阈值 + TTA），再逐步深入

---

## 十、常用命令速查

### 服务管理
```powershell
# 启动服务
e:/post-detect-master/.venv/Scripts/python.exe -m uvicorn backend.main:app --host 0.0.0.0 --port 8000

# 查找端口占用
netstat -ano | findstr :8000

# 杀掉进程
taskkill /PID 进程号 /F
```

### 环境管理
```powershell
# 安装依赖
e:/post-detect-master/.venv/Scripts/python.exe -m pip install -r requirements.txt

# 安装增强模块依赖
e:/post-detect-master/.venv/Scripts/python.exe -m pip install -r requirements-optional.txt

# 测试依赖是否正常
e:/post-detect-master/.venv/Scripts/python.exe -c "import ultralytics; import fastapi; import cv2; print('OK')"
```

### 诊断测试
```powershell
# 直接加载 main.py 看日志（绕过 uvicorn）
e:/post-detect-master/.venv/Scripts/python.exe -c "import backend.main; print('OK')"

# 检查模型文件
Get-Item "E:\post-detect-master\backend\models\best.pt" | Select-Object Length, LastWriteTime

# 查看检测结果图片
Get-ChildItem "E:\post-detect-master\backend\results"
```

### 网络相关
```powershell
# 获取本机局域网 IP
Get-NetIPConfiguration | Where-Object { $_.IPv4DefaultGateway } | Select-Object -ExpandProperty IPv4Address

# 添加防火墙规则（管理员）
New-NetFirewallRule -DisplayName "PestDetect" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow
```

---

## 十一、检测备选项去重

### 问题描述
合并不确定检测后，备选项中出现了**重复的害虫名称**（如两个"稻瘿蚊"），导致展示冗余。

```
检测到 1 个目标，共 5 种可能
  稻秆潜蝇 48.6%
  → 或 稻秆潜蝇 46.6%    ← 重复！同一个害虫出现两次
  → 或 稻水象甲 41.5%
  → 或 稻瘿蚊  38.4%
  → 或 稻瘿蚊  38.3%    ← 重复！
```

### 根因
YOLO 模型对同一位置可能输出**多个高度重叠的检测框**，每个框可能有不同的类别预测。合并函数 `_merge_uncertain_detections()` 在找备选项时，只要 **IoU > 0.5** 且 **置信度差 < 20%** 就全部加入，但没有检查是否已存在相同害虫名称。

```
主框:   稻秆潜蝇 (48.6%)
重叠框1: 稻秆潜蝇 (46.6%)  ← 和主框同一种类
重叠框2: 稻水象甲 (41.5%)
重叠框3: 稻瘿蚊  (38.4%)
重叠框4: 稻瘿蚊  (38.3%)  ← 和框3同一种类
```

### 排查手段
1. **查看前端展示**：发现备选项列表有重复名称
2. **分析合并逻辑**：`_merge_uncertain_detections()` 没有去重
3. **确认数据**：多个 IoU 高的框确实存在，但部分属于同一类别

### 解决方案
在合并后的备选项列表中添加**名称去重**逻辑：

```python
# 去重：同一害虫名称只保留置信度最高的一个
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
```

### 修复效果
```
修复前: 5 种可能（含 2 个重复）
修复后: 3 种可能（无重复）
  稻秆潜蝇 48.6%
  → 或 稻水象甲 41.5%
  → 或 稻瘿蚊  38.4%
```

### 经验总结
> ✅ 后处理逻辑不仅要考虑"合并"，还要考虑"去重"  
> ✅ 重复数据的源头往往不是 bug，而是模型输出特性（多框重叠）  
> ✅ 使用 `set` 记录已见过的名称是最简洁的去重方式  
> ✅ 修复原则：不要改模型输出，只在后处理阶段做过滤

---

## 📌 核心思维导图

```
遇到问题
  │
  ├─ 1. 看错误日志（服务终端 / 浏览器控制台）
  │     └─ 找到 Traceback 的最后一行 ← 关键错误信息
  │
  ├─ 2. 定位到具体文件和行号
  │     └─ 阅读上下文代码，理解业务逻辑
  │
  ├─ 3. 分析调用链（谁调用了谁，传了什么参数）
  │     └─ 对比正常路径和异常路径的参数差异
  │
  ├─ 4. 提出修复方案
  │     ├─ 最小改动原则：只改必要的代码
  │     └─ 降级方案：新功能出问题时能回退
  │
  └─ 5. 验证修复
        ├─ 重启服务，复现场景
        ├─ 检查终端无报错
        └─ 浏览器确认效果
```

---

> 🔑 **一句话总结**：遇到报错先看终端日志 → 定位报错文件和行号 → 分析调用链 → 最小化修复 → 重启验证。
