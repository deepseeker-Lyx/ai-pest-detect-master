# 📄 论文素材 — 系统工程健壮性（System Robustness）

> **对应论文章节**：第三章 方法 · 3.1 系统总体架构（实现小节）/ 第四章 实验 · **实验 D：系统性能**
> **支撑贡献点**：贡献 3「完整农业诊断系统：检测-问答-天气预警闭环，真实部署验证」+ 贡献 4「工程不确定性管理」
> **整理日期**：2026-08-04（提交 `a15e61f`，已推送 GitHub）
> **作用**：CAE 类期刊看重"应用 + 系统完整性"，本节素材让"系统实现"章节有工程深度，不再只是功能罗列

---

## 一、本次优化的论文价值定位

论文主线是**"感知-认知协同"**，本次后端健壮性优化不属于核心算法创新，而是**工程实现质量**。在 CAE 定位下，它的作用是：

1. **支撑实验 D（系统性能）**：为"CPU 部署下系统可靠、可长时间运行"提供工程依据
2. **强化贡献 3**：一个"能真实部署、能扛并发、能自恢复"的系统，比"只有模型"更有说服力
3. **写作策略**：把工程细节浓缩为**一段系统实现 + 一张系统健壮性表格**，不喧宾夺主

> ⚠️ **写作提醒**：健壮性优化**不要单独成章**（会冲淡算法主线），建议：
> - 方法章 3.1 系统架构下加 **一段"Engineering Design for Field Deployment"**
> - 实验章加 **表（系统健壮性 / 性能指标）**，配 3-5 句说明

---

## 二、六大优化的论文表述素材

### 2.1 异步非阻塞架构（Asynchronous Non-blocking Pipeline）

**工程描述**：
检测与 LLM 调用等 CPU 密集/网络阻塞操作放入线程池（`asyncio.to_thread`），流式 LLM 响应通过后台线程桥接（`_aiter_sync`），避免阻塞事件循环。修复前：单次 LLM 调用（最长 60s）会阻塞整个服务事件循环；修复后：高并发下服务仍可响应。

**论文英文表述（可直接用）**：
> To prevent long-running inference and LLM network calls from blocking the ASGI event loop, all CPU-intensive and network-bound operations are dispatched to a thread pool (`asyncio.to_thread`), and the synchronous streaming generator of the LLM is bridged through a background producer thread. This guarantees service responsiveness under concurrent requests even when a single LLM call takes up to 60 s.

**写作要点**：这里的"不阻塞"是**可用性**指标，可配合实验 D 的并发测试数据（待测，见第四节）。

---

### 2.2 上传安全校验（Input Validation）

**工程描述**：
文件类型白名单 + 文件头魔数校验（JPEG/PNG/BMP/GIF/WEBP）+ 10MB 大小上限。修复前仅校验扩展名，伪装文件可进入推理管线。

**论文英文表述**：
> Uploaded images are validated by both filename extension whitelist and binary magic-number checking, with a 10 MB size limit, rejecting malformed or disguised files before they reach the inference pipeline.

**写作要点**：一句带过即可，放在系统实现段落里作为"部署安全"的一环。可选配一个"拒绝伪装文件"的小实验（请求被 400 拒绝，作为鲁棒性证据）。

---

### 2.3 运行时文件生命周期管理（Storage Lifecycle Management）

**工程描述**：
后台调度线程周期性清理过期文件：检测结果图保留 72 h、上传临时文件保留 24 h、清理间隔 6 h。修复前结果目录无限增长（本次启动即清理 66 个过期文件）。

**论文英文表述**：
> A background scheduler periodically purges expired artifacts: annotated results are retained for 72 h and temporary uploads for 24 h, preventing unbounded disk growth during long-term deployment.

**写作要点**：为实验 D 的"7×24 稳定性"提供证据——长时间运行磁盘不涨。

---

### 2.4 配置与常量集中化（Centralized Configuration）

**工程描述**：
16 类害虫名单、类别置信度阈值、上传/清理限制统一收敛到 `backend/constants.py`，与训练脚本共享，消除多处重复定义的不一致风险。

**论文英文表述**：
> Class names, per-class confidence thresholds, and deployment limits are centralized in a single constants module shared with the training pipeline, eliminating inconsistency between detection and training configurations.

**写作要点**：可复现性（Reproducibility）的一部分——审稿人关心配置一致性。

---

### 2.5 存储层优化（SQLite WAL）

**工程描述**：
SQLite 由"每次操作新建连接"改为**全局单例连接 + WAL 模式 + busy_timeout=15s**，降低连接开销，WAL 提升读写并发。

**论文英文表述**：
> The history store uses a single SQLite connection with Write-Ahead Logging (WAL) and a 15 s busy timeout, reducing connection overhead and improving concurrent read/write throughput.

**写作要点**：历史记录/统计功能支撑论文中"数据闭环"叙述（检测与问答留痕 → 统计 → 预警），一句带过即可。

---

### 2.6 结构化访问日志（Structured Request Logging）

**工程描述**：
HTTP 中间件为每个请求生成唯一 `X-Request-ID`，记录方法/路径/状态码/耗时。修复前无任何请求日志。

**论文英文表述**：
> A middleware assigns a unique request ID to every HTTP request and logs method, path, status code, and latency, enabling traceability and performance auditing in production.

**写作要点**：可观测性（Observability）——支撑"系统已在真实环境部署运行"的可靠性叙述。

---

## 三、可直接引用的量化指标（当前实测值）

| 指标 | 实测值 | 场景/备注 | 论文用途 |
|------|--------|----------|---------|
| 单张推理延迟（合成图） | ~247 ms | 本地 CPU，含预处理/推理/后处理 | 实验 D |
| 流式问答分块数 | 582 chunks | 单次 LLM 流式响应 | 端到端体验 |
| 启动清理过期文件 | 66 个 | 首次运行即清理 | 存储管理证据 |
| 伪装文件拦截 | HTTP 400 | 魔数校验 | 安全鲁棒性 |
| LLM 调用超时上限 | 60 s | 线程池化，不阻塞其他请求 | 并发可用性 |
| 模型/知识库加载 | 全部成功 | 蒸馏 v11n + 80 块语义库 + Agent | 系统完整性 |

---

## 四、待补充的实验数据（写作前必须测 ⭐）

以下数据是论文"系统性能"表格的硬指标，**尚未采集**，需要部署后实测：

### 4.1 异步化前后并发对比（最有说服力）✅ 压测脚本已就绪

**压测脚本**：`benchmark/benchmark_concurrency.py`（零新依赖，ThreadPoolExecutor + requests）
- 支持端点：`detect`（检测）/ `qa`（问答，真实 LLM 调用）/ `all`
- 支持多并发档位（如 `--concurrency 1,5,10,20`），输出 P50/P95/P99/最大延迟、吞吐（req/s）、错误率
- 支持 `--compare` 合并多个结果 JSON，生成对比表 + P95 倍率
- 用法：
  ```bash
  # ① 启动服务
  uvicorn backend.main:app --host 127.0.0.1 --port 8000
  # ② 压测改造后版本
  python benchmark/benchmark_concurrency.py --url http://127.0.0.1:8000 \
      --label after --endpoint detect --concurrency 1,5,20 --requests 10
  python benchmark/benchmark_concurrency.py --url http://127.0.0.1:8000 \
      --label after --endpoint qa --concurrency 1,5,10 --requests 3
  # ③ 改造前版本用另一端口跑同样命令，加 --label before
  # ④ 生成对比表
  python benchmark/benchmark_concurrency.py --compare \
      benchmark/reports/concurrency_after.json benchmark/reports/concurrency_before.json
  ```

**初步验证数据**（本机 CPU，样本量小，仅证明脚本可用；正式论文需更大样本）：
| 端点 | 并发 | P50 | P95 | 吞吐 | 错误率 |
|------|------|-----|-----|------|--------|
| detect | 1 | 180ms | 244ms | 5.3 req/s | 0% |
| detect | 5 | 979ms | 1027ms | 5.1 req/s | 0% |
| qa | 1 | 12543ms | 12543ms | 0.1 req/s | 0% |

> 📌 **解读提示**：
> - `detect` 是 CPU 密集，并发下吞吐受单核推理瓶颈限制（吞吐不涨，但**不阻塞、不报错、服务始终可用**）
> - `qa` 是网络/LLM 阻塞，**异步化收益最明显**——对比 before/after 时重点看 qa 端点在并发下的 P95
> - 正式测量：`--requests` ≥ 10，并发档位 1/5/10/20，最好在服务器 CPU 上测

### 4.2 CPU 部署性能全景（实验 D 核心表）
| 指标 | 数值 | 说明 |
|------|------|------|
| 单张 CPU 推理延迟 | 待测 | best.pt on 阿里云 Xeon |
| 端到端延迟（上传→结果） | 待测 | 含网络 |
| 吞吐（并发下 req/s） | 待测 | |
| 内存占用（常驻） | 待测 | 模型 + 语义模型 + FAISS |
| 7×24 稳定性 | 待测 | systemd 自启 + 崩溃恢复 + 磁盘不涨 |

### 4.3 存储生命周期证据
- 连续运行 N 天，记录 results/ 目录文件数与磁盘占用曲线 → 证明清理机制生效

---

## 五、论文段落模板（可直接改写）

### 5.1 方法章 · 系统实现段落（中文草稿）

> 系统后端采用 FastAPI 构建。为实现田间真实部署所需的可靠性，工程实现上做了四点设计：
> **① 异步非阻塞**：YOLO 推理与 LLM 网络调用放入线程池执行，流式回答通过后台线程桥接，保证并发请求下事件循环不被单个慢操作阻塞；
> **② 输入安全**：上传图片经扩展名白名单与文件头魔数双重校验，并设 10MB 大小上限；
> **③ 存储生命周期**：后台线程周期性清理过期结果与临时文件，防止长时间运行导致磁盘膨胀；
> **④ 可观测性**：请求级唯一 ID 与结构化日志，便于线上问题追踪。
> 检测与问答历史写入 SQLite（WAL 模式）本地持久化，支撑后续统计与预警。

### 5.2 实验章 · 系统性能段落（中文草稿）

> 为验证系统的工程可用性，我们在无 GPU 的 CPU 服务器（Intel Xeon）上进行了并发与稳定性测试。结果表明：单张图片端到端推理约 X ms；在 20 并发下服务仍保持可响应，P95 延迟为 X ms（较未采用异步管线前的基线显著下降）；伪装/非法文件被安全校验以 400 拒绝；连续运行 X 天后结果目录文件数保持稳定（清理机制生效），未出现崩溃或内存泄漏。上述结果说明系统满足基层农业场景对"低资源、可长时间稳定运行"的部署要求。

---

## 六、与论文既有章节的衔接

| 论文位置 | 用这里的什么 | 怎么用 |
|----------|-------------|--------|
| 3.1 系统总体架构 | 2.1-2.6 的工程表述 | 架构图下加"工程实现"一段 |
| 实验 D（系统性能） | 第三节表格 + 第四节待测数据 | 组装成"系统性能表" |
| 第五章 讨论 | 并发/稳定性结果 | 佐证"工程价值"论点 |
| 摘要/贡献 3 | "真实部署验证" | 引用并发与稳定性数据 |

---

## 七、写作节奏建议

```
现在：把本页第四节"待补实验"列入第 1-2 周计划
  □ 并发压测（修复前 vs 修复后）→ 最有说服力的一张表
  □ CPU 部署性能全景（延迟/吞吐/内存）→ 实验 D 核心表
  □ 7×24 稳定性 → 连续运行截图/日志
之后：写方法章 3.1 工程段落（直接用 5.1 草稿）
写实验章 实验 D（用第三节 + 第四节实测值填表）
```

---

*本文档与 `paper-outline.md`、`paper-experiment-plan.md` 配合使用。*
