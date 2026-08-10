# 农作物害虫智能检测系统 — 增强模块设计文档

> 版本：1.0.0  
> 最后更新：2026-07-28

---

## 目录

1. [架构总览](#一架构总览)
2. [模块一：语义向量知识库](#二模块一语义向量知识库-semantickbpy)
3. [模块二：AI 智能防治 Agent](#三模块二ai-智能防治-agent-pest_agentpy)
4. [模块三：推理优化引擎](#四模块三推理优化引擎-inference_enginepy)
5. [集成方式](#五集成方式)
6. [未来展望](#六未来展望)

---

## 一、架构总览

### 1.1 增强后的系统架构

```
┌──────────────────────────────────────────────────────────────┐
│                       客户端（浏览器/手机）                    │
│   HTML + CSS + JS（纯静态）                                  │
│   图片上传 / 摄像头拍照 / 流式问答                            │
└──────────────────────┬───────────────────────────────────────┘
                       │ HTTP (REST API + SSE)
┌──────────────────────▼───────────────────────────────────────┐
│                    后端服务（FastAPI）                         │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  原有接口（保持兼容）                                  │    │
│  │  POST /detect/image    ← 图片检测（旧）               │    │
│  │  POST /detect/base64   ← Base64 检测（旧）           │    │
│  │  POST /qa/ask          ← 知识库问答（旧）             │    │
│  │  POST /qa/ask-stream   ← 流式问答（旧）              │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  新增增强接口（v2）                                   │    │
│  │  POST /detect/analyze  ← 检测+Agent分析（新）        │    │
│  │  GET  /enhanced/status ← 增强模块状态（新）          │    │
│  │  GET  /enhanced/perf   ← 性能报告（新）              │    │
│  │  GET  /enhanced/cache/clear ← 清空缓存（新）         │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  增强模块                                              │    │
│  │                                                        │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────┐   │    │
│  │  │ 语义知识库    │  │ AI Agent     │  │ 推理引擎  │   │    │
│  │  │ semantic_kb  │→│ pest_agent  │→│inference │   │    │
│  │  │ .py          │  │ .py          │  │_engine.py│   │    │
│  │  └──────────────┘  └──────────────┘  └──────────┘   │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

### 1.2 新旧对比

| 维度 | 原有系统 | 增强后系统 |
|------|---------|-----------|
| 知识库检索 | TF-IDF 关键词匹配 | 语义向量检索 + BM25 混合 + Rerank |
| 问答方式 | 一次性回答 | ReAct 多步推理 + Agent 工具调用 |
| 推理性能 | 每次 300-400ms | 缓存命中时 < 1ms，CLAHE 增强 |
| 输出格式 | 纯文本 | 结构化 JSON 报告 |
| 硬件适配 | 固定 CUDA | 自适应 CUDA/MPS/ONNX/CPU |
| 可观测性 | 无 | 每步耗时分析 + 性能报告 |

---

## 二、模块一：语义向量知识库 (`semantic_kb.py`)

### 2.1 设计背景

原有知识库使用 TF-IDF 关键词匹配检索，存在以下不足：

- **语义鸿沟**：搜索"叶子卷了"找不到"稻纵卷叶螟"
- **同义词盲区**："防治"和"治理"被视为不同词汇
- **排序粗糙**：仅按词频打分，缺乏语义相关性排序

### 2.2 技术方案

#### 2.2.1 三层检索架构

```
用户问题
    │
    ├── ① 语义检索（FAISS + sentence-transformers）
    │     将问题和知识块都转为向量（768维）
    │     用余弦相似度检索 Top-N
    │     理解语义："叶子卷了" ≈ "稻纵卷叶螟"
    │
    ├── ② BM25 关键词检索（rank-bm25）
    │     精确匹配关键词
    │     保证"二化螟 防治"这种精准查询不丢失
    │
    └── ③ RRF 融合 + Cross-Encoder 重排序
           Reciprocal Rank Fusion 合并两个结果集
           Cross-Encoder 对 Top-8 重新打分排序
           保证最相关的结果排在最前面
```

#### 2.2.2 核心类：`SemanticKnowledgeBase`

```
SemanticKnowledgeBase
├── __init__(mode="hybrid")
│   ├── mode="semantic" → 仅语义检索
│   ├── mode="hybrid"   → 语义 + 关键词混合（推荐）
│   └── mode="keyword"  → 降级到关键词检索
│
├── search(question, pest_name=None, limit=4)
│   └── 统一检索入口，根据 mode 自动路由
│
├── pest_info(pest_name)
│   └── 获取害虫详细信息
│
└── incremental_update()
    └── 增量更新索引（新增 JSON 后调用）
```

#### 2.2.3 使用的关键技术

| 技术 | 用途 | 替代方案 |
|------|------|---------|
| `sentence-transformers` | 语义嵌入模型 | 可用 BGE / text2vec 等替换 |
| `FAISS (IndexFlatIP)` | 向量相似度检索 | 可用 ChromaDB / Milvus 替换 |
| `rank-bm25` | 关键词检索 | 可用 Elasticsearch 替换 |
| `Cross-Encoder` | 结果重排序 | 可用 Cohere Rerank 替换 |

#### 2.2.4 索引构建流程

```
JSON 知识文件 (knowledge/pests/*.json)
    │
    ▼
分块（按 Section：特征/症状/发生/防治/问答）
    │
    ▼
向量化（sentence-transformers → 768维向量）
    │
    ▼
存入 FAISS 索引 + 持久化到磁盘
    │
    ▼
运行时加载到内存（启动约 2 秒）
```

### 2.3 创新点

1. **混合检索（Hybrid Search）**：语义 + 关键词双重保障，既懂意图又保精度
2. **重排序 Pipeline（Rerank）**：Cross-Encoder 二次排序，业界 RAG 2.0 标准做法
3. **增量更新**：新增害虫资料时无需重建全量索引
4. **优雅降级**：未安装依赖时自动降级为 TF-IDF，不影响系统运行

---

## 三、模块二：AI 智能防治 Agent (`pest_agent.py`)

### 3.1 设计背景

原有问答系统是一次性"检索 → 拼接 → 返回"，流程简单但存在以下问题：

- **缺乏推理能力**：不能根据置信度、害虫种类综合判断
- **输出单一**：只有文本摘要，没有结构化诊断
- **无状态**：每次问答都是独立请求，没有上下文记忆
- **无工具调用**：只能查知识库，不能进行多步分析

### 3.2 技术方案

#### 3.2.1 ReAct 推理框架

```
用户输入："稻纵卷叶螟怎么防治？"
    │
    ▼
┌─────────────────────────────────┐
│  Thought（思考）                 │
│  "用户想知道防治方法，我需要：    │
│   1. 查知识库获取资料            │
│   2. 分析置信度评估严重程度       │
│   3. 生成结构化报告"              │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Action（行动）                  │
│  ├─ tool_search_knowledge()     │
│  ├─ tool_analyze_risk()         │
│  └─ tool_format_report()        │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Observation（观察）             │
│  "知识库查到 3 条建议            │
│   置信度 85%，属于高风险"        │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Final Answer（最终回答）        │
│  DiagnosisReport {               │
│    assessment: "综合评估..."    │
│    analysis: "详细分析..."      │
│    recommendation: [...]        │
│    prevention: [...]            │
│    risk_level: "高"             │
│  }                              │
└─────────────────────────────────┘
```

#### 3.2.2 核心数据模型

```
PestDetection（YOLO 检测结果）
├── name: str          # 英文名
├── zh_name: str       # 中文名
├── confidence: float  # 置信度
├── bbox: list[int]    # 检测框 [x1,y1,x2,y2]
├── symptoms: list[str]
└── prevention: list[str]

DiagnosisReport（Agent 诊断报告）
├── pest_name: str
├── zh_name: str
├── confidence: float
├── assessment: str         # 综合评估（1-2句话）
├── analysis: str           # 详细分析（3-5句话）
├── recommendation: list[str]  # 防治建议
├── prevention: list[str]      # 预防措施
├── risk_level: str         # 风险等级（低/中/高）
└── sources: list[str]      # 知识来源
```

#### 3.2.3 Agent 工具集

| 工具名称 | 功能 | 输入 | 输出 |
|---------|------|------|------|
| `tool_search_knowledge` | 检索知识库 | question, pest_name | 相关资料列表 |
| `tool_analyze_risk` | 风险评估 | confidence, pest_name | 风险等级文字 |
| `tool_format_report` | 生成报告 | detection, contexts | DiagnosisReport |

### 3.3 双模式运行

```
Agent
├── LLM 模式（需要配置 API Key）
│   ├── 调用 GPT-4 / DeepSeek / 通义千问
│   ├── 生成自然语言结构化报告
│   └── 输出 JSON 格式
│
└── 模板模式（无需外部依赖）
    ├── 内置规则引擎
    ├── 从知识库提取关键信息
    └── 拼接结构化文本
```

### 3.4 创新点

1. **ReAct 推理框架**：模拟人类专家的思考过程，而非简单检索
2. **工具调用（Tool-use）**：Agent 可自主调用知识库、风险分析等工具
3. **记忆管理**：区分短期（对话历史）和长期（害虫知识）记忆
4. **结构化输出**：输出四段式报告（诊断→分析→建议→预防）
5. **双模式运行**：有 LLM 用 LLM，无 LLM 用模板，永不报错

---

## 四、模块三：推理优化引擎 (`inference_engine.py`)

### 4.1 设计背景

原有推理流程存在以下不足：

- **无缓存**：相同图片重复推理，每次都要 300-400ms
- **无预处理**：光照不足或雾天时检测效果差
- **同步阻塞**：推理过程卡住事件循环，影响并发
- **无可观测性**：不知道每步耗时，无法定位瓶颈

### 4.2 技术方案

#### 4.2.1 三层加速架构

```
                    ┌──────────────────────┐
用户上传图片 ──→    │  ① 感知哈希缓存       │
                    │  dHash + LRU 淘汰     │
                    │  命中 → 直接返回结果   │
                    │  未命中 → 继续         │
                    └──────────────────────┘
                            │ 未命中
                    ┌──────────────────────┐
                    │  ② 图像预处理         │
                    │  CLAHE 自适应直方图    │
                    │  + 自动去雾           │
                    └──────────────────────┘
                            │
                    ┌──────────────────────┐
                    │  ③ 模型推理           │
                    │  自动选后端           │
                    │  CUDA/MPS/ONNX/CPU    │
                    └──────────────────────┘
                            │
                        返回结果
```

#### 4.2.2 感知哈希缓存（Perceptual Hashing）

```
dHash（差异哈希）原理：

原始图片 (640x480)
    │ 转灰度 + 缩放到 9x8
    ▼
9x8 灰度图
    │ 每行相邻像素比较
    │ 右边 > 左边 = 1，否则 = 0
    ▼
72位二进制哈希: "101010011001..."
    │ 汉明距离 < 阈值（8位）→ 认定为相似图片
    ▼
缓存命中 / 未命中

应用场景：
- 同一片稻田反复拍照 → 秒回结果
- 移动端摄像头对同一区域多次拍摄
- 批量处理相似图片
```

#### 4.2.3 自适应后端选择

```python
后端检测逻辑：

if CUDA 可用 → "NVIDIA GPU 推理（最快）"
elif MPS 可用 → "Apple Silicon 推理"
elif ONNX Runtime 可用 → "ONNX 优化推理"
else → "CPU 推理（兜底）"
```

#### 4.2.4 去雾和光照增强

```
CLAHE 自适应直方图均衡化：

输入图片（暗光/逆光）
    │
    ▼
转 LAB 色彩空间
    │ 对 L 通道做 CLAHE
    ▼
增强后的 L 通道
    │ 合并回 LAB 并转 BGR
    ▼
对比度检查
    │ if std < 40 → 触发去雾
    ▼
输出增强图片 → 送入 YOLO 检测
```

### 4.3 核心类

```
AdaptiveInferenceEngine
├── preprocess(image) → 图像增强
├── infer(image) → (result, timeline)
├── get_perf_report() → 性能报告
└── switch_model(new_model) → 热切换模型

PerceptualCache
├── get(image) → (hit, result)
├── put(image, result)
├── hit_rate → 缓存命中率
└── clear()

BatchProcessor
└── submit(image) → 提交到批处理队列
```

### 4.4 创新点

1. **感知哈希缓存（dHash）**：非精确匹配缓存，光照变化也能命中
2. **自适应硬件后端**：自动检测最佳推理设备，零配置
3. **CLAHE 增强 + 自动去雾**：被动适应恶劣环境，不改模型
4. **每步耗时分析**：精确到毫秒的时间线，定位性能瓶颈
5. **LRU 淘汰策略**：缓存满时自动淘汰最久未使用的条目

---

## 五、集成方式

### 5.1 依赖安装

```bash
# 增强模块依赖（可选，不装则自动降级）
pip install -r requirements-optional.txt
```

### 5.2 代码集成

在 `main.py` 中：

```python
# 自动检测并加载增强模块
try:
    from backend.semantic_kb import create_knowledge_base as create_semantic_kb
    from backend.pest_agent import create_agent, detection_to_report, PestDetection
    from backend.inference_engine import create_engine
    _HAS_NEW_MODULES = True
except ImportError:
    _HAS_NEW_MODULES = False
```

### 5.3 新增 API 路由

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/enhanced/status` | 查看增强模块状态 |
| `POST` | `/detect/analyze` | 检测 + Agent 分析一步完成 |
| `GET` | `/enhanced/perf` | 推理引擎性能报告 |
| `GET` | `/enhanced/cache/clear` | 清空推理缓存 |

### 5.4 优雅降级机制

```
依赖缺失 → 打印提示 → 使用原始版本 → 完全不影响原有功能

新模块初始化失败 → 捕获异常 → semantic_kb = None → 跳过增强路由
```

---

## 六、未来展望

### 6.1 短期可优化项

| 方向 | 具体方案 | 难度 |
|------|---------|------|
| 向量数据库 | 替换 FAISS 为 Chromadb（支持过滤、混合检索） | ⭐⭐ |
| 图片重排序 | 用 CLIP 做图片级语义检索，用户可搜"长这样的害虫" | ⭐⭐⭐ |
| 多模态 RAG | 知识库中存害虫图片，回答时附带图片参考 | ⭐⭐⭐ |
| WebRTC 实时检测 | 视频流实时检测，替代单张上传 | ⭐⭐⭐⭐ |

### 6.2 长期演进方向

```
当前                         未来
─────────────────────────────────────────────────
TF-IDF 检索          →    多模态 RAG（文本+图片）
单轮问答             →    多轮对话 Agent
单张图片检测         →    视频流实时检测
单一 YOLO 模型       →    模型集成（YOLO + DETR）
CPU 推理             →    TensorRT / CoreML 加速
手动部署             →    Docker + CI/CD 自动部署
```

---

> 本文档对应的源代码位于 `backend/` 目录：
> - `backend/semantic_kb.py` — 语义向量知识库
> - `backend/pest_agent.py` — AI 智能防治 Agent
> - `backend/inference_engine.py` — 推理优化引擎
> - `backend/main.py` — 集成入口（含新增 API 路由）
