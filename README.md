# 🌾 农作物害虫智能检测系统（ai-pest-detect-master）

基于 **YOLOv11 知识蒸馏** 的水稻害虫智能检测系统：拍照识别 16 类水稻害虫 → 自动给出防治建议 → 联动天气预警。模型轻量（CPU 可跑），知识蒸馏后 **mAP@0.5 达 74.6%**。

---

## ✨ 功能特性

| 特性 | 说明 |
|------|------|
| 🖼️ **害虫识别** | 图片上传 / 手机拍照，识别 16 类水稻害虫并标注位置与置信度 |
| 🧠 **轻量高效** | 知识蒸馏模型（2.6M 参数），无 GPU 也能流畅部署 |
| 💬 **智能问答** | RAG + 大模型，流式打字机回答；支持**多害虫分别回答** |
| 🛡️ **开放集识别** | 对未知虫种"拒识"提示，不强行归类 |
| 🌤️ **天气预警** | 结合当地天气评估虫害风险等级 |
| 📊 **历史统计** | 检测与问答记录留存，支持统计与趋势 |
| 🐳 **一键部署** | Docker Compose 一条命令部署到服务器 |

---

## 📂 目录结构与文件说明

```
ai-pest-detect-master/
│
├── 🖥 backend/                        # FastAPI 后端（核心服务）
│   ├── main.py                        # 应用入口：创建 app / 挂载路由 / 后台加载模型 / 健康检查
│   ├── routes_detect.py               # 检测路由 /detect/*（上传 / Base64 / 增强检测 + 开放集识别）
│   ├── routes_qa.py                   # 问答路由 /qa/*（普通 / 流式 / 多害虫分别回答）
│   ├── routes_history.py              # 历史路由 /history/*（检测 / 问答记录 + 统计）
│   ├── routes_weather.py              # 天气路由 /weather/*（实时天气 + 虫害预警）
│   ├── routes_enhanced.py             # 增强路由 /enhanced/*（语义库 / Agent / 推理引擎状态）
│   ├── config.py                      # 路径配置（模型 / 上传 / 结果 / 前端目录）
│   ├── state.py                       # 全局运行时状态（模型 / 引擎 / Agent 共享）
│   ├── schemas.py                     # Pydantic 数据模型（请求 / 响应结构）
│   ├── constants.py                   # 共享常量（16 类名单 / 置信度阈值 / 上传限制）
│   ├── inference_engine.py            # 推理加速：CLAHE 增强 + 伪装色增强 + 感知哈希缓存
│   ├── knowledge_base.py              # 知识检索 + 大模型调用（流式 / 非流式）
│   ├── semantic_kb.py                 # 语义知识库：FAISS 向量 + BM25 混合检索
│   ├── pest_agent.py                  # AI 防治助手：生成诊断 / 分析 / 防治报告
│   ├── weather_service.py             # 天气获取 + 按天气评估虫害风险
│   ├── storage.py                     # SQLite 历史存储（检测 / 问答 / 统计）
│   ├── models/                        # 模型权重（best.pt 蒸馏模型，未纳入 Git）
│   ├── uploads/ · results/            # 运行时临时上传 / 检测结果图（自动生成）
│   └── pest_data.db                   # 运行数据库（自动生成）
│
├── 🎨 frontend/                       # Web 前端（移动优先）
│   ├── templates/
│   │   ├── index.html                 # 主页面（识别 + 聊天问答界面）
│   │   └── loading.html               # 品牌加载页（入场动画）
│   └── static/
│       ├── css/style.css              # 全部样式（设计系统 / 毛玻璃 / 进度条）
│       ├── js/app.js                  # 前端逻辑（上传 / 检测 / 流式问答 / 多害虫 / 天气）
│       └── img/logo.png               # 系统 Logo
│
├── 📚 knowledge/                      # 农业知识库
│   ├── pests/*.json                   # 16 类害虫知识（识别特征 / 危害症状 / 防治 / 问答）
│   └── vector_store/                  # 语义向量索引（FAISS，可自动重建）
│
├── 🔬 scripts/                        # 工具脚本
│   ├── debug_detect.py                # 检测诊断：定位"识别不出"的原因
│   ├── multiscale_detect.py           # 多尺度 / 滑窗推理（小目标增强）
│   ├── enhance_knowledge_base.py      # 知识库数据增强生成
│   └── test_model.py                  # 模型加载自检（服务器用）
│
├── 📊 benchmark/                      # 性能评测
│   └── benchmark_concurrency.py       # 并发压测（P50/P95/吞吐/错误率）
│
├── 📖 docs/                           # 项目文档
│   ├── design/                        # 设计与实现（含学习指南、项目结构速查表）
│   ├── training/                      # 训练与部署（训练记录、GPU 指南、Docker 部署）
│   ├── paper/                         # 论文素材（大纲 / 实验设计 / 健壮性 / 小目标）
│   └── _archive/                      # 历史备份（旧报告、可视化文档）
│
├── 🎓 训练脚本（GPU 服务器用）
│   ├── train_distill.py               # 知识蒸馏训练（教师 v11m → 学生 v11n）
│   ├── train_ablation.py              # 论文消融对比训练（6 个模型统一超参）
│   └── train_small_object.py          # 小目标优化训练（高分辨率 + copy-paste 增强）
│
├── 🐳 部署配置
│   ├── Dockerfile                     # Docker 镜像构建（CPU 推理）
│   ├── docker-compose.yml             # 一键编排部署（端口 / 数据卷 / 自恢复）
│   ├── deploy_server.sh               # 服务器传统方式一键部署脚本
│   ├── DEPLOY_SERVER.md               # 服务器部署文档
│   ├── start.sh · start.ps1           # 本地启动脚本（Linux / Windows）
│   └── requirements*.txt              # 依赖清单（主 / 服务端 / CPU torch / 可选）
│
└── 📄 根文档
    ├── README.md                      # 本说明
    └── .env.example                   # 环境变量模板（LLM 大模型配置）
```

---

## 🚀 快速开始（本地）

```bash
# 1. 安装依赖（CPU 版 PyTorch）
pip install -r requirements.txt

# 2. 可选：安装增强模块（语义知识库 / Agent / LLM）
pip install -r requirements-optional.txt

# 3. 配置大模型（可选，不配则用本地知识库）
#    复制 .env.example 为 .env，填入 LLM_API_KEY 等

# 4. 启动
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

访问：**http://localhost:8000**（局域网用 `http://<本机IP>:8000`）

> 💡 也可用 `bash start.sh`（Linux）或 `start.ps1`（Windows）一键启动。

---

## 🐳 Docker 部署（推荐）

```bash
# 1. 放置模型权重
mkdir -p backend/models && scp best.pt root@服务器:~/post-detect/backend/models/

# 2. 构建并启动（首次约 5-15 分钟）
docker compose up -d --build

# 3. 验证
curl http://服务器IP:8000/health   # → {"status":"ok","model":"best.pt"}
```

详细步骤见 `docs/training/DOCKER_DEPLOY.md`。

---

## 📚 文档索引

| 入口 | 内容 |
|------|------|
| `docs/design/project-structure.md` | 项目结构 + **全部文件作用速查表** |
| `docs/design/LEARNING_GUIDE.md` | 项目完全掌握指南（知识地图 + 一次请求的旅程） |
| `docs/paper/` | 论文素材（大纲 / 实验 / 健壮性 / 小目标） |
| `docs/training/` | 训练与部署（训练记录 / GPU 指南 / Docker） |

---

## ❓ 常见问题

**Q: 找不到 `best.pt`？**  
A: 确认 `backend/models/best.pt` 存在；不存在则将训练好的权重复制到该路径。

**Q: 手机打不开页面？**  
A: 手机与电脑连同一局域网，用电脑局域网 IP（非 localhost）访问。

**Q: 识别不出的图怎么排查？**  
A: 运行 `python scripts/debug_detect.py --source 图片.jpg`，判断是"阈值过滤"还是"模型未识别"。

