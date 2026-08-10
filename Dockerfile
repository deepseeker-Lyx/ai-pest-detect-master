# =============================================================
#  农作物害虫智能检测系统 — 生产镜像（CPU 推理）
#  目标环境：无 GPU 的云服务器（阿里云轻量 / 普通 Xeon）
#  说明：
#    - 使用 CPU 版 PyTorch（见 requirements.txt 固定 wheel）
#    - 模型权重 *.pt 不打包进镜像，通过 volume 挂载（体积大且需按需更新）
#    - 知识库 pests JSON 与 FAISS 索引随镜像打包（保持开箱即用）
# =============================================================
FROM python:3.11-slim

# ── 环境变量 ─────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ── 系统依赖（opencv 运行时 + 编译工具兜底） ─────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── 先装依赖（利用 Docker 层缓存，代码改动不重装依赖） ──────
COPY requirements.txt requirements-server.txt requirements-optional.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r requirements-optional.txt

# ── 拷贝项目代码 ─────────────────────────────────────────────
COPY backend/ backend/
COPY frontend/ frontend/
COPY knowledge/ knowledge/
COPY start.sh ./

# ── 端口与启动 ───────────────────────────────────────────────
EXPOSE 8000

# 模型与运行时数据通过 volume 挂载（见 docker-compose.yml）：
#   backend/models（best.pt）、backend/uploads、backend/results
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
