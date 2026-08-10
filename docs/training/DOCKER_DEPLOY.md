# 🐳 Docker 化部署指南

> 目标：用 Docker Compose 在服务器上**一条命令**部署整套系统，替代手动装依赖/激活虚拟环境。
> 适用：阿里云轻量服务器等**无 GPU 的 CPU 服务器**（Ubuntu + Docker）。
> 论文用途：支撑实验 D"系统性能"——部署可复现性 + 7×24 稳定性（`restart: unless-stopped` 自恢复）。

---

## 一、涉及文件

| 文件 | 作用 |
|------|------|
| `Dockerfile` | 生产镜像（python:3.11-slim + CPU PyTorch + 全部依赖 + 代码） |
| `docker-compose.yml` | 编排：端口映射、数据卷挂载、健康检查、自恢复 |
| `.dockerignore` | 构建上下文排除（.venv/.git/模型/数据集/文档），缩小镜像 |

---

## 二、服务器准备（首次）

```bash
# 1. 安装 Docker（官方脚本）
curl -fsSL https://get.docker.com | bash -s docker
sudo systemctl enable --now docker

# 2. 可选：国内加速镜像（阿里云容器镜像加速器）
#    在 /etc/docker/daemon.json 配置 registry-mirrors 后 sudo systemctl restart docker
```

## 三、获取代码与模型

```bash
# 1. 拉取代码
cd ~
git clone https://github.com/deepseeker-Lyx/ai-pest-detect-master.git post-detect
cd post-detect

# 2. 放置模型权重（⚠️ best.pt 不在 Git 仓库，必须手动放）
#    本地执行：scp backend/models/best.pt root@服务器IP:~/post-detect/backend/models/
mkdir -p backend/models
# → 确认 backend/models/best.pt 存在
```

## 四、配置大模型（可选，不配则用本地知识库降级）

项目根目录已有 `.env` 文件（若存在则自动生效）。没有则手动创建：

```bash
cat > .env << 'EOF'
LLM_API_KEY=你的key
LLM_BASE_URL=https://api.xxx.com/v1
LLM_MODEL=你的模型名
LLM_WIRE_API=chat          # 或 responses
EOF
```

> `docker-compose.yml` 里通过 `env_file: .env` 注入；`.env` 已被 `.dockerignore` 排除，不会打包进镜像。

## 五、构建并启动

```bash
docker compose up -d --build
```

首次构建会下载 Python 镜像 + 依赖（CPU 版 PyTorch 约 200MB + 语义模型依赖），约 5-15 分钟。

## 六、验证

```bash
# 查看启动日志（应看到"增强模块加载完成"）
docker compose logs -f pest-detect

# 健康检查（start_period 40s 内为加载中）
curl http://127.0.0.1:8000/health
# → {"status":"ok","model":"best.pt"}

# 浏览器访问
http://服务器公网IP:8000
```

> 记得在阿里云控制台**安全组放行 TCP 8000**。

## 七、常用运维命令

```bash
docker compose ps                 # 状态
docker compose logs -f pest-detect  # 实时日志
docker compose restart             # 重启
docker compose down                # 停止（保留数据卷）
docker compose up -d --build       # 更新代码后重建
```

---

## 八、数据与持久化说明

| 路径（宿主机） | 挂载到容器 | 说明 |
|---------------|-----------|------|
| `backend/models/` | `/app/backend/models` | 模型权重（宿主放 best.pt，更新模型只需替换文件+重启） |
| `backend/results/` | `/app/backend/results` | 检测结果图（宿主机可直接查看/备份） |
| `backend/uploads/` | `/app/backend/uploads` | 上传临时文件 |
| `knowledge/vector_store/` | `/app/knowledge/vector_store` | 语义索引（首次自动构建后持久化） |

⚠️ **注意**：
1. **模型目录不能为空**：bind mount 会"覆盖"镜像内同名目录，若 `backend/models` 为空，容器里将看不到任何模型 → 启动失败。务必先放好 `best.pt`。
2. **检测历史（SQLite）**：`pest_data.db` 默认留在容器层，`docker compose down` 不丢失（restart 保留），但**重建容器（up -d --build）会清空历史记录**。如需持久化，可在 compose 中为其增加命名卷：
   ```yaml
   volumes:
     - pest_data:/app/backend/pest_data.db   # 命名卷挂载到 db 文件（首次自动创建文件）
   ```
   （然后取消 `.dockerignore` 中对该文件的排除不需要，因为不在构建上下文。）

---

## 九、与传统部署对比（论文可写）

| 维度 | 传统脚本部署 | Docker 部署 |
|------|-------------|-------------|
| 部署步骤 | 装 python/venv/依赖/手动启动 | 一条 `up -d` |
| 环境一致性 | 依赖系统环境，易污染 | 镜像隔离，可复现 |
| 崩溃恢复 | 需 systemd 自建 | `restart: unless-stopped` 自动拉起 |
| 版本更新 | 手动拉代码+重装 | 重建镜像 |
| 论文价值 | — | 实验 D"部署可复现性"加分项 |

---

## 十、排障速查

| 现象 | 处理 |
|------|------|
| 启动即退出 | `docker compose logs` 看报错；多为 `backend/models` 为空或缺少 `.env` 相关 |
| `/health` 一直 loading | 模型加载中（约 30s）；超时看日志是否报错 |
| 语义索引重新构建慢 | `knowledge/vector_store` 首次为空目录，容器会自动构建；构建后持久化到宿主机 |
| 端口被占用 | 改 `docker-compose.yml` 中 `8000:8000` 左侧端口 |

---

*配合 `DEPLOY_SERVER.md`（传统部署）使用；Docker 为推荐方式。*
