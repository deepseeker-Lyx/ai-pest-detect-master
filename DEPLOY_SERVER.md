# 阿里云轻量服务器部署

这个项目是 YOLO 图片识别项目，需要 PyTorch。普通阿里云轻量服务器通常没有 NVIDIA GPU，所以部署时应该安装 CPU 版 PyTorch，不要安装 CUDA、nvidia-cublas、nvidia-cudnn、triton 等 GPU 依赖。

推荐使用 Ubuntu + Python 3.11。仓库里的 `requirements.txt` 已固定 Linux x86_64 / Python 3.11 的 CPU PyTorch；如果阿里云部署工具会自动读取它，也不会再走 GPU 依赖。

## 首次部署

```bash
cd ~/post-detect
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev
bash deploy_server.sh
```

服务启动后默认监听：

```text
0.0.0.0:8000
```

阿里云控制台需要在防火墙或安全组里放行 TCP 8000 端口，然后可以访问：

```text
http://服务器公网IP:8000
```

## 以后启动

依赖装好以后，不要每次都重新安装依赖。

```bash
cd ~/post-detect
source ~/post-detect-env/bin/activate
cd backend
python main.py
```

如果终端前面只显示 `(base)`，说明还在 Conda 的 base 环境里，要先激活：

```bash
source ~/post-detect-env/bin/activate
```

## 如果手动安装依赖

优先执行部署脚本：

```bash
bash deploy_server.sh
```

如果必须手动安装，可以执行：

```bash
python -m pip install -r requirements-server.txt -i https://mirrors.aliyun.com/pypi/simple/
python -m pip install -r requirements-torch-cpu.txt --index-url https://download.pytorch.org/whl/cpu
python -m pip install ultralytics==8.4.21 --no-deps -i https://mirrors.aliyun.com/pypi/simple/
```

如果部署工具又开始下载 `nvidia-*`、`cuda-*`、`triton`，说明它没有使用当前仓库的 CPU 依赖文件，应立即停止并重新拉取最新代码。
