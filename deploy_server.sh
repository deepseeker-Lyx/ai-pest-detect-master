#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v python3.11 >/dev/null 2>&1; then
    echo "Python 3.11 is required. Install it first:"
    echo "sudo apt update && sudo apt install -y python3.11 python3.11-venv python3.11-dev"
    exit 1
fi

python3.11 -m venv "$HOME/post-detect-env"
source "$HOME/post-detect-env/bin/activate"

python -m pip install --upgrade pip
python -m pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
python -m pip config set install.trusted-host mirrors.aliyun.com

# ⚠️ 先装 CPU 版 PyTorch，再装其他依赖
# 顺序很重要：如果先装 requirements-server.txt，pip 会自动拉 CUDA 版 PyTorch
python -m pip install -r requirements-torch-cpu.txt --index-url https://download.pytorch.org/whl/cpu --force-reinstall --no-deps

python -m pip install -r requirements-server.txt -i https://mirrors.aliyun.com/pypi/simple/
python -m pip install ultralytics==8.4.21 --no-deps -i https://mirrors.aliyun.com/pypi/simple/

cd backend
python main.py
