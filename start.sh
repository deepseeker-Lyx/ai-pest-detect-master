#!/bin/bash
# ──────────────────────────────────────────────────────────────────
# 农作物害虫检测 Web 系统 — 一键启动脚本 (Linux/macOS)
# 用法：bash start.sh
# 公网访问：启动后通过 http://公网IP:8000 访问
# ──────────────────────────────────────────────────────────────────

set -e
cd "$(dirname "$0")"

# 激活 conda 环境（如果存在）
if command -v conda &>/dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate pest 2>/dev/null || true
fi

# 安装依赖
echo "📦 检查依赖..."
pip install -q -r requirements.txt

# ── 显示访问地址 ──
echo ""
echo "🚀 启动服务中..."
echo ""

# 本机访问
echo "   📍 本机:   http://localhost:8000"

# 局域网 IP
LAN_IP=$(ip -4 addr show | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | grep -v '127.0.0.1' | head -1 2>/dev/null || \
         ipconfig getifaddr en0 2>/dev/null || \
         hostname -I 2>/dev/null | awk '{print $1}')
if [ -n "$LAN_IP" ]; then
    echo "   🏠 局域网: http://${LAN_IP}:8000"
fi

# 公网 IP
PUBLIC_IP=$(curl -s --max-time 3 https://api.ipify.org 2>/dev/null || echo "")
if [ -n "$PUBLIC_IP" ]; then
    echo "   🌐 公网:   http://${PUBLIC_IP}:8000"
    echo ""
    echo "   ⚠️  公网访问要求："
    echo "       1. 服务器防火墙需开放 8000 端口"
    echo "       2. 云服务器需在安全组添加 8000 端口规则"
    echo "       3. 家庭宽带需路由器端口转发"
fi

echo ""
printf '━%.0s' {1..50}
echo ""

# 启动服务（绑定 0.0.0.0 = 所有网卡均可访问）
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
