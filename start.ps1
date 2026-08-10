# 农作物害虫检测 Web 系统 - Windows 启动脚本
# 用法：在项目根目录运行 .\start.ps1
# 公网访问：启动后会自动配置防火墙规则，外网通过 http://公网IP:8000 访问

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "Checking Python dependencies..."
python -m pip install -r requirements.txt -q

# ── 配置 Windows 防火墙（允许公网访问 8000 端口） ──
$ruleName = "PestDetect-Port8000"
$existing = Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue
if (-not $existing) {
    try {
        New-NetFirewallRule -DisplayName $ruleName `
            -Direction Inbound -Protocol TCP -LocalPort 8000 `
            -Action Allow -Profile Any | Out-Null
        Write-Host "🔓 防火墙规则已添加：允许公网访问端口 8000"
    } catch {
        Write-Host "⚠️  无法添加防火墙规则（请以管理员身份运行）"
    }
} else {
    Write-Host "🔓 防火墙规则已存在：端口 8000 已开放"
}

# ── 显示访问地址 ──
Write-Host ""
Write-Host "🚀 Starting service..."
Write-Host ""

# 本机访问
Write-Host "   📍 本机:   http://localhost:8000" -ForegroundColor Green

# 局域网 IP
$lanIp = (Get-NetIPAddress -AddressFamily IPv4 |
    Where-Object {
        $_.IPAddress -notlike "127.*" -and
        $_.IPAddress -notlike "169.254.*" -and
        $_.PrefixOrigin -ne "WellKnown"
    } |
    Select-Object -First 1 -ExpandProperty IPAddress)

if ($lanIp) {
    Write-Host "   🏠 局域网: http://${lanIp}:8000" -ForegroundColor Yellow
}

# 公网 IP（通过 API 查询）
try {
    $publicIp = (Invoke-WebRequest -Uri "https://api.ipify.org" -TimeoutSec 3).Content.Trim()
    Write-Host "   🌐 公网:   http://${publicIp}:8000" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "   ⚠️  公网访问要求："
    Write-Host "       1. 本机防火墙已开放（脚本已自动配置 ↑）"
    Write-Host "       2. 路由器需设置端口转发（虚拟服务器 → TCP 8000 → 本机IP）"
    Write-Host "       3. 部分运营商（如移动）不提供公网IP，可用内网穿透工具"
} catch {
    Write-Host "   🌐 公网IP: 查询失败（可能无网络）" -ForegroundColor DarkGray
}

Write-Host ""
Write-Host "━" * 50
Write-Host ""

# ── 启动服务（绑定 0.0.0.0 = 所有网卡均可访问） ──
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
