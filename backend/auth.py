# -*- coding: utf-8 -*-
"""
认证模块 v2 — 数据库账号 + PBKDF2 密码哈希 + JWT(HS256)
=====================================================
角色：
  - user   ：普通用户，使用检测 / 问答等业务功能
  - admin  ：管理员，可查看系统提交信息等管理功能
  - expert ：专家（预留，可扩展）

特性：
  - 账号存 SQLite（users 表），密码 PBKDF2-SHA256 加盐哈希（不存明文）
  - JWT 无状态认证（HS256 自实现，零依赖），Web/小程序/App 通用
  - 无状态 token → 服务重启不失效、可跨端共享
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import random
import sqlite3
import sys
import time
from pathlib import Path

# 确保 backend 包可导入
if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import Depends, Header, HTTPException


# ── 极简 .env 加载器（零依赖，兼容 python-dotenv 子集） ─────────
# 规则：KEY=VALUE 每行一个；# 开头为注释；已存在的环境变量不覆盖；
# 不做引号展开/变量引用（够用即可）。生产环境建议直接用系统环境变量。
def _load_dotenv(path: str = ".env") -> None:
    try:
        p = Path(__file__).resolve().parent.parent / path
        if not p.exists():
            return
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val
    except Exception:
        pass  # .env 读不到不影响启动


_load_dotenv()  # 启动时加载项目根目录 .env（必须在读取环境变量之前）

# ── 数据库（复用 pest_data.db） ──────────────────────────────────
DB_PATH = Path(__file__).resolve().parent / "pest_data.db"

# JWT 密钥：优先从环境变量读取，否则用默认（生产环境务必改）
JWT_SECRET = os.getenv("JWT_SECRET", "pest-detect-dev-secret-change-me")
JWT_EXPIRES_SECONDS = int(os.getenv("JWT_EXPIRES_SECONDS", str(24 * 3600)))  # 24h


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), timeout=15, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _execute(sql: str, params: tuple = ()) -> None:
    conn = _conn()
    try:
        conn.execute(sql, params)
        conn.commit()
    finally:
        conn.close()


def _query(sql: str, params: tuple = (), one: bool = False):
    conn = _conn()
    try:
        cur = conn.execute(sql, params)
        if one:
            row = cur.fetchone()
            return dict(row) if row else None
        return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


# ── 初始化（建表 + 首次 seed 默认账号） ──────────────────────────
def init_db() -> None:
    """创建 users 表（幂等）；若表为空，写入默认账号"""
    conn = _conn()
    try:
        conn.execute("""CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user',
            display TEXT DEFAULT '',
            phone TEXT DEFAULT '',
            wechat_openid TEXT DEFAULT '',
            created_at TEXT
        )""")
        # 旧库迁移：补充手机号 / 微信 openid 列（幂等）
        ucols = {r["name"] for r in conn.execute("PRAGMA table_info(users)").fetchall()}
        if "phone" not in ucols:
            conn.execute("ALTER TABLE users ADD COLUMN phone TEXT DEFAULT ''")
        if "wechat_openid" not in ucols:
            conn.execute("ALTER TABLE users ADD COLUMN wechat_openid TEXT DEFAULT ''")
        conn.commit()
        count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    finally:
        conn.close()

    if count == 0:
        _create_user("admin", "admin123", "admin", "管理员")
        _create_user("user", "user123", "user", "普通用户")
        print("✅ [认证] 已初始化默认账号：admin/admin123、user/user123")


# ── 密码哈希（PBKDF2-SHA256，标准库零依赖） ─────────────────────
def _hash_password(password: str) -> str:
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)
    return f"pbkdf2${salt.hex()}${dk.hex()}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        _algo, salt_hex, dk_hex = stored.split("$")
        salt = bytes.fromhex(salt_hex)
        dk = bytes.fromhex(dk_hex)
        calc = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)
        return hmac.compare_digest(calc, dk)
    except Exception:
        return False


# ── JWT（HS256，标准库零依赖实现） ──────────────────────────────
def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def _sign(data: str) -> str:
    return _b64url(hmac.new(JWT_SECRET.encode("utf-8"), data.encode("utf-8"), hashlib.sha256).digest())


def create_token(payload: dict, expires_seconds: int = JWT_EXPIRES_SECONDS) -> str:
    """签发 JWT（HS256）"""
    header = {"alg": "HS256", "typ": "JWT"}
    body = dict(payload)
    body["iat"] = int(time.time())
    body["exp"] = int(time.time()) + expires_seconds
    h = _b64url(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    p = _b64url(json.dumps(body, separators=(",", ":")).encode("utf-8"))
    return f"{h}.{p}.{_sign(h + '.' + p)}"


def decode_token(token: str) -> dict | None:
    """解析并校验 JWT；无效或过期返回 None"""
    try:
        h, p, sig = token.split(".")
        if not hmac.compare_digest(sig, _sign(h + "." + p)):
            return None
        payload = json.loads(_b64url_decode(p))
        if payload.get("exp", 0) < time.time():
            return None
        return payload
    except Exception:
        return None


# ── 用户操作 ────────────────────────────────────────────────────
def _create_user(username: str, password: str, role: str = "user", display: str = "") -> bool:
    """创建用户；用户名已存在返回 False"""
    try:
        _execute(
            "INSERT INTO users (username, password_hash, role, display, created_at) VALUES (?,?,?,?,?)",
            (username, _hash_password(password), role, display, time.strftime("%Y-%m-%d %H:%M:%S")),
        )
        return True
    except sqlite3.IntegrityError:
        return False


def get_user_by_username(username: str) -> dict | None:
    return _query("SELECT * FROM users WHERE username=?", (username,), one=True)


def list_users() -> list[dict]:
    """返回全部用户（不含密码哈希）"""
    return _query("SELECT id, username, role, display, created_at FROM users ORDER BY id")


def get_user_by_id(user_id: int) -> dict | None:
    """按 ID 查用户（不含密码哈希）"""
    return _query(
        "SELECT id, username, role, display, created_at FROM users WHERE id=?",
        (user_id,), one=True,
    )


def update_user(user_id: int, display: str | None = None,
                role: str | None = None, password: str | None = None) -> bool:
    """更新用户资料：昵称 / 角色 / 密码。返回是否成功"""
    sets, params = [], []
    if display is not None:
        sets.append("display=?"); params.append(display)
    if role is not None:
        if role not in ("user", "admin", "expert"):
            return False
        sets.append("role=?"); params.append(role)
    if password:
        sets.append("password_hash=?"); params.append(_hash_password(password))
    if not sets:
        return False
    params.append(user_id)
    try:
        _execute(f"UPDATE users SET {', '.join(sets)} WHERE id=?", tuple(params))
        return True
    except Exception:
        return False


def delete_user(user_id: int) -> bool:
    """删除用户（调用方需确保非 admin / 非本人）"""
    try:
        _execute("DELETE FROM users WHERE id=?", (user_id,))
        return True
    except Exception:
        return False


def create_user(username: str, password: str, role: str = "user",
                display: str = "") -> dict | None:
    """创建用户（管理员用）；成功返回用户信息（不含密码哈希），失败返回 None"""
    if not username or not password or len(password) < 6:
        return None
    if role not in ("user", "admin", "expert"):
        role = "user"
    if _create_user(username, password, role, display or username):
        return _query(
            "SELECT id, username, role, display, created_at FROM users WHERE username=?",
            (username,), one=True,
        )
    return None


def login(username: str, password: str):
    """校验账号密码，成功签发 JWT；失败返回 None"""
    user = get_user_by_username(username)
    if not user or not _verify_password(password, user["password_hash"]):
        return None
    token = create_token({"sub": user["username"], "role": user["role"], "display": user["display"]})
    return token, {
        "username": user["username"],
        "role": user["role"],
        "display": user["display"],
    }


def register(username: str, password: str, display: str = "") -> dict | None:
    """注册普通用户；成功返回用户信息，用户名冲突返回 None"""
    if not username or not password:
        return None
    if len(password) < 6:
        return None
    if _create_user(username, password, "user", display or username):
        return {"username": username, "role": "user", "display": display or username}
    return None


# ── FastAPI 认证依赖 ─────────────────────────────────────────────
def require_login(authorization: str = Header(default="")) -> dict:
    """要求已登录：解析并校验 JWT"""
    token = authorization.removeprefix("Bearer ").strip()
    payload = decode_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="未登录或登录已过期")
    return {
        "username": payload.get("sub"),
        "role": payload.get("role"),
        "display": payload.get("display"),
    }


def optional_user(authorization: str = Header(default="")) -> str:
    """可选身份解析：已登录返回用户名，未登录/游客返回空字符串"""
    token = authorization.removeprefix("Bearer ").strip()
    payload = decode_token(token)
    if not payload:
        return ""
    return payload.get("sub", "")


def require_admin(user: dict = Depends(require_login)) -> dict:
    """要求管理员角色"""
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="需要管理员权限")
    return user


def require_expert(user: dict = Depends(require_login)) -> dict:
    """要求专家或管理员角色（专家工作台 / 识别质量审核）"""
    if user.get("role") not in ("admin", "expert"):
        raise HTTPException(status_code=403, detail="需要专家权限")
    return user


# ── 手机号 / 微信登录 ──────────────────────────────────────────
# 短信验证码（内存存储，生产环境可换 Redis；重启即失效，演示够用）
_SMS_CODES: dict[str, dict] = {}      # phone -> {code, expires_at, last_send_at}
SMS_CODE_TTL = 600                     # 验证码有效期：10 分钟
SMS_RESEND_SECONDS = 60                # 重发间隔：60 秒（防短信轰炸）
# 短信服务商：aliyun / tencent / 留空=调试模式（不真正发短信，用 DEBUG_SMS_CODE）
SMS_PROVIDER = os.getenv("SMS_PROVIDER", "").strip().lower()
# 调试验证码：仅在未配置短信服务商时使用（真实短信配置后自动失效）
DEBUG_SMS_CODE = os.getenv("DEBUG_SMS_CODE", "123456")

# 微信开放平台凭据（环境变量配置；未配置时微信登录会明确降级提示）
WECHAT_APPID = os.getenv("WECHAT_APPID", "")
WECHAT_SECRET = os.getenv("WECHAT_SECRET", "")


def _is_valid_phone(phone: str) -> bool:
    """简单手机号校验：11 位数字、1 开头"""
    return len(phone) == 11 and phone.isdigit() and phone.startswith("1")


def send_sms_code(phone: str) -> dict:
    """发送短信验证码：含频率限制；未接短信服务时用调试码并打印日志。
    返回 {"ok": bool, "msg": str, "debug_code": str|None}"""
    phone = phone.strip()
    if not _is_valid_phone(phone):
        return {"ok": False, "msg": "手机号格式不正确（需 11 位数字）"}
    now = time.time()
    rec = _SMS_CODES.get(phone)
    if rec and now - rec.get("last_send_at", 0) < SMS_RESEND_SECONDS:
        wait = int(SMS_RESEND_SECONDS - (now - rec["last_send_at"]))
        return {"ok": False, "msg": f"发送过于频繁，请 {wait} 秒后再试"}
    # 生成 6 位验证码（真实短信场景也先生成，发送成功后才入库）
    code = f"{random.randint(0, 999999):06d}"

    # 已配置真实短信服务商 → 真正发送，失败则不生成记录
    if SMS_PROVIDER in ("aliyun", "tencent"):
        sent, err = _send_sms(SMS_PROVIDER, phone, code)
        if not sent:
            return {"ok": False, "msg": f"短信发送失败: {err}"}
        _SMS_CODES[phone] = {"code": code, "expires_at": now + SMS_CODE_TTL, "last_send_at": now}
        return {"ok": True, "msg": "验证码已发送"}

    # 调试模式：固定调试码（前端自动填入），便于未接短信时测试全流程
    if DEBUG_SMS_CODE:
        code = DEBUG_SMS_CODE
    _SMS_CODES[phone] = {"code": code, "expires_at": now + SMS_CODE_TTL, "last_send_at": now}
    print(f"📱 [短信] 向 {phone} 发送验证码: {code}（有效期 {SMS_CODE_TTL // 60} 分钟）")
    return {"ok": True, "msg": "验证码已发送", "debug_code": code if DEBUG_SMS_CODE else None}


def _send_sms(provider: str, phone: str, code: str) -> tuple[bool, str]:
    """按配置的短信服务商发送验证码；返回 (是否成功, 错误信息)"""
    if provider == "aliyun":
        return _send_sms_aliyun(phone, code)
    if provider == "tencent":
        return _send_sms_tencent(phone, code)
    return False, f"未知短信服务商: {provider}"


def _send_sms_aliyun(phone: str, code: str) -> tuple[bool, str]:
    """阿里云短信：需配置 ALIYUN_ACCESS_KEY_ID/SECRET + 签名 + 模板（模板变量名必须是 code）"""
    try:
        from alibabacloud_dysmsapi20170525.client import Client
        from alibabacloud_tea_openapi import models as open_api_models
    except ImportError:
        return False, "缺少阿里云短信 SDK（pip install alibabacloud-dysmsapi20170525 alibabacloud-tea-openapi alibabacloud-tea-util）"

    ak = os.getenv("ALIYUN_ACCESS_KEY_ID", "")
    sk = os.getenv("ALIYUN_ACCESS_KEY_SECRET", "")
    sign = os.getenv("ALIYUN_SMS_SIGN_NAME", "")
    tpl = os.getenv("ALIYUN_SMS_TEMPLATE_CODE", "")
    if not (ak and sk and sign and tpl):
        return False, "缺少阿里云短信配置（ALIYUN_ACCESS_KEY_ID/ALIYUN_ACCESS_KEY_SECRET/ALIYUN_SMS_SIGN_NAME/ALIYUN_SMS_TEMPLATE_CODE）"

    config = open_api_models.Config(access_key_id=ak, access_key_secret=sk)
    config.endpoint = "dysmsapi.aliyuncs.com"
    client = Client(config)
    from alibabacloud_dysmsapi20170525 import models as sms_models
    req = sms_models.SendSmsRequest(
        phone_numbers=phone,
        sign_name=sign,
        template_code=tpl,
        template_param='{"code":"%s"}' % code,
    )
    try:
        resp = client.send_sms(req)
        if resp.body.code == "OK":
            return True, ""
        return False, f"{resp.body.code}: {resp.body.message}"
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def _send_sms_tencent(phone: str, code: str) -> tuple[bool, str]:
    """腾讯云短信：需配置 TENCENT_SECRET_ID/KEY + SDKAppID + 签名 + 模板"""
    try:
        from tencentcloud.common import credential
        from tencentcloud.common.profile.client_profile import ClientProfile
        from tencentcloud.common.profile.http_profile import HttpProfile
        from tencentcloud.sms.v20210111 import sms_client, models
    except ImportError:
        return False, "缺少腾讯云短信 SDK（pip install tencentcloud-sdk-python）"

    sid = os.getenv("TENCENT_SECRET_ID", "")
    skey = os.getenv("TENCENT_SECRET_KEY", "")
    appid = os.getenv("TENCENT_SMS_SDK_APP_ID", "")
    sign = os.getenv("TENCENT_SMS_SIGN_NAME", "")
    tpl = os.getenv("TENCENT_SMS_TEMPLATE_ID", "")
    if not (sid and skey and appid and sign and tpl):
        return False, "缺少腾讯云短信配置（TENCENT_SECRET_ID/TENCENT_SECRET_KEY/TENCENT_SMS_SDK_APP_ID/TENCENT_SMS_SIGN_NAME/TENCENT_SMS_TEMPLATE_ID）"

    cred = credential.Credential(sid, skey)
    http_profile = HttpProfile()
    http_profile.endpoint = "sms.tencentcloudapi.com"
    client_profile = ClientProfile()
    client_profile.httpProfile = http_profile
    client = sms_client.SmsClient(cred, "", client_profile)
    req = models.SendSmsRequest()
    req.SmsSdkAppId = appid
    req.SignName = sign
    req.TemplateId = tpl
    req.PhoneNumberSet = [f"+86{phone}"]
    req.TemplateParamSet = [code]
    try:
        resp = client.SendSms(req)
        st = (resp.SendStatusSet or [None])[0]
        if st and st.Code == "Ok":
            return True, ""
        return False, f"{st.Code}: {st.Message}" if st else "发送失败"
    except Exception as e:  # noqa: BLE001
        return False, str(e)

def verify_sms_code(phone: str, code: str) -> bool:
    """校验验证码：一次性 + 有效期 + 防时序攻击"""
    rec = _SMS_CODES.get(phone.strip())
    if not rec:
        return False
    if time.time() > rec["expires_at"]:
        _SMS_CODES.pop(phone.strip(), None)  # 过期即清除
        return False
    ok = hmac.compare_digest(str(rec["code"]), str(code.strip()))
    if ok:
        _SMS_CODES.pop(phone.strip(), None)  # 一次性使用
    return ok


def get_user_by_phone(phone: str) -> dict | None:
    return _query("SELECT * FROM users WHERE phone=?", (phone,), one=True)


def get_user_by_openid(openid: str) -> dict | None:
    return _query("SELECT * FROM users WHERE wechat_openid=?", (openid,), one=True)


def login_by_phone(phone: str) -> tuple | None:
    """手机号登录：有账号直接登录；无账号自动注册（手机号即账号，免密注册合一）"""
    phone = phone.strip()
    user = get_user_by_phone(phone)
    if not user:
        username = f"u{phone}"   # 避免与普通用户名冲突
        if not _create_user(username, os.urandom(16).hex(), "user", f"手机用户{phone[-4:]}"):
            return None
        _execute("UPDATE users SET phone=? WHERE username=?", (phone, username))
        user = get_user_by_phone(phone)
    token = create_token({"sub": user["username"], "role": user["role"], "display": user["display"]})
    return token, {"username": user["username"], "role": user["role"], "display": user["display"]}


def login_by_wechat(openid: str) -> tuple | None:
    """微信登录：按 openid 查账号，无则自动注册（openid 即微信侧唯一标识）"""
    user = get_user_by_openid(openid)
    if not user:
        username = f"wx_{openid[:12]}"
        if not _create_user(username, os.urandom(16).hex(), "user", "微信用户"):
            return None
        _execute("UPDATE users SET wechat_openid=? WHERE username=?", (openid, username))
        user = get_user_by_openid(openid)
    token = create_token({"sub": user["username"], "role": user["role"], "display": user["display"]})
    return token, {"username": user["username"], "role": user["role"], "display": user["display"]}


# 模块导入时自动初始化（建表 + seed）
init_db()

