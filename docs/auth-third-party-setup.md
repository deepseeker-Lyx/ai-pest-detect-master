# 第三方登录/短信服务接入配置指南

本系统已内置「手机号验证码登录」与「微信登录」的完整代码逻辑，但真实使用需要接入第三方服务。
本文档教你**需要准备什么、去哪里申请、填到哪个文件**。

## 0. 核心概念：配置文件 `.env`

所有凭据统一从**项目根目录的 `.env` 文件**读取（系统启动时会自动加载，`.env` 已被 `.gitignore` 忽略，不会泄露到 GitHub）。

- 模板在 **`.env.example`**，先复制一份：

  ```bash
  # Windows (PowerShell)
  Copy-Item .env.example .env
  ```

- 用记事本/VS Code 打开 `.env`，把 `#` 注释掉的行取消注释、填上你的真实值
- **改完重启服务生效**

---

## 1. 手机号验证码登录（短信服务）

当前「调试模式」用固定验证码 `123456` 且前端自动填入，**只适合开发演示**。要让验证码真正发到用户手机，任选一家短信服务商：

### 方式 A：阿里云短信（推荐，国内最常用）

1. **注册**：阿里云官网 → 开通「短信服务」
2. **申请签名**：控制台 → 短信服务 → 国内消息 → 签名管理 → 添加签名（如「智检植保」）
3. **申请模板**：模板管理 → 添加模板，正文建议：
   ```
   您的验证码为：${code}，${minutes}分钟内有效。请勿泄露给他人。
   ```
   （变量名必须包含 `code`，系统就是用 `code` 替换验证码的）
4. **获取 AccessKey**：右上角头像 → AccessKey 管理 → 创建 AccessKey，拿到 `AccessKey ID` / `AccessKey Secret`（**Secret 只显示一次，务必保存**）
5. **安装 SDK**（在项目虚拟环境）：
   ```bash
   .venv\Scripts\pip install alibabacloud-dysmsapi20170525 alibabacloud-tea-openapi alibabacloud-tea-util
   ```
6. **填 `.env`**：
   ```
   SMS_PROVIDER=aliyun
   ALIYUN_ACCESS_KEY_ID=你的AccessKeyID
   ALIYUN_ACCESS_KEY_SECRET=你的AccessKeySecret
   ALIYUN_SMS_SIGN_NAME=你的签名
   ALIYUN_SMS_TEMPLATE_CODE=SMS_XXXXXXXX
   ```

### 方式 B：腾讯云短信

1. 腾讯云 → 开通「短信」
2. 申请签名 + 正文模板（变量用 `{1}`，对应验证码）
3. 获取 API 密钥（`SecretId` / `SecretKey`）与短信 `SDKAppID`
4. 安装 SDK：
   ```bash
   .venv\Scripts\pip install tencentcloud-sdk-python
   ```
5. 填 `.env`：
   ```
   SMS_PROVIDER=tencent
   TENCENT_SECRET_ID=...
   TENCENT_SECRET_KEY=...
   TENCENT_SMS_SDK_APP_ID=1400XXXXXX
   TENCENT_SMS_SIGN_NAME=你的签名
   TENCENT_SMS_TEMPLATE_ID=1234567
   ```

### 验证方式

配置后重启服务，登录页「手机号」Tab 点「获取验证码」→ 手机会收到真实短信 → 验证码框**不再自动填入**（需手动输入收到的码）。

> 提示：国内短信服务都需要**实名认证 + 签名审核**（通常几小时到 1 天）。调试时建议用自己/同事的手机号，每个号每天有免费条数限制。

---

## 2. 微信登录（网站应用 OAuth2）

微信登录要求**企业主体**（个人无法注册开放平台网站应用），流程较长，建议先准备资质再操作：

1. **注册开放平台**：<https://open.weixin.qq.com> → 注册开发者账号（需**企业认证**，300 元/年审核费）
2. **创建网站应用**：管理中心 → 网站应用 → 创建应用，填写应用名称/图标/简介
3. **审核通过后**拿到：
   - `AppID`
   - `AppSecret`（只显示一次，立即保存）
4. **配置授权回调域**：网站应用 → 开发信息 → 授权回调域，填你的服务器域名（如 `http://118.178.253.XX:8000`）
5. **填 `.env`**：
   ```
   WECHAT_APPID=wxXXXXXXXXXXXXXXXX
   WECHAT_SECRET=你的AppSecret
   ```
6. **可选**：修改回调地址。代码默认回调到 `http://127.0.0.1:8000/login`（`routes_auth.py` 的 `/auth/wechat/auth-url`），部署到服务器后需改成你的公网地址（建议把 `redirect_uri` 参数化到环境变量 `WECHAT_REDIRECT_URI`）。

### 说明

- 未配置 `WECHAT_APPID` / `WECHAT_SECRET` 时，微信 Tab 会显示明确降级提示（当前就是这样），**不会做假登录**。
- `openid` 是微信侧唯一标识，自动注册的账号用户名形如 `wx_xxxxxxxxxxxx`。

---

## 3. 安全提醒

| 事项 | 说明 |
|---|---|
| `.env` 权限 | 已加入 `.gitignore`，不要手动提交到 GitHub |
| AccessKey/Secret | 云厂商都有子账号（RAM）体系，建议用最小权限子账号而不是主账号 Key |
| 短信费用 | 按条计费（约 0.04-0.05 元/条），系统已有 60 秒重发限制防轰炸 |
| 验证码有效期 | 10 分钟，一次性使用，已内置 |

## 4. 代码位置速查

| 功能 | 文件 |
|---|---|
| `.env` 加载 + 短信发送 + 微信登录函数 | `backend/auth.py` |
| 短信/微信 HTTP 接口 | `backend/routes_auth.py` |
| 前端手机号/微信 Tab | `frontend/templates/login.html` |
| 配置模板 | `.env.example` |
