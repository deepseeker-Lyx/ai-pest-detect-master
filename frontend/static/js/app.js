/* ================================================================
   农作物害虫检测 Web — YOLOv11 识别系统
   ================================================================ */

const API_BASE = '';   // 与后端同源，留空即可

// ── DOM 引用 ─────────────────────────────────────────────────────
const messagesArea = document.getElementById('messagesArea');
const messageInput = document.getElementById('messageInput');
const sendBtn = document.getElementById('sendBtn');
const cameraBtn = document.getElementById('cameraBtn');
const uploadBtn = document.getElementById('uploadBtn');
const fileInput = document.getElementById('fileInput');
const cameraInput = document.getElementById('cameraInput');
const imagePreviewOverlay = document.getElementById('imagePreviewOverlay');
const previewImage = document.getElementById('previewImage');
const closePreview = document.getElementById('closePreview');
const uploadHero = document.getElementById('uploadHero');
const heroCamera = document.getElementById('heroCamera');
const heroUpload = document.getElementById('heroUpload');
const inputArea = document.getElementById('inputArea');

let currentPest = null;
let currentPests = [];        // 最近检测到的所有害虫名
let pendingPestNames = [];    // 待处理的多害虫名单（点击“全部防治”时使用）
let lastDetections = [];      // 最近一次检测结果（用于选择害虫/全部防治）
let pestSelected = false;     // 是否主动选中某一种害虫（false 时直接提问覆盖全部）
let qaHistory = [];
let qaBusy = false;

// ── 初始化 ────────────────────────────────────────────────────────
function initApp() {
  showWelcomeMessage();
  bindEvents();
  setupAuthUI();
}

// ── 登录状态与角色 UI（用户 / 管理员 / 专家） ────────────────────
function setupAuthUI() {
  const role = sessionStorage.getItem('pest_role');
  const isAdmin = role === 'admin';
  const isExpert = role === 'expert';  // 专家工作台：仅专家可见（管理员从管理后台监督专家成果）

  // 专家工作台入口（专家 / 管理员可见）
  const expertBtn = document.getElementById('expertBtn');
  if (expertBtn) {
    expertBtn.style.display = isExpert ? '' : 'none';
    if (isExpert) {
      expertBtn.addEventListener('click', () => { location.href = '/expert?v=6'; });
    }
  }

  // 管理员入口按钮（仅管理员可见）
  const adminBtn = document.getElementById('adminBtn');
  if (adminBtn) {
    adminBtn.style.display = isAdmin ? '' : 'none';
    if (isAdmin) {
      adminBtn.addEventListener('click', () => { location.href = '/admin?v=6'; });
    }
  }

  // 顶部导航栏：登录用户显示「退出登录」图标（能进入即已登录）
  const logoutBtn = document.getElementById('logoutBtn');
  if (logoutBtn) {
    logoutBtn.style.display = '';
    logoutBtn.title = '退出登录';
    logoutBtn.innerHTML = '<i class="fa-solid fa-right-from-bracket"></i>';
    logoutBtn.onclick = () => doLogout();
  }

  // 内部测试模式标识：专家/管理员在主应用的操作视为测试数据（不进入用户统计）
  const testModeTag = document.getElementById('testModeTag');
  if (testModeTag) {
    testModeTag.style.display = (isAdmin || isExpert) ? '' : 'none';
  }
}

function doLogout() {
  const token = sessionStorage.getItem('pest_token');
  if (token) {
    fetch('/auth/logout', {
      method: 'POST',
      headers: { 'Authorization': 'Bearer ' + token },
    }).catch(() => {});
  }
  sessionStorage.removeItem('pest_token');
  sessionStorage.removeItem('pest_role');
  sessionStorage.removeItem('pest_username');
  sessionStorage.removeItem('pest_display');
  sessionStorage.removeItem('pest_guest');
  location.href = '/login';
}

function showWelcomeMessage() {
  const welcome = document.createElement('div');
  welcome.className = 'welcome-message';
  welcome.id = 'welcomeMessage';
  welcome.innerHTML = `
    <div class="welcome-message-title">🌾 用户您好！我是您的智能植保助手</div>
    <div class="welcome-message-text">
      我是基于 YOLOv11 知识蒸馏模型训练的智能植保助手，可精准识别 16 类水稻害虫。
      上传害虫照片，即可获得识别结果、防治建议与当地虫害风险预警，
      全程云端服务，为您的水稻丰收保驾护航！
    </div>
  `;
  // 将欢迎消息插入到 uploadHero 之前
  const hero = document.getElementById('uploadHero');
  if (hero) {
    hero.parentNode.insertBefore(welcome, hero);
  }
}

function bindEvents() {
  // 大按钮上传
  heroCamera.addEventListener('click', () => cameraInput.click());
  heroUpload.addEventListener('click', () => fileInput.click());

  // 发送消息
  sendBtn.addEventListener('click', handleSendMessage);
  messageInput.addEventListener('keypress', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  });

  // 底部上传按钮
  uploadBtn.addEventListener('click', () => fileInput.click());
  cameraBtn.addEventListener('click', () => cameraInput.click());
  fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) handleFile(fileInput.files[0]);
  });
  cameraInput.addEventListener('change', () => {
    if (cameraInput.files[0]) handleFile(cameraInput.files[0]);
  });

  // 图片预览
  closePreview.addEventListener('click', closeImagePreview);
  imagePreviewOverlay.addEventListener('click', e => {
    if (e.target === imagePreviewOverlay) closeImagePreview();
  });
}

// ── 处理发送消息 ──────────────────────────────────────────────────
async function handleSendMessage() {
  const question = messageInput.value.trim();
  if (!question || qaBusy) return;

  messageInput.value = '';
  appendUserMessage(question);
  await askQuestionStream(question);
}

// ── 添加用户消息 ──────────────────────────────────────────────────
function appendUserMessage(text) {
  const message = document.createElement('div');
  message.className = 'message user';
  message.innerHTML = `<div class="message-bubble">${escapeHTML(text)}</div>`;
  messagesArea.appendChild(message);
  scrollToBottom();
}

// ── 添加助手消息 ──────────────────────────────────────────────────
function appendAssistantMessage(html) {
  const message = document.createElement('div');
  message.className = 'message assistant';
  message.innerHTML = `<div class="message-bubble">${html}</div>`;
  messagesArea.appendChild(message);
  scrollToBottom();
  return message;
}

// ── 添加加载消息 ──────────────────────────────────────────────────
function appendLoadingMessage(type) {
  const message = document.createElement('div');
  message.className = 'message assistant';
  const label = type === 'qa' ? '正在分析' : '正在识别';

  message.innerHTML = `
    <div class="message-bubble message-bubble-loading">
      <div class="loading-progress">
        <div class="loading-progress-header">
          <span class="loading-label">${label}</span>
          <div class="loading-dots">
            <span></span>
            <span></span>
            <span></span>
          </div>
        </div>
        <div class="loading-bar-segmented" data-segs>
          ${'<div class="seg"></div>'.repeat(20)}
        </div>
        <div class="loading-pct" data-pct>0%</div>
        <div data-steps>
          ${type === 'qa' ? `
            <div class="loading-progress-step"><span class="step-icon">📋</span> 接收请求</div>
            <div class="loading-progress-step"><span class="step-icon">🔍</span> 检索知识库</div>
            <div class="loading-progress-step"><span class="step-icon">💡</span> 生成回答</div>
          ` : `
            <div class="loading-progress-step"><span class="step-icon">📷</span> 分析图片</div>
            <div class="loading-progress-step"><span class="step-icon">🧠</span> 模型推理</div>
            <div class="loading-progress-step"><span class="step-icon">📋</span> 生成结果</div>
          `}
        </div>
      </div>
    </div>
  `;

  // ── 指数衰减进度曲线（指数反函数时间表） ──────────────────
  // 进度公式：P(t) = MAX_PCT * (1 - e^(-t / TAU))
  // 第 i 格点亮时刻由指数反函数推导：
  //   t_i = -TAU * ln(1 - i / DENOM)   （DENOM = SEG_COUNT * MAX_PCT）
  // 效果：格间间隔前期密（~0.15s，快）、后期疏（~0.6s，慢），
  //   肉眼可见"先快后慢"的指数曲率，渐近线使进度逼近 88% 而不触顶。
  const MAX_PCT = 0.88;          // 渐近线：进度无限逼近 88% 而不触顶
  const TAU = 2.5;               // 时间常数（秒）：控制指数曲率陡缓
  const SEG_COUNT = 20;
  const DENOM = SEG_COUNT * MAX_PCT;   // 渐近格数 ≈ 17.6，其余由收尾补满
  // ⭐ 最短展示时长：即使任务秒完成，也保证进度条按指数曲线表演这么久
  //    （调大后能让"先快后慢、渐近趋稳"的曲率展示得更完整）
  const MIN_SHOW_MS = type === 'qa' ? 2000 : 2600;
  const segs = message.querySelector('[data-segs]');
  const pctEl = message.querySelector('[data-pct]');
  const stepEls = message.querySelectorAll('.loading-progress-step');
  const startTime = Date.now();
  let timerId = null;            // 指数时间表调度器（setTimeout）
  let filled = false;
  let displayedSegs = 0;         // 已点亮的格数

  // 第 i 格（1-based）相对 startTime 的点亮时刻（ms）
  // 由指数反函数推导：t_i = -TAU * ln(1 - i / DENOM)
  function tickTimeFor(i) {
    if (i >= DENOM) return Infinity;   // 渐近线之外，自然永远不会点亮
    return -TAU * 1000 * Math.log(1 - i / DENOM);
  }

  // 时刻 now 下指数曲线应点亮的格数
  function currentSegs(now) {
    const t = (now - startTime) / 1000;
    return Math.min(SEG_COUNT, Math.floor(DENOM * (1 - Math.exp(-t / TAU))));
  }

  // 指数时间表调度：到点点亮下一格（间隔随格号递增）
  function scheduleNext() {
    const nextSeg = displayedSegs + 1;
    const when = tickTimeFor(nextSeg);
    if (!isFinite(when)) { timerId = null; return; }
    const delay = Math.max(0, when - (Date.now() - startTime));
    timerId = setTimeout(() => {
      if (filled) return;
      if (displayedSegs < nextSeg) {
        displayedSegs = nextSeg;
        render();
      }
      scheduleNext();
    }, delay);
  }

  // 检测/问答步骤配置
  const stepConfigs = type === 'qa'
    ? [
        { maxSeg: Math.round(0.25 * SEG_COUNT), label: '📋 接收请求' },
        { maxSeg: Math.round(0.60 * SEG_COUNT), label: '🔍 检索知识库' },
        { maxSeg: Math.round(0.88 * SEG_COUNT), label: '💡 生成回答' },
      ]
    : [
        { maxSeg: Math.round(0.30 * SEG_COUNT), label: '📷 分析图片' },
        { maxSeg: Math.round(0.65 * SEG_COUNT), label: '🧠 模型推理' },
        { maxSeg: Math.round(0.88 * SEG_COUNT), label: '📋 生成结果' },
      ];

  // 渲染当前 displayedSegs 到界面
  function render() {
    const n = Math.min(displayedSegs, SEG_COUNT);

    // 更新竖条（最后一个点亮 + 边缘呼吸）
    for (let i = 0; i < SEG_COUNT; i++) {
      const seg = segs.children[i];
      if (!seg) break;
      seg.className = i < n ? 'seg done' : (i === n && n < SEG_COUNT ? 'seg pulsing' : 'seg');
    }

    // 百分比 = 完全由竖条数决定
    pctEl.textContent = `${Math.round((n / SEG_COUNT) * 100)}%`;

    // 更新步骤高亮
    let stepIdx = 0;
    for (let i = stepConfigs.length - 1; i >= 0; i--) {
      if (n >= stepConfigs[i].maxSeg) { stepIdx = i; break; }
    }
    stepEls.forEach((el, i) => {
      el.className = 'loading-progress-step';
      if (i < stepIdx) el.classList.add('done');
      else if (i === stepIdx) el.classList.add('active');
    });
  }

  // 外部调用的"填满"方法 —— 返回 Promise，动画播完后 resolve
  // ⭐ 特色：即使任务实际完成得很快，也保证至少展示 MIN_SHOW_MS 毫秒。
  //    收尾采用 ease-out 缓动（先快后慢）平滑补满，与指数曲线节奏一致，
  //    避免"末段冲刺"破坏整体"先快后慢"的渐近观感。
  message.fillProgress = function() {
    return new Promise(resolve => {
      if (filled) { resolve(); return; }
      filled = true;
      if (timerId) { clearTimeout(timerId); timerId = null; }

      const remain = SEG_COUNT - displayedSegs;
      if (remain <= 0) { displayedSegs = SEG_COUNT; render(); resolve(); return; }

      // ⭐ 平滑收尾：从当前格数以 ease-out（1-(1-k)^3）补满到 100%，
      //    先快后慢，最后 100% 前自然放缓（渐近感），无突兀冲刺
      const smoothComplete = () => {
        const from = displayedSegs;
        const t0 = Date.now();
        const DURATION = 900;   // 收尾时长（ms）
        const step = () => {
          const k = Math.min(1, (Date.now() - t0) / DURATION);
          const e = 1 - Math.pow(1 - k, 3);   // ease-out cubic：先快后慢
          const target = Math.round(from + (SEG_COUNT - from) * e);
          if (target > displayedSegs) {
            displayedSegs = target;
            render();
          }
          if (k < 1) {
            requestAnimationFrame(step);
          } else {
            displayedSegs = SEG_COUNT;
            render();
            resolve();
          }
        };
        requestAnimationFrame(step);
      };

      // 最短展示时长兜底：若任务完成得太快，先继续按同一指数曲线推进
      // （currentSegs 渐近于 88%），到点后再平滑收尾补满
      const waitMs = Math.max(0, MIN_SHOW_MS - (Date.now() - startTime));
      if (waitMs > 0) {
        const graceTimer = setInterval(() => {
          const n = currentSegs(Date.now());
          if (displayedSegs < n) {
            displayedSegs = n;
            render();
          }
        }, 100);
        setTimeout(() => {
          clearInterval(graceTimer);
          smoothComplete();
        }, waitMs);
      } else {
        smoothComplete();
      }
    });
  };

  // 启动：按指数时间表点亮进度（格间间隔递增 → 先快后慢）
  scheduleNext();
  messagesArea.appendChild(message);
  scrollToBottom();
  return message;
}

// ── 流式问答逻辑 ──────────────────────────────────────────────────
async function askQuestionStream(question) {
  if (qaBusy) return;

  qaBusy = true;
  sendBtn.disabled = true;
  messageInput.disabled = true;

  // 先显示加载消息
  const loadingMessage = appendLoadingMessage('qa');

  let fullText = '';

  try {
    // 多害虫策略：
    //   1) 点击“全部防治”→ pendingPestNames 优先，传全部害虫
    //   2) 主动选中某害虫（pestSelected）→ 只针对该害虫
    //   3) 检测到多种害虫且未选中 → 直接提问默认覆盖全部害虫，后端分别回答
    //   4) 单害虫 → 传该害虫
    const requestedPests = pendingPestNames.slice();
    pendingPestNames = [];
    let pestNames;
    if (requestedPests.length) {
      pestNames = requestedPests;
    } else if (pestSelected && currentPest) {
      pestNames = [currentPest.name];
    } else if (currentPests.length > 1) {
      pestNames = [...new Set(currentPests)];
    } else if (currentPest) {
      pestNames = [currentPest.name];
    } else {
      pestNames = [];
    }

    const response = await fetch(`${API_BASE}/qa/ask-stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer ' + (sessionStorage.getItem('pest_token') || '') },
      body: JSON.stringify({
        question,
        pest_names: pestNames,
        pest_name: pestNames.length === 1 ? pestNames[0] : '',
        history: qaHistory.slice(-6),
      }),
    });

    if (!response.ok) {
      throw new Error('问答服务错误');
    }

    // 收到第一个数据后，移除加载消息，创建回答消息
    let messageDiv = null;
    let bubble = null;
    let isFirstChunk = true;

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const {done, value} = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value);
      const lines = chunk.split('\n');

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const dataStr = line.slice(6);
          try {
            const data = JSON.parse(dataStr);

            if (data.text) {
              // 收到第一个内容块时，移除加载消息并创建回答消息
              if (isFirstChunk) {
                if (loadingMessage._stepTimer) clearInterval(loadingMessage._stepTimer);
                if (loadingMessage.fillProgress) await loadingMessage.fillProgress();
                loadingMessage.remove();
                messageDiv = appendAssistantMessage('');
                bubble = messageDiv.querySelector('.message-bubble');
                isFirstChunk = false;
              }

              fullText += data.text;
              bubble.innerHTML = renderMarkdown(fullText);
              scrollToBottom();
            }

            if (data.sources && data.sources.length > 0) {
              const sourcesHtml = `<div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid var(--border); font-size: 12px; color: var(--text-sub);">
                📚 资料来源：${data.sources.map(s => escapeHTML(s.title)).join('、')}
              </div>`;
              bubble.innerHTML += sourcesHtml;
            }

            if (data.done) {
              // 更新历史
              qaHistory.push({ role: 'user', content: question });
              qaHistory.push({ role: 'assistant', content: fullText });
            }
          } catch (e) {
            console.error('解析数据失败:', e);
          }
        }
      }
    }

    // 如果没有收到任何内容，移除加载消息并显示错误
    if (isFirstChunk) {
      if (loadingMessage._stepTimer) clearInterval(loadingMessage._stepTimer);
      if (loadingMessage.fillProgress) await loadingMessage.fillProgress();
      loadingMessage.remove();
      appendAssistantMessage('抱歉，没有收到回答。');
    }

  } catch (err) {
    if (loadingMessage._stepTimer) clearInterval(loadingMessage._stepTimer);
    if (loadingMessage.fillProgress) await loadingMessage.fillProgress();
    loadingMessage.remove();
    appendAssistantMessage(`抱歉，问答失败：${escapeHTML(err.message)}`);
    console.error(err);
  } finally {
    qaBusy = false;
    sendBtn.disabled = false;
    messageInput.disabled = false;
  }
}

// ── 处理文件上传 ──────────────────────────────────────────────────
async function handleFile(file) {
  // 类型校验
  const allowed = ['image/jpeg', 'image/png', 'image/bmp', 'image/gif', 'image/webp'];
  if (!allowed.includes(file.type)) {
    appendAssistantMessage('不支持的文件格式，请使用 JPG / PNG / BMP 格式的图片。');
    return;
  }

  // 隐藏大按钮区域和欢迎消息，显示输入栏
  if (uploadHero) {
    uploadHero.style.display = 'none';
  }
  const welcomeMsg = document.getElementById('welcomeMessage');
  if (welcomeMsg) {
    welcomeMsg.style.display = 'none';
  }
  inputArea.style.display = 'flex';

  // 显示用户上传的图片
  const objectUrl = URL.createObjectURL(file);
  const userImageMsg = document.createElement('div');
  userImageMsg.className = 'message user';
  userImageMsg.innerHTML = `
    <div class="message-bubble">
      <img src="${objectUrl}" style="max-width: 200px; border-radius: 8px; cursor: pointer;" onclick="showImagePreview('${objectUrl}')" />
    </div>
  `;
  messagesArea.appendChild(userImageMsg);
  scrollToBottom();

  // 显示加载消息
  const loadingMessage = appendLoadingMessage('detect');
  loadingMessage.querySelector('.loading-progress-header span').textContent = '正在识别害虫';

  // 调用后端
  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await fetch(`${API_BASE}/detect/image`, {
      method: 'POST',
      headers: { 'Authorization': 'Bearer ' + (sessionStorage.getItem('pest_token') || '') },
      body: formData,
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: '服务器错误' }));
      throw new Error(err.detail || '识别失败');
    }

    const data = await response.json();

    // 移除加载消息（等待进度条动画播完，保留节奏感）
    if (loadingMessage._stepTimer) clearInterval(loadingMessage._stepTimer);
    if (loadingMessage.fillProgress) await loadingMessage.fillProgress();
    loadingMessage.remove();

    // 显示识别结果
    renderDetectionResult(data);

    // 尝试获取天气和虫害预警
    fetchWeatherAndPestRisk(data);

    // 更新占位符（多害虫时提示可分别询问 / 查看全部防治）
    if (data.detections.length > 0) {
      const topDetection = [...data.detections].sort((a, b) => b.confidence - a.confidence)[0];
      currentPest = topDetection;
      currentPests = data.detections.map(d => d.name);
      messageInput.placeholder = currentPests.length > 1
        ? `检测到 ${currentPests.length} 种害虫，点结果可分别询问，或点「全部防治方法」`
        : `问我关于${topDetection.zh_name || topDetection.name}的问题...`;
    } else {
      currentPest = null;
      currentPests = [];
      messageInput.placeholder = '未识别到害虫，可重新上传或提问通用问题...';
    }

  } catch (err) {
    if (loadingMessage._stepTimer) clearInterval(loadingMessage._stepTimer);
    if (loadingMessage.fillProgress) await loadingMessage.fillProgress();
    loadingMessage.remove();
    appendAssistantMessage(`识别失败：${escapeHTML(err.message)}`);
    console.error(err);
  }
}

// ── 渲染检测摘要描述 ──────────────────────────────────────────────
function renderDetectionSummary(data) {
  const hasUncertain = data.detections.some(d => d.alternatives && d.alternatives.length > 0);
  if (hasUncertain) {
    let totalTargets = data.detections.length;
    let totalPossibilities = data.detections.length;
    data.detections.forEach(d => {
      if (d.alternatives) totalPossibilities += d.alternatives.length;
    });
    return `检测到 ${totalTargets} 个目标，共 ${totalPossibilities} 种可能`;
  }
  return `检测到 ${data.total} 个害虫目标`;
}

// ── 渲染识别结果 ──────────────────────────────────────────────────
function renderDetectionResult(data) {
  lastDetections = data.detections || [];
  pestSelected = false;   // 新检测后默认未选中：直接提问将覆盖全部害虫
  if (data.detections.length === 0 && !data.is_unknown) {
    appendAssistantMessage('未检测到害虫目标，可以尝试上传更清晰的图片。');
    return;
  }

  const detectionCard = document.createElement('div');
  detectionCard.className = 'detection-card';

  // YOLOv11 标识
  let html = `<span class="yolo-badge">YOLOv11 检测结果</span>`;

  // ⭐ 未知虫种警告横幅
  if (data.is_unknown) {
    let warningIcon = '❓';
    let warningTitle = '疑似未知虫种';
    if (data.unknown_type === 'no_detection') {
      warningIcon = '🔍';
      warningTitle = '未检测到已知害虫';
    }
    html += `
      <div class="unknown-warning">
        <div class="unknown-warning-icon">${warningIcon}</div>
        <div class="unknown-warning-content">
          <div class="unknown-warning-title">${warningTitle}</div>
          <div class="unknown-warning-text">${escapeHTML(data.unknown_suggestion || '当前检测结果置信度较低，可能不在已知的16类水稻害虫范围内。')}</div>
        </div>
      </div>
    `;
  }

  // 技术指标
  const maxConf = data.detections.length > 0 ? Math.max(...data.detections.map(d => d.confidence)) : 0;
  html += `
    <div class="detection-stats">
      <div class="stat-item">
        <div class="stat-value">${data.total}</div>
        <div class="stat-label">检测数量</div>
      </div>
      <div class="stat-item">
        <div class="stat-value">${maxConf.toFixed(1)}%</div>
        <div class="stat-label">最高置信度</div>
      </div>
      <div class="stat-item">
        <div class="stat-value">${data.elapsed_ms.toFixed(0)}ms</div>
        <div class="stat-label">检测用时</div>
      </div>
      <div class="stat-item">
        <div class="stat-value">YOLOv11</div>
        <div class="stat-label">识别模型</div>
      </div>
    </div>
  `;

  // 结果图片
  const resultImgUrl = API_BASE + data.result_image + '?t=' + Date.now();
  html += `
    <img src="${resultImgUrl}" class="detection-card-image" onclick="showImagePreview('${resultImgUrl}')" alt="检测结果" />
    <div class="detection-card-title">
      <i class="fa-solid fa-bug"></i>
      ${renderDetectionSummary(data)}
    </div>
    <div class="detection-list">
  `;

  // 害虫列表（交错入场）
  data.detections.forEach((det, idx) => {
    // 判断是否有不确定的备选（重叠框合并）
    const hasAlternatives = det.alternatives && det.alternatives.length > 0;
    const delay = idx * 0.08;
    // 稻穗金：高置信度（>=75%）用金色强调
    const confHigh = det.confidence >= 75;
    const confHtml = confHigh
      ? `<div class="detection-item-confidence conf-high">⭐ ${det.confidence.toFixed(1)}%</div>`
      : `<div class="detection-item-confidence">${det.confidence.toFixed(1)}%</div>`;
    html += `
      <div class="detection-item ${hasAlternatives ? 'detection-item-uncertain' : ''}" style="animation-delay: ${delay}s" onclick="selectPest('${escapeHTML(det.name)}')" data-pest="${escapeHTML(det.name)}">
        <div class="detection-item-name">
          <div class="detection-item-name-zh">${escapeHTML(det.zh_name || det.name)}</div>
          ${det.zh_name ? `<div class="detection-item-name-en">${escapeHTML(det.name)}</div>` : ''}
        </div>
        ${confHtml}
      </div>
    `;
    // 显示备选可能性
    if (hasAlternatives) {
      html += `<div class="detection-alternatives">`;
      det.alternatives.forEach((alt, altIdx) => {
        const altDelay = delay + (altIdx + 1) * 0.06;
        html += `
          <div class="detection-item detection-item-alt" style="animation-delay: ${altDelay}s">
            <div class="detection-item-name">
              <div class="detection-item-name-zh">${escapeHTML(alt.zh_name || alt.name)}</div>
              ${alt.zh_name ? `<div class="detection-item-name-en">${escapeHTML(alt.name)}</div>` : ''}
            </div>
            <div class="detection-item-confidence">${alt.confidence.toFixed(1)}%</div>
          </div>
        `;
      });
      html += `</div>`;
    }
  });

  html += '</div>';

  // 快捷操作按钮：每种害虫独立一组 + 多害虫时提供“全部防治”
  if (data.detections.length > 0) {
    const unique = [];
    const seenNames = new Set();
    [...data.detections].sort((a, b) => b.confidence - a.confidence).forEach(d => {
      if (!seenNames.has(d.name)) { seenNames.add(d.name); unique.push(d); }
    });

    html += `<div class="quick-actions">`;
    if (unique.length > 1) {
      html += `
        <div class="multi-pest-tip">💡 检测到 ${unique.length} 种害虫：直接输入问题（如“这些怎么防治”）将<b>分别解答每一种</b>；也可点下方某害虫单独追问。</div>
        <button class="quick-action-btn quick-all" onclick="askAllPests()">
          📋 查看全部 ${unique.length} 种害虫的防治方法
        </button>
      `;
    }
    unique.forEach(d => {
      const zh = d.zh_name || d.name;
      html += `
        <div class="quick-actions-group" data-pest="${escapeHTML(d.name)}">
          <div class="quick-actions-label">${escapeHTML(zh)}</div>
          <button class="quick-action-btn" onclick="askQuick('${escapeHTML(d.name)}', '怎么防治${zh}？')">🛡️ 防治</button>
          <button class="quick-action-btn" onclick="askQuick('${escapeHTML(d.name)}', '${zh}用什么药？')">🧪 用药</button>
          <button class="quick-action-btn" onclick="askQuick('${escapeHTML(d.name)}', '${zh}危害大吗？')">⚠️ 危害</button>
        </div>
      `;
    });
    html += `</div>`;
  }

  detectionCard.innerHTML = html;
  messagesArea.appendChild(detectionCard);
  scrollToBottom();
}

window.askQuick = function(pestName, question) {
  // 修复：把传入的害虫设为当前上下文（此前该参数未生效，导致总是问第一种）
  const det = lastDetections.find(d => d.name === pestName);
  if (det) currentPest = det;
  pestSelected = true;   // 用户明确指定了某害虫，后续提问只针对它
  messageInput.value = question;
  handleSendMessage();
};

// ── 选择害虫（交互性：点击检测结果项，切换当前问答对象） ────────
window.selectPest = function(name) {
  const det = lastDetections.find(d => d.name === name);
  if (!det) return;
  currentPest = det;
  pestSelected = true;
  messageInput.placeholder = `问我关于${det.zh_name || det.name}的问题...`;
  document.querySelectorAll('.detection-item').forEach(el => {
    el.classList.toggle('selected', el.dataset.pest === name);
  });
};

// ── 全部防治（多害虫：一次请求后端分别回答每种害虫） ────────────
window.askAllPests = function() {
  const names = [...new Set(lastDetections.map(d => d.name))];
  if (!names.length) return;
  pendingPestNames = names;
  messageInput.value = '请分别告诉我这些害虫的防治方法';
  handleSendMessage();
};

// ── 图片预览 ──────────────────────────────────────────────────────
window.showImagePreview = function(imageSrc) {
  previewImage.src = imageSrc;
  imagePreviewOverlay.classList.add('active');
};

function closeImagePreview() {
  imagePreviewOverlay.classList.remove('active');
}

// ── 工具函数 ──────────────────────────────────────────────────────
function scrollToBottom() {
  setTimeout(() => {
    messagesArea.scrollTop = messagesArea.scrollHeight;
  }, 100);
}

function escapeHTML(value) {
  return String(value || '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

function renderMarkdown(value) {
  const escaped = escapeHTML(value);
  return escaped
    // 标题: ### → h4, ## → h3, # → h2
    .replace(/^###\s+(.+)$/gm, '<h4 class="qa-heading">$1</h4>')
    .replace(/^##\s+(.+)$/gm, '<h3 class="qa-heading">$1</h3>')
    .replace(/^#\s+(.+)$/gm, '<h2 class="qa-heading">$1</h2>')
    // 分隔线: ---
    .replace(/^---+$/gm, '<hr class="qa-hr">')
    // 粗体: **text**
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    // 无序列表: - item
    .replace(/^\s*-\s+(.+)$/gm, '<div class="qa-list-item">• $1</div>')
    // 有序列表: 1. item
    .replace(/^\s*\d+\.\s+(.+)$/gm, '<div class="qa-list-item">$1</div>')
    // 换行
    .replace(/\n/g, '<br>');
}

// ── 天气与虫害预警 ────────────────────────────────────────────────

function fetchWeatherAndPestRisk(detectData) {
  // 先尝试浏览器定位（移动端 HTTP 下可能不工作）
  if (navigator.geolocation) {
    navigator.geolocation.getCurrentPosition(
      async (position) => {
        const lat = position.coords.latitude;
        const lng = position.coords.longitude;
        await fetchWeatherByCoord(lat, lng, detectData);
      },
      () => {
        // 浏览器定位失败 → 用 IP 定位降级
        fetchWeatherByIP(detectData);
      },
      { timeout: 5000, enableHighAccuracy: false }
    );
  } else {
    // 不支持浏览器定位 → 用 IP 定位降级
    fetchWeatherByIP(detectData);
  }
}

async function fetchWeatherByCoord(lat, lng, detectData) {
  let weatherData = null;
  try {
    const resp = await fetch(`${API_BASE}/weather/current?lat=${lat}&lng=${lng}`);
    weatherData = await resp.json();
  } catch (e) {
    return;
  }
  if (!weatherData || weatherData.error) return;
  await fetchPestRisk(detectData, lat, lng, weatherData);
}

async function fetchWeatherByIP(detectData) {
  try {
    const resp = await fetch(`${API_BASE}/weather/locate`);
    const data = await resp.json();
    if (data.error || !data.weather) return;
    const loc = data.location || {};
    await fetchPestRisk(detectData, loc.lat, loc.lng, data.weather);
  } catch (e) {}
}

async function fetchPestRisk(detectData, lat, lng, weatherData) {
  const topPest = detectData.detections.length > 0
    ? detectData.detections.sort((a, b) => b.confidence - a.confidence)[0]
    : null;

  let riskAlerts = [];
  if (topPest) {
    try {
      const resp = await fetch(
        `${API_BASE}/weather/pest-risk?pest=${encodeURIComponent(topPest.name)}&lat=${lat}&lng=${lng}`
      );
      const riskData = await resp.json();
      if (riskData.alerts) riskAlerts = riskData.alerts;
    } catch (e) {}
  }

  renderWeatherBar(weatherData, riskAlerts, topPest);
}

function renderWeatherBar(weather, alerts, topPest) {
  const bar = document.createElement('div');
  bar.className = 'weather-bar';

  // 天气行
  let weatherHtml = `
    <div class="weather-bar-row weather-bar-main">
      <span class="weather-bar-icon">🌤</span>
      <span class="weather-bar-text">
        ${escapeHTML(weather.weather || '')}
        ${weather.temperature != null ? ` ${weather.temperature}°C` : ''}
        ${weather.humidity != null ? `  💧 ${weather.humidity}%` : ''}
      </span>
    </div>
  `;

  // 虫害预警行（最多显示2条）
  let alertHtml = '';
  if (alerts && alerts.length > 0) {
    const showAlerts = alerts.slice(0, 2);
    alertHtml = showAlerts.map(a => `
      <div class="weather-bar-row weather-bar-alert">
        <span class="weather-bar-icon">${a.level.split(' ')[0] || '💡'}</span>
        <span class="weather-bar-text">${escapeHTML(a.detail)}</span>
      </div>
    `).join('');
  }

  // 展开/折叠按钮（如果内容多）
  const hasExtra = alerts && alerts.length > 2;
  bar.innerHTML = `
    ${weatherHtml}
    ${alertHtml}
    ${hasExtra ? '<div class="weather-bar-more" onclick="this.style.display=\'none\';this.nextElementSibling.style.display=\'block\'">查看更多预警 ▼</div>' : ''}
    ${hasExtra ? `<div class="weather-bar-extras" style="display:none">${alerts.slice(2).map(a => `
      <div class="weather-bar-row weather-bar-alert">
        <span class="weather-bar-icon">${a.level.split(' ')[0] || '💡'}</span>
        <span class="weather-bar-text">${escapeHTML(a.detail)}</span>
      </div>
    `).join('')}</div>` : ''}
    <div class="weather-bar-footer">基于您的位置 · 数据仅供参考</div>
  `;

  // 在检测卡片之后插入
  const detectionCard = document.querySelector('.detection-card:last-child');
  if (detectionCard) {
    detectionCard.parentNode.insertBefore(bar, detectionCard.nextSibling);
  } else {
    messagesArea.appendChild(bar);
  }
  scrollToBottom();
}

// ═══════════════════════════════════════════════════════════════
// 📊 历史记录与统计面板（SQLite）
// ═══════════════════════════════════════════════════════════════

function buildHistoryPanel() {
  const overlay = document.createElement('div');
  overlay.className = 'history-overlay';
  overlay.id = 'historyOverlay';
  overlay.innerHTML = `
    <div class="history-panel">
      <div class="history-header">
        <div class="history-title"><i class="fa-solid fa-chart-column"></i> 历史记录</div>
        <button class="history-close" id="historyClose" title="关闭">&times;</button>
      </div>
      <div class="history-tabs">
        <button class="history-tab active" data-tab="stats">📊 统计</button>
        <button class="history-tab" data-tab="detect">🔍 识别</button>
        <button class="history-tab" data-tab="qa">💬 问答</button>
      </div>
      <div class="history-body" id="historyBody"></div>
    </div>
  `;
  document.body.appendChild(overlay);
  return overlay;
}

function openHistory() {
  const overlay = document.getElementById('historyOverlay') || buildHistoryPanel();
  overlay.classList.add('show');
  document.body.style.overflow = 'hidden';
  loadHistoryTab('stats');

  const closeBtn = overlay.querySelector('#historyClose');
  const onClose = () => {
    overlay.classList.remove('show');
    document.body.style.overflow = '';
  };
  closeBtn.onclick = onClose;
  overlay.addEventListener('click', e => {
    if (e.target === overlay) onClose();
  });

  overlay.querySelectorAll('.history-tab').forEach(tab => {
    tab.onclick = () => {
      overlay.querySelectorAll('.history-tab').forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      loadHistoryTab(tab.dataset.tab);
    };
  });
}

function loadHistoryTab(tab) {
  const body = document.getElementById('historyBody');
  if (!body) return;
  // 携带登录 token，让后端按当前用户隔离记录（游客无 token）
  const headers = {};
  const token = sessionStorage.getItem('pest_token');
  if (token) headers['Authorization'] = 'Bearer ' + token;
  const opts = { headers };

  if (tab === 'stats') {
    body.innerHTML = '<div class="history-empty">加载统计中…</div>';
    fetch(`${API_BASE}/history/stats`, opts)
      .then(r => r.json())
      .then(d => renderStats(body, d))
      .catch(() => { body.innerHTML = '<div class="history-empty">统计加载失败</div>'; });
  } else if (tab === 'detect') {
    body.innerHTML = '<div class="history-empty">加载识别记录…</div>';
    fetch(`${API_BASE}/history/detections?limit=30`, opts)
      .then(r => r.json())
      .then(d => renderDetections(body, d))
      .catch(() => { body.innerHTML = '<div class="history-empty">识别记录加载失败</div>'; });
  } else if (tab === 'qa') {
    body.innerHTML = '<div class="history-empty">加载问答记录…</div>';
    fetch(`${API_BASE}/history/qa?limit=30`, opts)
      .then(r => r.json())
      .then(d => renderQa(body, d))
      .catch(() => { body.innerHTML = '<div class="history-empty">问答记录加载失败</div>'; });
  }
}

// 统计视图：数字卡片 + 害虫频率条形 + 每日趋势
function renderStats(body, data) {
  if (!data.success) {
    body.innerHTML = '<div class="history-empty">暂无统计数据</div>';
    return;
  }
  const cards = `
    <div class="stat-cards">
      <div class="stat-card"><div class="stat-num">${data.total_detections || 0}</div><div class="stat-label">总识别</div></div>
      <div class="stat-card"><div class="stat-num">${data.total_qa || 0}</div><div class="stat-label">总问答</div></div>
      <div class="stat-card"><div class="stat-num">${data.unknown_count || 0}</div><div class="stat-label">疑似未知</div></div>
    </div>
  `;

  // 害虫频率条形图（纯 CSS）
  let pestHtml = '<div class="stat-section-title">🐛 害虫识别频率 TOP10</div>';
  const top = data.top_pests || [];
  if (top.length === 0) {
    pestHtml += '<div class="history-empty">暂无识别数据</div>';
  } else {
    const max = top[0].count || 1;
    pestHtml += '<div class="stat-bars">' + top.map(p => `
      <div class="stat-bar-row">
        <span class="stat-bar-name">${escapeHTML(p.name)}</span>
        <div class="stat-bar-track"><div class="stat-bar-fill" style="width:${Math.round(p.count / max * 100)}%"></div></div>
        <span class="stat-bar-count">${p.count}</span>
      </div>
    `).join('') + '</div>';
  }

  // 每日趋势
  let trendHtml = '<div class="stat-section-title">📅 近 7 天识别量</div>';
  const trend = data.daily_trend || [];
  if (trend.length === 0) {
    trendHtml += '<div class="history-empty">暂无数据</div>';
  } else {
    const tmax = Math.max(...trend.map(t => t.count), 1);
    trendHtml += '<div class="stat-trend">' + trend.map(t => `
      <div class="stat-trend-col">
        <div class="stat-trend-bar" style="height:${Math.max(6, Math.round(t.count / tmax * 80))}px"></div>
        <div class="stat-trend-label">${escapeHTML((t.date || '').slice(5))}</div>
      </div>
    `).join('') + '</div>';
  }

  body.innerHTML = cards + pestHtml + trendHtml;
}

// 识别记录视图
function renderDetections(body, data) {
  const records = data.records || [];
  if (records.length === 0) {
    body.innerHTML = '<div class="history-empty">还没有识别记录，去上传一张图片吧 🌾</div>';
    return;
  }
  window.__detectRecords = records;  // 供“查看明细”按 id 取完整数据
  body.innerHTML = records.map(r => {
    let dets;
    try { dets = JSON.parse(r.detections_json || '[]'); } catch (e) { dets = []; }
    const names = dets.map(d => escapeHTML(d.zh_name || d.name)).join('、') || (r.is_unknown ? '未识别/未知' : '未检出');
    const time = escapeHTML((r.created_at || '').slice(5, 16));
    const badge = r.is_unknown ? '<span class="history-badge history-badge-warn">未知</span>' : '';
    // 专家复核反馈（仅模糊/失败时给出建设性提示，识别成功不打扰）
    let expertTip = '';
    if (r.expert_mark === 'ambiguous') {
      expertTip = '<div class="history-expert-tip">🔬 本条识别已由植保专家复核：⚠️ 识别模糊，建议重新拍摄更清晰的图片</div>';
    } else if (r.expert_mark === 'failed') {
      expertTip = '<div class="history-expert-tip history-expert-tip-warn">🔬 本条识别已由植保专家复核：❌ 未能有效识别，建议补充拍摄细节或咨询当地农技站</div>';
    }
    return `
      <div class="history-item">
        <div class="history-item-icon">🔍</div>
        <div class="history-item-main">
          <div class="history-item-title">${names} ${badge}</div>
          <div class="history-item-sub">${r.total} 个目标 · ${r.elapsed_ms}ms</div>
          ${expertTip}
          <button class="history-detail-btn" onclick="window.__showDetectDetail(${r.id})">查看识别明细 →</button>
        </div>
        <div class="history-item-time">${time}</div>
      </div>
    `;
  }).join('');
}

// 问答记录视图
function renderQa(body, data) {
  const records = data.records || [];
  if (records.length === 0) {
    body.innerHTML = '<div class="history-empty">还没有问答记录，去问一个问题吧 💬</div>';
    return;
  }
  window.__qaRecords = records;  // 供“查看完整对话”按 id 取完整数据
  body.innerHTML = records.map(r => {
    const q = escapeHTML((r.question || '').slice(0, 40));
    const a = escapeHTML((r.answer || '').replace(/\s+/g, ' ').slice(0, 60));
    const time = escapeHTML((r.created_at || '').slice(5, 16));
    const llm = r.used_llm ? '<span class="history-badge">AI</span>' : '';
    return `
      <div class="history-item">
        <div class="history-item-icon">💬</div>
        <div class="history-item-main">
          <div class="history-item-title">${q} ${llm}</div>
          <div class="history-item-sub">${a}</div>
          <button class="history-detail-btn" onclick="window.__showQaDetail(${r.id})">查看完整对话 →</button>
        </div>
        <div class="history-item-time">${time}</div>
      </div>
    `;
  }).join('');
}

// ── 历史详情弹窗（完整对话 / 识别明细） ───────────────────────
function getDetailOverlay() {
  let o = document.getElementById('detailOverlay');
  if (!o) {
    o = document.createElement('div');
    o.className = 'detail-overlay';
    o.id = 'detailOverlay';
    o.innerHTML = `
      <div class="detail-panel">
        <div class="detail-header">
          <span class="detail-title" id="detailTitle"></span>
          <button class="detail-close" id="detailClose">&times;</button>
        </div>
        <div class="detail-body" id="detailBody"></div>
      </div>`;
    document.body.appendChild(o);
    o.querySelector('#detailClose').onclick = () => { o.style.display = 'none'; };
    o.addEventListener('click', e => { if (e.target === o) o.style.display = 'none'; });
  }
  return o;
}

window.__showQaDetail = function (id) {
  const r = (window.__qaRecords || []).find(x => x.id === id);
  if (!r) return;
  const o = getDetailOverlay();
  document.getElementById('detailTitle').textContent = '💬 对话详情';
  document.getElementById('detailBody').innerHTML = `
    <div class="detail-meta">${escapeHTML(r.created_at || '')} · 关联: ${escapeHTML(r.pest_name || '—')} ${r.used_llm ? '<span class="history-badge">AI</span>' : '<span class="history-badge">知识库</span>'}</div>
    <div class="detail-question">❓ ${escapeHTML(r.question || '')}</div>
    <div class="detail-answer">${escapeHTML(r.answer || '')}</div>
  `;
  o.style.display = 'flex';
};

window.__showDetectDetail = function (id) {
  const r = (window.__detectRecords || []).find(x => x.id === id);
  if (!r) return;
  let dets = [];
  try { dets = JSON.parse(r.detections_json || '[]'); } catch (e) {}
  const o = getDetailOverlay();
  document.getElementById('detailTitle').textContent = '🔍 识别明细';
  const list = dets.length
    ? dets.map(d => `
      <div class="detail-det-row">
        <span class="detail-det-name">${escapeHTML(d.zh_name || d.name)}</span>
        <span class="detail-det-conf">置信度 ${Math.round(d.confidence)}%</span>
      </div>`).join('')
    : '<div class="detail-empty">该次未检出害虫</div>';
  document.getElementById('detailBody').innerHTML = `
    <div class="detail-meta">${escapeHTML(r.created_at || '')} · 共 ${r.total} 个目标 · 耗时 ${r.elapsed_ms}ms · ${r.is_unknown ? '<span class="history-badge history-badge-warn">模型判定未知</span>' : '<span class="history-badge">识别正常</span>'}</div>
    ${list}
  `;
  o.style.display = 'flex';
};

// 绑定历史按钮
(function initHistory() {
  const btn = document.getElementById('historyBtn');
  if (btn) btn.addEventListener('click', openHistory);
})();

// ── 启动应用 ──────────────────────────────────────────────────────
initApp();
