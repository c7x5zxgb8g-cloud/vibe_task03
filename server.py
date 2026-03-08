"""FastAPI 后端服务：提供 AI 客服对话 API + 静态页面服务。"""

from __future__ import annotations

import json
import os
import uuid
import shutil
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import Config
from rag_engine import RAGEngine, get_available_materials
from vectorstore import ChromaStore
from utils.logger import get_logger

logger = get_logger(__name__)

# In-memory session store for conversation history
_sessions: dict[str, list[dict]] = {}
_rag_engine: RAGEngine | None = None

# Paths
_BASE_DIR = os.path.dirname(__file__)
_WECHAT_CONFIG_PATH = os.path.join(_BASE_DIR, "wechat_config.json")
_WECHAT_IMAGE_DIR = os.path.join(_BASE_DIR, "static", "wechat")
_FEEDBACK_PATH = os.path.join(_BASE_DIR, "feedback.json")
_FEEDBACK_RULES_PATH = os.path.join(_BASE_DIR, "feedback_rules.json")
_MATERIALS_DIR = os.path.join(_BASE_DIR, "docs", "理博基金知识库")


def _ensure_dirs():
    os.makedirs(_WECHAT_IMAGE_DIR, exist_ok=True)


def _load_json(path, default=None):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return default or {}


def _save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _rag_engine
    _ensure_dirs()
    store = ChromaStore()
    _rag_engine = RAGEngine(store=store)
    stats = store.get_stats()
    logger.info(f"Server started. Knowledge base: {stats['total_documents']} documents")
    yield
    _rag_engine = None
    _sessions.clear()


app = FastAPI(title="证券AI客服", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic models ─────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    reply: str
    session_id: str


class StatsResponse(BaseModel):
    total_documents: int
    collection_name: str
    active_sessions: int


class FeedbackRequest(BaseModel):
    session_id: str
    message_index: int
    rating: str  # "good" or "bad"
    comment: str = ""


class FeedbackRuleRequest(BaseModel):
    rule: str
    active: bool = True


# ── API Endpoints ────────────────────────────────────────────────

@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Handle a chat message and return AI response."""
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="消息不能为空")

    session_id = req.session_id or str(uuid.uuid4())
    history = _sessions.setdefault(session_id, [])

    reply = _rag_engine.chat(query=req.message, history=history)

    history.append({"role": "user", "content": req.message})
    history.append({"role": "assistant", "content": reply})

    if len(history) > 40:
        _sessions[session_id] = history[-20:]

    return ChatResponse(reply=reply, session_id=session_id)


@app.post("/api/session/reset")
async def reset_session(session_id: str | None = None):
    """Reset a conversation session."""
    if session_id and session_id in _sessions:
        del _sessions[session_id]
    return {"status": "ok"}


@app.get("/api/stats", response_model=StatsResponse)
async def stats():
    """Get knowledge base statistics."""
    store_stats = _rag_engine._store.get_stats()
    return StatsResponse(
        total_documents=store_stats["total_documents"],
        collection_name=store_stats["collection_name"],
        active_sessions=len(_sessions),
    )


# ── WeChat Card API ─────────────────────────────────────────────

@app.get("/api/wechat-card")
async def get_wechat_card():
    """Get current WeChat card configuration."""
    config = _load_json(_WECHAT_CONFIG_PATH, {"wechat_id": "", "image_url": ""})
    return config


@app.post("/api/wechat-card")
async def update_wechat_card(
    wechat_id: str = Form(""),
    image: UploadFile | None = File(None),
):
    """Update WeChat card: upload image and/or set WeChat ID."""
    config = _load_json(_WECHAT_CONFIG_PATH, {"wechat_id": "", "image_url": ""})

    if wechat_id:
        config["wechat_id"] = wechat_id

    if image and image.filename:
        _ensure_dirs()
        ext = os.path.splitext(image.filename)[1].lower() or ".jpg"
        save_name = f"wechat_card{ext}"
        save_path = os.path.join(_WECHAT_IMAGE_DIR, save_name)
        with open(save_path, "wb") as f:
            content = await image.read()
            f.write(content)
        config["image_url"] = f"/static/wechat/{save_name}"

    _save_json(_WECHAT_CONFIG_PATH, config)
    return config


# ── Materials API ────────────────────────────────────────────────

@app.get("/api/materials")
async def list_materials():
    """List available knowledge base materials for sharing."""
    materials = get_available_materials()
    return {"materials": materials}


@app.get("/api/materials/{filename:path}")
async def download_material(filename: str):
    """Download a specific material file."""
    file_path = os.path.join(_MATERIALS_DIR, filename)
    # Security: prevent directory traversal
    real_path = os.path.realpath(file_path)
    real_dir = os.path.realpath(_MATERIALS_DIR)
    if not real_path.startswith(real_dir):
        raise HTTPException(status_code=403, detail="Access denied")
    if not os.path.isfile(real_path):
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(
        real_path,
        filename=filename,
        media_type="application/octet-stream",
    )


# ── Feedback API ─────────────────────────────────────────────────

@app.post("/api/feedback")
async def submit_feedback(req: FeedbackRequest):
    """Submit feedback for a specific AI response."""
    data = _load_json(_FEEDBACK_PATH, {"feedbacks": []})
    data["feedbacks"].append({
        "session_id": req.session_id,
        "message_index": req.message_index,
        "rating": req.rating,
        "comment": req.comment,
        "timestamp": datetime.now().isoformat(),
    })
    _save_json(_FEEDBACK_PATH, data)
    return {"status": "ok"}


@app.get("/api/feedback/rules")
async def get_feedback_rules():
    """Get all feedback-based reply rules."""
    data = _load_json(_FEEDBACK_RULES_PATH, {"rules": []})
    return data


@app.post("/api/feedback/rules")
async def add_feedback_rule(req: FeedbackRuleRequest):
    """Add or update a feedback-based reply rule."""
    data = _load_json(_FEEDBACK_RULES_PATH, {"rules": []})
    data["rules"].append({
        "rule": req.rule,
        "active": req.active,
        "created_at": datetime.now().isoformat(),
    })
    _save_json(_FEEDBACK_RULES_PATH, data)
    # Reload rules in RAG engine
    if _rag_engine:
        _rag_engine.reload_rules()
    return {"status": "ok", "total_rules": len(data["rules"])}


@app.delete("/api/feedback/rules/{index}")
async def delete_feedback_rule(index: int):
    """Delete a feedback rule by index."""
    data = _load_json(_FEEDBACK_RULES_PATH, {"rules": []})
    if 0 <= index < len(data["rules"]):
        removed = data["rules"].pop(index)
        _save_json(_FEEDBACK_RULES_PATH, data)
        if _rag_engine:
            _rag_engine.reload_rules()
        return {"status": "ok", "removed": removed}
    raise HTTPException(status_code=404, detail="规则不存在")


# ── Serve static files ───────────────────────────────────────────

# Mount static directory for WeChat card images
if not os.path.exists(os.path.join(_BASE_DIR, "static", "wechat")):
    os.makedirs(os.path.join(_BASE_DIR, "static", "wechat"), exist_ok=True)
app.mount("/static", StaticFiles(directory=os.path.join(_BASE_DIR, "static")), name="static")


# ── Web UI ───────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return _CHAT_HTML


_CHAT_HTML = """\
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>理博基金 - 智能客服</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", "Microsoft YaHei", sans-serif; background: #f5f6fa; height: 100vh; display: flex; flex-direction: column; }

  .header { background: linear-gradient(135deg, #1a3a5c, #2c5f8a); color: #fff; padding: 14px 24px; display: flex; align-items: center; gap: 14px; box-shadow: 0 2px 12px rgba(0,0,0,.12); }
  .header .logo { font-size: 20px; font-weight: 700; letter-spacing: 2px; }
  .header .logo span { font-size: 12px; font-weight: 400; opacity: .7; letter-spacing: 1px; margin-left: 8px; }
  .header .actions { margin-left: auto; display: flex; gap: 12px; align-items: center; }
  .header .stats { font-size: 12px; opacity: .7; }
  .reset-btn, .settings-btn { background: rgba(255,255,255,.15); border: 1px solid rgba(255,255,255,.25); color: #fff; border-radius: 6px; padding: 5px 12px; font-size: 12px; cursor: pointer; transition: all .2s; }
  .reset-btn:hover, .settings-btn:hover { background: rgba(255,255,255,.25); }

  .chat-container { flex: 1; overflow-y: auto; padding: 20px 16px; display: flex; flex-direction: column; gap: 16px; max-width: 820px; width: 100%; margin: 0 auto; }

  .msg { display: flex; gap: 10px; max-width: 88%; animation: fadeIn .3s ease; }
  .msg.user { align-self: flex-end; flex-direction: row-reverse; }
  .msg.assistant { align-self: flex-start; }

  .avatar { width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 14px; flex-shrink: 0; font-weight: 600; }
  .msg.user .avatar { background: #2c5f8a; color: #fff; }
  .msg.assistant .avatar { background: linear-gradient(135deg, #e8f0fe, #d4e4f7); color: #1a3a5c; border: 1px solid #c8d8e8; }

  .msg-content { display: flex; flex-direction: column; gap: 6px; }
  .bubble { padding: 12px 16px; border-radius: 16px; font-size: 14px; line-height: 1.8; white-space: pre-wrap; word-break: break-word; }
  .msg.user .bubble { background: #2c5f8a; color: #fff; border-bottom-right-radius: 4px; }
  .msg.assistant .bubble { background: #fff; color: #333; border-bottom-left-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,.06); }

  .typing .bubble::after { content: '\\25CF\\25CF\\25CF'; animation: blink 1.2s infinite; letter-spacing: 3px; color: #999; }
  @keyframes blink { 0%,100%{opacity:.2} 50%{opacity:1} }
  @keyframes fadeIn { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:none} }

  /* Feedback buttons */
  .feedback-row { display: flex; gap: 8px; align-items: center; padding: 2px 0; }
  .feedback-btn { background: none; border: 1px solid #e0e0e0; border-radius: 14px; padding: 3px 10px; font-size: 13px; cursor: pointer; color: #888; transition: all .2s; display: flex; align-items: center; gap: 4px; }
  .feedback-btn:hover { border-color: #2c5f8a; color: #2c5f8a; }
  .feedback-btn.active-good { border-color: #4caf50; color: #4caf50; background: #e8f5e9; }
  .feedback-btn.active-bad { border-color: #f44336; color: #f44336; background: #fce4ec; }
  .feedback-comment { display: none; margin-top: 4px; }
  .feedback-comment.show { display: flex; gap: 6px; }
  .feedback-comment input { flex: 1; border: 1px solid #dde3ea; border-radius: 8px; padding: 5px 10px; font-size: 12px; outline: none; }
  .feedback-comment button { border: none; background: #2c5f8a; color: #fff; border-radius: 8px; padding: 5px 12px; font-size: 12px; cursor: pointer; }

  /* WeChat card */
  .wechat-card { background: linear-gradient(135deg, #f0f9f0, #e8f5e8); border: 1px solid #c8e6c9; border-radius: 12px; padding: 14px 16px; margin: 8px 0; display: flex; align-items: center; gap: 14px; animation: fadeIn .3s ease; }
  .wechat-card img { width: 80px; height: 80px; border-radius: 8px; object-fit: cover; border: 1px solid #ddd; }
  .wechat-card .info { flex: 1; }
  .wechat-card .info .title { font-size: 14px; font-weight: 600; color: #2e7d32; margin-bottom: 4px; }
  .wechat-card .info .wechat-id { font-size: 13px; color: #555; margin-bottom: 4px; }
  .wechat-card .info .hint { font-size: 11px; color: #999; }
  .wechat-card-placeholder { background: linear-gradient(135deg, #f0f9f0, #e8f5e8); border: 1px solid #c8e6c9; border-radius: 12px; padding: 14px 16px; margin: 8px 0; text-align: center; }
  .wechat-card-placeholder .title { font-size: 14px; font-weight: 600; color: #2e7d32; margin-bottom: 4px; }
  .wechat-card-placeholder .hint { font-size: 12px; color: #888; }

  /* Material card */
  .material-card { background: linear-gradient(135deg, #eff6ff, #e0ecff); border: 1px solid #c4d5f0; border-radius: 12px; padding: 12px 16px; margin: 6px 0; display: flex; align-items: center; gap: 12px; animation: fadeIn .3s ease; cursor: pointer; transition: all .2s; }
  .material-card:hover { box-shadow: 0 2px 8px rgba(44,95,138,.15); border-color: #2c5f8a; }
  .material-card .icon { font-size: 24px; }
  .material-card .name { flex: 1; font-size: 13px; color: #333; font-weight: 500; }
  .material-card .dl-btn { background: #2c5f8a; color: #fff; border: none; border-radius: 6px; padding: 5px 14px; font-size: 12px; cursor: pointer; transition: all .2s; }
  .material-card .dl-btn:hover { background: #1a3a5c; }

  .input-area { background: #fff; border-top: 1px solid #e4e8ee; padding: 14px 16px; }
  .input-wrap { max-width: 820px; margin: 0 auto; display: flex; gap: 10px; align-items: flex-end; }
  .input-wrap textarea { flex: 1; border: 1.5px solid #dde3ea; border-radius: 12px; padding: 10px 16px; font-size: 14px; resize: none; outline: none; max-height: 120px; line-height: 1.5; font-family: inherit; transition: border .2s; }
  .input-wrap textarea:focus { border-color: #2c5f8a; box-shadow: 0 0 0 3px rgba(44,95,138,.1); }

  .send-btn { width: 42px; height: 42px; border-radius: 50%; border: none; background: #2c5f8a; color: #fff; cursor: pointer; display: flex; align-items: center; justify-content: center; transition: all .2s; flex-shrink: 0; }
  .send-btn:hover { background: #1a3a5c; transform: scale(1.05); }
  .send-btn:disabled { background: #c8d0d8; cursor: not-allowed; transform: none; }

  .welcome { text-align: center; padding: 50px 20px; color: #666; }
  .welcome .brand { font-size: 28px; font-weight: 700; color: #1a3a5c; margin-bottom: 4px; letter-spacing: 3px; }
  .welcome .slogan { font-size: 13px; color: #999; letter-spacing: 2px; margin-bottom: 16px; }
  .welcome .intro { font-size: 15px; color: #555; margin-bottom: 24px; line-height: 1.6; }
  .suggestions { display: flex; flex-wrap: wrap; gap: 10px; justify-content: center; max-width: 600px; margin: 0 auto; }
  .suggestions button { background: #fff; border: 1.5px solid #dde3ea; border-radius: 22px; padding: 9px 18px; font-size: 13px; cursor: pointer; transition: all .2s; color: #444; }
  .suggestions button:hover { border-color: #2c5f8a; color: #2c5f8a; background: #f0f5fa; box-shadow: 0 2px 8px rgba(44,95,138,.08); }

  .footer { text-align: center; padding: 6px; font-size: 11px; color: #bbb; background: #fff; border-top: 1px solid #f0f0f0; }

  /* Settings Modal */
  .modal-overlay { display: none; position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,.4); z-index: 1000; animation: fadeIn .2s ease; }
  .modal-overlay.show { display: flex; align-items: center; justify-content: center; }
  .modal { background: #fff; border-radius: 16px; padding: 28px; max-width: 520px; width: 90%; max-height: 80vh; overflow-y: auto; box-shadow: 0 20px 60px rgba(0,0,0,.2); }
  .modal h3 { font-size: 18px; color: #1a3a5c; margin-bottom: 20px; }
  .modal-section { margin-bottom: 20px; }
  .modal-section h4 { font-size: 14px; color: #555; margin-bottom: 10px; font-weight: 600; }
  .modal-section label { display: block; font-size: 13px; color: #666; margin-bottom: 6px; }
  .modal-section input[type="text"] { width: 100%; border: 1.5px solid #dde3ea; border-radius: 8px; padding: 8px 12px; font-size: 13px; outline: none; transition: border .2s; }
  .modal-section input[type="text"]:focus { border-color: #2c5f8a; }
  .modal-section input[type="file"] { font-size: 13px; }
  .modal-preview { margin-top: 8px; }
  .modal-preview img { max-width: 120px; border-radius: 8px; border: 1px solid #ddd; }
  .modal-btn-row { display: flex; gap: 10px; justify-content: flex-end; margin-top: 20px; }
  .modal-btn { padding: 8px 20px; border-radius: 8px; font-size: 13px; cursor: pointer; transition: all .2s; }
  .modal-btn.primary { background: #2c5f8a; color: #fff; border: none; }
  .modal-btn.primary:hover { background: #1a3a5c; }
  .modal-btn.secondary { background: #f5f5f5; color: #666; border: 1px solid #ddd; }
  .modal-btn.secondary:hover { background: #eee; }

  /* Rules list */
  .rules-list { margin-top: 8px; }
  .rule-item { display: flex; align-items: center; gap: 8px; padding: 8px 10px; background: #f8f9fa; border-radius: 8px; margin-bottom: 6px; font-size: 13px; }
  .rule-item .rule-text { flex: 1; color: #333; }
  .rule-item .rule-del { background: none; border: none; color: #e53935; cursor: pointer; font-size: 16px; padding: 0 4px; }
  .add-rule-row { display: flex; gap: 8px; margin-top: 8px; }
  .add-rule-row input { flex: 1; border: 1.5px solid #dde3ea; border-radius: 8px; padding: 8px 12px; font-size: 13px; outline: none; }
  .add-rule-row button { background: #2c5f8a; color: #fff; border: none; border-radius: 8px; padding: 8px 16px; font-size: 13px; cursor: pointer; white-space: nowrap; }
</style>
</head>
<body>

<div class="header">
  <div class="logo">理博基金<span>LESS IS MORE</span></div>
  <div class="actions">
    <span class="stats" id="stats"></span>
    <button class="settings-btn" onclick="openSettings()">⚙ 设置</button>
    <button class="reset-btn" onclick="resetChat()">新对话</button>
  </div>
</div>

<div class="chat-container" id="chat">
  <div class="welcome">
    <div class="brand">理博小助手</div>
    <div class="slogan">辞简理博 / LESS IS MORE</div>
    <div class="intro">您好！我是理博基金的智能客服助手，可以为您解答关于公司、策略、产品、投资流程等方面的问题。</div>
    <div class="suggestions">
      <button onclick="ask(this.textContent)">理博基金是一家什么样的公司？</button>
      <button onclick="ask(this.textContent)">你们有哪些投资策略？</button>
      <button onclick="ask(this.textContent)">2025年度各策略收益如何？</button>
      <button onclick="ask(this.textContent)">理博万象7号产品介绍一下</button>
      <button onclick="ask(this.textContent)">如何购买你们的基金产品？</button>
      <button onclick="ask(this.textContent)">资金安全性如何保障？</button>
    </div>
  </div>
</div>

<div class="input-area">
  <div class="input-wrap">
    <textarea id="input" rows="1" placeholder="请输入您想了解的问题..." onkeydown="handleKey(event)"></textarea>
    <button class="send-btn" id="sendBtn" onclick="send()">
      <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/></svg>
    </button>
  </div>
</div>

<div class="footer">杭州理博私募基金管理有限公司 | 基金业协会备案编码：P1072890</div>

<!-- Settings Modal -->
<div class="modal-overlay" id="settingsModal">
  <div class="modal">
    <h3>⚙ 系统设置</h3>

    <div class="modal-section">
      <h4>📱 微信名片设置</h4>
      <label>微信号</label>
      <input type="text" id="wechatIdInput" placeholder="请输入微信号">
      <label style="margin-top:10px">名片图片</label>
      <input type="file" id="wechatImageInput" accept="image/*">
      <div class="modal-preview" id="wechatPreview"></div>
    </div>

    <div class="modal-section">
      <h4>📝 回复规则管理</h4>
      <p style="font-size:12px;color:#999;margin-bottom:8px;">添加规则来调整 AI 的回复策略，例如"不要推荐发邮件"、"优先推荐加微信"</p>
      <div class="rules-list" id="rulesList"></div>
      <div class="add-rule-row">
        <input type="text" id="newRuleInput" placeholder="输入新规则...">
        <button onclick="addRule()">添加</button>
      </div>
    </div>

    <div class="modal-btn-row">
      <button class="modal-btn secondary" onclick="closeSettings()">关闭</button>
      <button class="modal-btn primary" onclick="saveSettings()">保存微信设置</button>
    </div>
  </div>
</div>

<script>
const chatEl = document.getElementById('chat');
const inputEl = document.getElementById('input');
const sendBtn = document.getElementById('sendBtn');
let sessionId = null;
let sending = false;
let msgIndex = 0;
let wechatConfig = { wechat_id: '', image_url: '' };

// Load initial data
fetch('/api/stats').then(r=>r.json()).then(d=>{
  document.getElementById('stats').textContent = `知识库 ${d.total_documents} 条`;
}).catch(()=>{});

fetch('/api/wechat-card').then(r=>r.json()).then(d=>{
  wechatConfig = d;
}).catch(()=>{});

inputEl.addEventListener('input', () => {
  inputEl.style.height = 'auto';
  inputEl.style.height = Math.min(inputEl.scrollHeight, 120) + 'px';
});

function handleKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
}

function ask(text) { inputEl.value = text; send(); }

async function resetChat() {
  if (sessionId) {
    fetch('/api/session/reset', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({session_id: sessionId}) }).catch(()=>{});
  }
  sessionId = null;
  msgIndex = 0;
  chatEl.innerHTML = `
    <div class="welcome">
      <div class="brand">理博小助手</div>
      <div class="slogan">辞简理博 / LESS IS MORE</div>
      <div class="intro">您好！我是理博基金的智能客服助手，可以为您解答关于公司、策略、产品、投资流程等方面的问题。</div>
      <div class="suggestions">
        <button onclick="ask(this.textContent)">理博基金是一家什么样的公司？</button>
        <button onclick="ask(this.textContent)">你们有哪些投资策略？</button>
        <button onclick="ask(this.textContent)">2025年度各策略收益如何？</button>
        <button onclick="ask(this.textContent)">理博万象7号产品介绍一下</button>
        <button onclick="ask(this.textContent)">如何购买你们的基金产品？</button>
        <button onclick="ask(this.textContent)">资金安全性如何保障？</button>
      </div>
    </div>`;
}

async function send() {
  const msg = inputEl.value.trim();
  if (!msg || sending) return;
  sending = true;
  sendBtn.disabled = true;

  const welcome = chatEl.querySelector('.welcome');
  if (welcome) welcome.remove();

  addMsg('user', msg);
  inputEl.value = '';
  inputEl.style.height = 'auto';

  const typingEl = addMsg('assistant', '', true);

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: msg, session_id: sessionId })
    });
    const data = await res.json();
    sessionId = data.session_id;
    typingEl.remove();
    addMsg('assistant', data.reply);
    msgIndex++;
  } catch (e) {
    typingEl.remove();
    addMsg('assistant', '网络出现了一点问题，请稍后再试。');
  }
  sending = false;
  sendBtn.disabled = false;
  inputEl.focus();
}

function addMsg(role, text, typing = false) {
  const div = document.createElement('div');
  div.className = `msg ${role}` + (typing ? ' typing' : '');
  const avatarText = role === 'user' ? 'You' : 'LB';

  if (typing) {
    div.innerHTML = `<div class="avatar">${avatarText}</div><div class="msg-content"><div class="bubble"></div></div>`;
  } else if (role === 'user') {
    div.innerHTML = `<div class="avatar">${avatarText}</div><div class="msg-content"><div class="bubble">${escHtml(text)}</div></div>`;
  } else {
    // Parse special markers for assistant messages
    const { cleanText, hasWechat, materials } = parseMarkers(text);
    const currentIdx = msgIndex;

    let html = `<div class="avatar">${avatarText}</div><div class="msg-content"><div class="bubble">${escHtml(cleanText)}</div>`;

    // Add material cards
    for (const mat of materials) {
      html += `<div class="material-card" onclick="downloadMaterial('${escAttr(mat)}')">
        <span class="icon">📄</span>
        <span class="name">${escHtml(mat)}</span>
        <button class="dl-btn">下载</button>
      </div>`;
    }

    // Add WeChat card
    if (hasWechat) {
      html += renderWechatCard();
    }

    // Add feedback buttons
    html += `<div class="feedback-row">
      <button class="feedback-btn" onclick="feedback(this, ${currentIdx}, 'good')" title="有帮助">👍</button>
      <button class="feedback-btn" onclick="feedback(this, ${currentIdx}, 'bad')" title="需改进">👎</button>
    </div>
    <div class="feedback-comment" id="fb-comment-${currentIdx}">
      <input type="text" placeholder="请告诉我们如何改进..." id="fb-input-${currentIdx}">
      <button onclick="submitComment(${currentIdx})">提交</button>
    </div>`;

    html += `</div>`;
    div.innerHTML = html;
  }

  chatEl.appendChild(div);
  chatEl.scrollTop = chatEl.scrollHeight;
  return div;
}

function parseMarkers(text) {
  let cleanText = text;
  let hasWechat = false;
  const materials = [];

  // Check for [微信名片]
  if (cleanText.includes('[微信名片]')) {
    hasWechat = true;
    cleanText = cleanText.replace(/\\[微信名片\\]/g, '').trim();
  }

  // Check for [材料推荐:filename]
  const matRegex = /\\[材料推荐[:：]([^\\]]+)\\]/g;
  let match;
  while ((match = matRegex.exec(cleanText)) !== null) {
    materials.push(match[1].trim());
  }
  cleanText = cleanText.replace(/\\[材料推荐[:：][^\\]]+\\]/g, '').trim();

  return { cleanText, hasWechat, materials };
}

function renderWechatCard() {
  if (wechatConfig.image_url && wechatConfig.wechat_id) {
    return `<div class="wechat-card">
      <img src="${wechatConfig.image_url}" alt="微信名片">
      <div class="info">
        <div class="title">📱 添加微信咨询</div>
        <div class="wechat-id">微信号：${escHtml(wechatConfig.wechat_id)}</div>
        <div class="hint">长按识别二维码或搜索微信号添加</div>
      </div>
    </div>`;
  } else if (wechatConfig.wechat_id) {
    return `<div class="wechat-card-placeholder">
      <div class="title">📱 添加微信咨询</div>
      <div class="hint">微信号：${escHtml(wechatConfig.wechat_id)}</div>
    </div>`;
  } else if (wechatConfig.image_url) {
    return `<div class="wechat-card">
      <img src="${wechatConfig.image_url}" alt="微信名片">
      <div class="info">
        <div class="title">📱 添加微信咨询</div>
        <div class="hint">长按识别二维码添加</div>
      </div>
    </div>`;
  } else {
    return `<div class="wechat-card-placeholder">
      <div class="title">📱 欢迎添加微信咨询</div>
      <div class="hint">请联系管理员在设置中配置微信名片</div>
    </div>`;
  }
}

function downloadMaterial(filename) {
  window.open('/api/materials/' + encodeURIComponent(filename), '_blank');
}

function feedback(btn, idx, rating) {
  const row = btn.parentElement;
  row.querySelectorAll('.feedback-btn').forEach(b => {
    b.classList.remove('active-good', 'active-bad');
  });
  btn.classList.add(rating === 'good' ? 'active-good' : 'active-bad');

  fetch('/api/feedback', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId || '', message_index: idx, rating: rating, comment: '' })
  }).catch(()=>{});

  // Show comment box for bad ratings
  if (rating === 'bad') {
    const commentEl = document.getElementById('fb-comment-' + idx);
    if (commentEl) commentEl.classList.add('show');
  }
}

function submitComment(idx) {
  const input = document.getElementById('fb-input-' + idx);
  const comment = input.value.trim();
  if (!comment) return;

  fetch('/api/feedback', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId || '', message_index: idx, rating: 'bad', comment: comment })
  }).then(()=>{
    const commentEl = document.getElementById('fb-comment-' + idx);
    commentEl.innerHTML = '<span style="font-size:12px;color:#4caf50;">感谢您的反馈！</span>';
  }).catch(()=>{});
}

// Settings modal
function openSettings() {
  document.getElementById('settingsModal').classList.add('show');
  document.getElementById('wechatIdInput').value = wechatConfig.wechat_id || '';
  if (wechatConfig.image_url) {
    document.getElementById('wechatPreview').innerHTML = `<img src="${wechatConfig.image_url}" alt="preview">`;
  }
  loadRules();
}

function closeSettings() {
  document.getElementById('settingsModal').classList.remove('show');
}

async function saveSettings() {
  const formData = new FormData();
  const wechatId = document.getElementById('wechatIdInput').value.trim();
  const imageInput = document.getElementById('wechatImageInput');

  formData.append('wechat_id', wechatId);
  if (imageInput.files.length > 0) {
    formData.append('image', imageInput.files[0]);
  }

  try {
    const res = await fetch('/api/wechat-card', { method: 'POST', body: formData });
    wechatConfig = await res.json();
    alert('微信设置已保存！');
    if (wechatConfig.image_url) {
      document.getElementById('wechatPreview').innerHTML = `<img src="${wechatConfig.image_url}" alt="preview">`;
    }
  } catch(e) {
    alert('保存失败，请重试');
  }
}

async function loadRules() {
  try {
    const res = await fetch('/api/feedback/rules');
    const data = await res.json();
    const list = document.getElementById('rulesList');
    list.innerHTML = '';
    (data.rules || []).forEach((r, i) => {
      list.innerHTML += `<div class="rule-item">
        <span class="rule-text">${escHtml(r.rule)}</span>
        <button class="rule-del" onclick="deleteRule(${i})">✕</button>
      </div>`;
    });
  } catch(e) {}
}

async function addRule() {
  const input = document.getElementById('newRuleInput');
  const rule = input.value.trim();
  if (!rule) return;

  try {
    await fetch('/api/feedback/rules', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ rule: rule, active: true })
    });
    input.value = '';
    loadRules();
  } catch(e) {
    alert('添加失败');
  }
}

async function deleteRule(index) {
  try {
    await fetch('/api/feedback/rules/' + index, { method: 'DELETE' });
    loadRules();
  } catch(e) {
    alert('删除失败');
  }
}

function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/\\n/g,'<br>');
}

function escAttr(s) {
  return s.replace(/'/g, "\\\\'").replace(/"/g, '&quot;');
}
</script>
</body>
</html>
"""
