"""FastAPI 后端服务：提供 AI 客服对话 API + 静态页面服务。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import secrets
import uuid
import shutil
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Header, Query
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


_KB_STAGING_DIR = os.path.join(_BASE_DIR, "uploads", "kb_staging")


def _ensure_dirs():
    os.makedirs(_WECHAT_IMAGE_DIR, exist_ok=True)
    os.makedirs(_KB_STAGING_DIR, exist_ok=True)


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

    # Initialize leads database
    from database import init_db
    init_db()

    # Start background message processor
    from background_tasks import start_message_processor, stop_message_processor, set_rag_engine
    set_rag_engine(_rag_engine)
    processor_task = asyncio.create_task(start_message_processor())

    yield

    # Stop background processor
    stop_message_processor()
    processor_task.cancel()

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


# ── Webhook & Admin models ─────────────────────────────────────

class WebhookMessageRequest(BaseModel):
    sender_id: str
    sender_name: str = ""
    group_id: str = ""
    group_name: str = ""
    content: str
    msg_type: str = "text"
    msg_id: str | None = None
    timestamp: str | None = None


class WebhookMessageSentRequest(BaseModel):
    follow_up_id: int
    success: bool = True
    error: str = ""


class AdminLoginRequest(BaseModel):
    password: str


class FollowUpConfirmRequest(BaseModel):
    admin_note: str = ""


class FollowUpSendRequest(BaseModel):
    follow_up_id: int
    message: str = ""


class KBChunkEditRequest(BaseModel):
    category: str = ""
    keywords: str = ""
    summary: str = ""
    importance: str = "中"
    related_products: str = ""


class TeamMemberRequest(BaseModel):
    wechat_id: str
    name: str = ""


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


# ── Webhook API (for WeChat robot) ──────────────────────────────

@app.post("/api/webhook/message")
async def webhook_receive_message(req: WebhookMessageRequest):
    """Receive a group chat message from the WeChat robot."""
    from database import upsert_customer, insert_message

    # Optional: verify webhook secret
    # (robot can pass secret as query param or header if configured)

    # Filter out messages from team members (bot + sales staff)
    from database import get_team_member_ids
    team_ids = get_team_member_ids()
    if req.sender_id in team_ids:
        logger.info(f"Ignored team member message from: {req.sender_id}")
        return {"status": "ignored", "reason": "team_member"}

    customer_id = upsert_customer(
        wechat_user_id=req.sender_id,
        name=req.sender_name or req.sender_id,
        group_chat_id=req.group_id,
        group_chat_name=req.group_name,
    )

    msg_id = insert_message(
        customer_id=customer_id,
        content=req.content,
        msg_type=req.msg_type,
        group_chat_id=req.group_id,
        msg_id=req.msg_id,
        received_at=req.timestamp,
    )

    if msg_id is None:
        return {"status": "duplicate", "message": "消息已存在"}

    return {"status": "ok", "message_id": msg_id, "customer_id": customer_id}


@app.get("/api/webhook/pending-messages")
async def webhook_get_pending_messages():
    """Get follow-up messages ready to be sent by the robot."""
    from database import get_pending_follow_ups

    base_url = Config.ADMIN_BASE_URL.rstrip("/")

    pending = get_pending_follow_ups()
    messages = []
    for f in pending:
        # Parse attachment files and build full download URLs
        attachment_files_raw = f.get("attachment_files", "[]")
        try:
            attachment_files = json.loads(attachment_files_raw) if attachment_files_raw else []
        except (json.JSONDecodeError, TypeError):
            attachment_files = []

        file_downloads = []
        for fname in attachment_files:
            if fname.startswith("__wechat_card__:"):
                # WeChat card image file
                card_file = fname.split(":", 1)[1]
                file_downloads.append({
                    "filename": card_file,
                    "type": "wechat_card",
                    "download_url": f"{base_url}/static/wechat/{card_file}",
                })
            else:
                file_downloads.append({
                    "filename": fname,
                    "type": "material",
                    "download_url": f"{base_url}/api/materials/{fname}",
                })

        messages.append({
            "follow_up_id": f["id"],
            "target_user_id": f["wechat_user_id"],
            "group_name": f.get("group_chat_name", ""),
            "content": f["generated_message"],
            "customer_name": f["customer_name"],
            "attachment_files": file_downloads,
        })
    return {"messages": messages}


@app.post("/api/webhook/message-sent")
async def webhook_message_sent(req: WebhookMessageSentRequest):
    """Callback from robot confirming a follow-up message was sent."""
    from database import update_follow_up

    if req.success:
        update_follow_up(
            req.follow_up_id,
            status="sent",
            sent_at=datetime.now().isoformat(),
        )
    else:
        update_follow_up(
            req.follow_up_id,
            status="failed",
            error_message=req.error,
        )
    return {"status": "ok"}


# ── Admin API ──────────────────────────────────────────────────

# Simple token-based auth
_admin_tokens: set[str] = set()


def _check_admin(authorization: str | None) -> bool:
    if not authorization:
        return False
    token = authorization.replace("Bearer ", "")
    return token in _admin_tokens


@app.post("/api/admin/login")
async def admin_login(req: AdminLoginRequest):
    """Admin login with simple password."""
    if req.password != Config.ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="密码错误")
    token = secrets.token_hex(16)
    _admin_tokens.add(token)
    return {"status": "ok", "token": token}


@app.get("/api/admin/dashboard")
async def admin_dashboard(authorization: str | None = Header(None)):
    """Get dashboard statistics."""
    if not _check_admin(authorization):
        raise HTTPException(status_code=401, detail="未授权")
    from database import get_dashboard_stats
    return get_dashboard_stats()


@app.get("/api/admin/leads")
async def admin_list_leads(
    intent_level: str | None = None,
    limit: int = 50,
    offset: int = 0,
    authorization: str | None = Header(None),
):
    """List leads with intent analysis results."""
    if not _check_admin(authorization):
        raise HTTPException(status_code=401, detail="未授权")
    from database import get_high_intent_leads
    leads = get_high_intent_leads(intent_level=intent_level, limit=limit, offset=offset)
    return {"leads": leads}


@app.get("/api/admin/leads/{customer_id}")
async def admin_lead_detail(customer_id: int, authorization: str | None = Header(None)):
    """Get detailed info for a specific lead."""
    if not _check_admin(authorization):
        raise HTTPException(status_code=401, detail="未授权")
    from database import get_customer, get_messages_by_customer, get_follow_ups
    import sqlite3

    customer = get_customer(customer_id)
    if not customer:
        raise HTTPException(status_code=404, detail="客户不存在")

    messages = get_messages_by_customer(customer_id, limit=100)

    # Get intent analyses for this customer
    from database import get_connection
    conn = get_connection()
    try:
        analyses = [dict(r) for r in conn.execute(
            "SELECT * FROM intent_analyses WHERE customer_id=? ORDER BY analyzed_at DESC",
            (customer_id,)
        ).fetchall()]
    finally:
        conn.close()

    follow_ups = get_follow_ups()
    customer_followups = [f for f in follow_ups if f["customer_id"] == customer_id]

    return {
        "customer": customer,
        "messages": messages,
        "analyses": analyses,
        "follow_ups": customer_followups,
    }


@app.get("/api/admin/messages")
async def admin_list_messages(
    customer_id: int | None = None,
    group_chat_id: str | None = None,
    limit: int = 100,
    offset: int = 0,
    authorization: str | None = Header(None),
):
    """List all messages with optional filters."""
    if not _check_admin(authorization):
        raise HTTPException(status_code=401, detail="未授权")
    from database import list_messages
    msgs = list_messages(customer_id=customer_id, group_chat_id=group_chat_id,
                         limit=limit, offset=offset)
    return {"messages": msgs}


@app.delete("/api/admin/messages/{message_id}")
async def admin_delete_message(
    message_id: int,
    authorization: str | None = Header(None),
):
    """Delete a single message and its related analyses/follow-ups."""
    if not _check_admin(authorization):
        raise HTTPException(status_code=401, detail="未授权")
    from database import delete_message
    deleted = delete_message(message_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="消息不存在")
    return {"status": "ok", "message": f"消息 {message_id} 已删除"}


@app.delete("/api/admin/messages")
async def admin_delete_all_messages(
    authorization: str | None = Header(None),
):
    """Delete all messages and related analyses/follow-ups."""
    if not _check_admin(authorization):
        raise HTTPException(status_code=401, detail="未授权")
    from database import delete_all_messages
    count = delete_all_messages()
    return {"status": "ok", "message": f"已清空 {count} 条消息记录", "deleted_count": count}


@app.post("/api/admin/follow-up/{customer_id}/confirm")
async def admin_confirm_follow_up(
    customer_id: int,
    req: FollowUpConfirmRequest,
    authorization: str | None = Header(None),
):
    """Confirm follow-up: create record and generate AI message."""
    if not _check_admin(authorization):
        raise HTTPException(status_code=401, detail="未授权")

    from database import (
        get_customer, get_messages_by_customer, create_follow_up,
        update_follow_up, get_connection
    )
    from followup_generator import generate_followup_message

    customer = get_customer(customer_id)
    if not customer:
        raise HTTPException(status_code=404, detail="客户不存在")

    # Get latest intent analysis
    conn = get_connection()
    try:
        latest_analysis = conn.execute(
            "SELECT * FROM intent_analyses WHERE customer_id=? ORDER BY analyzed_at DESC LIMIT 1",
            (customer_id,)
        ).fetchone()
    finally:
        conn.close()

    intent_summary = dict(latest_analysis)["intent_summary"] if latest_analysis else ""
    analysis_id = dict(latest_analysis)["id"] if latest_analysis else None

    # Get recent messages for context
    messages = get_messages_by_customer(customer_id, limit=10)
    latest_content = messages[0]["content"] if messages else ""

    # Create follow-up record
    follow_up_id = create_follow_up(
        customer_id=customer_id,
        intent_analysis_id=analysis_id,
        target_user_id=customer["wechat_user_id"],
        target_group_id=customer.get("group_chat_id", ""),
    )

    update_follow_up(
        follow_up_id,
        status="confirmed",
        admin_note=req.admin_note,
        confirmed_at=datetime.now().isoformat(),
    )

    # Generate follow-up message via DeepSeek
    result = await asyncio.to_thread(
        generate_followup_message,
        customer_name=customer["name"],
        message_content=latest_content,
        intent_summary=intent_summary,
        recent_messages=messages,
        admin_note=req.admin_note,
    )

    generated = result.get("message", "") if result else ""
    attachment_files = result.get("recommended_files", []) if result else []

    if generated:
        update_follow_up(
            follow_up_id,
            status="message_generated",
            generated_message=generated,
            attachment_files=json.dumps(attachment_files, ensure_ascii=False),
        )
    else:
        update_follow_up(follow_up_id, status="failed", error_message="消息生成失败")

    return {
        "status": "ok",
        "follow_up_id": follow_up_id,
        "generated_message": generated,
        "attachment_files": attachment_files,
    }


@app.post("/api/admin/follow-up/send")
async def admin_send_follow_up(
    req: FollowUpSendRequest,
    authorization: str | None = Header(None),
):
    """Approve and queue the follow-up message for sending."""
    if not _check_admin(authorization):
        raise HTTPException(status_code=401, detail="未授权")

    from database import get_follow_up, update_follow_up

    follow_up = get_follow_up(req.follow_up_id)
    if not follow_up:
        raise HTTPException(status_code=404, detail="跟进记录不存在")

    # Update message if admin edited it
    message = req.message or follow_up["generated_message"]
    if not message:
        raise HTTPException(status_code=400, detail="消息内容不能为空")

    update_follow_up(
        req.follow_up_id,
        status="message_generated",
        generated_message=message,
    )

    return {"status": "ok", "message": "消息已加入发送队列，等待机器人拉取"}


@app.get("/api/admin/follow-ups")
async def admin_list_follow_ups(
    status: str | None = None,
    reply_type: str | None = None,
    limit: int = 50,
    offset: int = 0,
    authorization: str | None = Header(None),
):
    """List follow-up records."""
    if not _check_admin(authorization):
        raise HTTPException(status_code=401, detail="未授权")
    from database import get_follow_ups
    follow_ups = get_follow_ups(status=status, reply_type=reply_type, limit=limit, offset=offset)
    return {"follow_ups": follow_ups}


@app.post("/api/admin/follow-up/{follow_up_id}/approve")
async def admin_approve_follow_up(
    follow_up_id: int,
    req: FollowUpSendRequest,
    authorization: str | None = Header(None),
):
    """Admin approves a pending follow-up message for sending."""
    if not _check_admin(authorization):
        raise HTTPException(status_code=401, detail="未授权")

    from database import get_follow_up, update_follow_up

    follow_up = get_follow_up(follow_up_id)
    if not follow_up:
        raise HTTPException(status_code=404, detail="跟进记录不存在")
    if follow_up["status"] != "pending_approval":
        raise HTTPException(status_code=400, detail="该记录不在待审核状态")

    message = req.message or follow_up["generated_message"]
    if not message:
        raise HTTPException(status_code=400, detail="消息内容不能为空")

    update_follow_up(
        follow_up_id,
        status="message_generated",
        generated_message=message,
        confirmed_at=datetime.now().isoformat(),
    )

    return {"status": "ok", "message": "跟单消息已审核通过，等待机器人发送"}


@app.post("/api/admin/follow-up/{follow_up_id}/reject")
async def admin_reject_follow_up(
    follow_up_id: int,
    authorization: str | None = Header(None),
):
    """Admin rejects a pending follow-up message."""
    if not _check_admin(authorization):
        raise HTTPException(status_code=401, detail="未授权")

    from database import get_follow_up, update_follow_up

    follow_up = get_follow_up(follow_up_id)
    if not follow_up:
        raise HTTPException(status_code=404, detail="跟进记录不存在")

    update_follow_up(follow_up_id, status="rejected")
    return {"status": "ok", "message": "已驳回"}


# ── Team Members Admin API ────────────────────────────────────


@app.get("/api/admin/team-members")
async def admin_list_team_members(authorization: str | None = Header(None)):
    """List all team member wechat IDs."""
    if not _check_admin(authorization):
        raise HTTPException(status_code=401, detail="未授权")
    from database import list_team_members
    return {"team_members": list_team_members()}


@app.post("/api/admin/team-members")
async def admin_add_team_member(
    req: TeamMemberRequest,
    authorization: str | None = Header(None),
):
    """Add a team member wechat ID."""
    if not _check_admin(authorization):
        raise HTTPException(status_code=401, detail="未授权")
    if not req.wechat_id.strip():
        raise HTTPException(status_code=400, detail="微信ID不能为空")
    from database import add_team_member
    member_id = add_team_member(req.wechat_id.strip(), req.name.strip())
    if member_id is None:
        raise HTTPException(status_code=409, detail="该微信ID已存在")
    return {"status": "ok", "id": member_id}


@app.delete("/api/admin/team-members/{member_id}")
async def admin_remove_team_member(
    member_id: int,
    authorization: str | None = Header(None),
):
    """Remove a team member."""
    if not _check_admin(authorization):
        raise HTTPException(status_code=401, detail="未授权")
    from database import remove_team_member
    remove_team_member(member_id)
    return {"status": "ok"}


# ── Knowledge Base Admin API ──────────────────────────────────


def _process_kb_document(doc_id: int):
    """Background task (sync): parse, chunk, label a document and store results in SQLite."""
    from database import update_kb_document, insert_kb_chunks, get_kb_document
    from parsers.router import parse_file
    from chunkers.text_chunker import chunk_document
    from labelers.llm_labeler import label_chunks

    try:
        doc = get_kb_document(doc_id)
        if not doc:
            return

        update_kb_document(doc_id, status="processing")

        # Parse
        parsed = parse_file(doc["storage_path"])

        # Chunk
        chunks = chunk_document(parsed)
        if not chunks:
            update_kb_document(doc_id, status="failed",
                               error_message="文档解析后无有效内容")
            return

        # Label via LLM
        chunks = label_chunks(chunks)

        # Store chunks into SQLite staging table
        chunks_data = []
        for c in chunks:
            chunks_data.append({
                "chunk_index": c.chunk_index,
                "text": c.text,
                "page_number": c.metadata.get("page_number", 0),
                "category": c.category or "其他",
                "keywords": ", ".join(c.keywords) if c.keywords else "",
                "summary": c.summary or "",
                "importance": "中",
                "related_products": "",
            })
        insert_kb_chunks(doc_id, chunks_data)

        update_kb_document(
            doc_id,
            status="ready",
            total_pages=len(parsed.pages),
            total_chunks=len(chunks),
            processed_at=datetime.now().isoformat(),
        )
        logger.info(f"KB document {doc_id} processed: {len(chunks)} chunks")

    except Exception as e:
        logger.error(f"KB document {doc_id} processing failed: {e}")
        update_kb_document(doc_id, status="failed", error_message=str(e)[:500])


@app.post("/api/admin/kb/upload")
async def admin_kb_upload(
    file: UploadFile = File(...),
    authorization: str | None = Header(None),
):
    """Upload a document for knowledge base processing."""
    if not _check_admin(authorization):
        raise HTTPException(status_code=401, detail="未授权")

    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    # Save file to staging directory
    _ensure_dirs()
    safe_name = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}"
    save_path = os.path.join(_KB_STAGING_DIR, safe_name)
    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    ext = os.path.splitext(file.filename)[1].lower()
    from database import insert_kb_document
    doc_id = insert_kb_document(
        file_name=file.filename,
        file_type=ext,
        file_size=len(content),
        storage_path=save_path,
    )

    # Process in background thread
    asyncio.create_task(asyncio.to_thread(_process_kb_document, doc_id))

    return {"status": "ok", "document_id": doc_id, "file_name": file.filename}


@app.get("/api/admin/kb/documents")
async def admin_kb_list_documents(
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
    authorization: str | None = Header(None),
):
    """List knowledge base documents."""
    if not _check_admin(authorization):
        raise HTTPException(status_code=401, detail="未授权")
    from database import list_kb_documents, get_kb_document_stats
    docs = list_kb_documents(status=status, limit=limit, offset=offset)
    stats = get_kb_document_stats()
    return {"documents": docs, "stats": stats}


@app.get("/api/admin/kb/documents/{doc_id}")
async def admin_kb_document_detail(
    doc_id: int,
    authorization: str | None = Header(None),
):
    """Get document detail with all chunks."""
    if not _check_admin(authorization):
        raise HTTPException(status_code=401, detail="未授权")
    from database import get_kb_document, list_kb_chunks
    doc = get_kb_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="文档不存在")
    chunks = list_kb_chunks(doc_id)
    return {"document": doc, "chunks": chunks}


@app.put("/api/admin/kb/chunks/{chunk_id}")
async def admin_kb_edit_chunk(
    chunk_id: int,
    req: KBChunkEditRequest,
    authorization: str | None = Header(None),
):
    """Edit a single chunk's labels."""
    if not _check_admin(authorization):
        raise HTTPException(status_code=401, detail="未授权")
    from database import get_kb_chunk, update_kb_chunk
    chunk = get_kb_chunk(chunk_id)
    if not chunk:
        raise HTTPException(status_code=404, detail="切片不存在")
    update_kb_chunk(
        chunk_id,
        category=req.category,
        keywords=req.keywords,
        summary=req.summary,
        importance=req.importance,
        related_products=req.related_products,
        manually_edited=1,
    )
    return {"status": "ok"}


@app.post("/api/admin/kb/documents/{doc_id}/publish")
async def admin_kb_publish(
    doc_id: int,
    authorization: str | None = Header(None),
):
    """Publish document chunks to ChromaDB knowledge base."""
    if not _check_admin(authorization):
        raise HTTPException(status_code=401, detail="未授权")
    from database import get_kb_document, list_kb_chunks, update_kb_document
    from chunkers.text_chunker import Chunk

    doc = get_kb_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="文档不存在")
    if doc["status"] not in ("ready", "published"):
        raise HTTPException(status_code=400, detail=f"文档状态不正确: {doc['status']}")

    chunks_data = list_kb_chunks(doc_id)
    if not chunks_data:
        raise HTTPException(status_code=400, detail="文档没有切片数据")

    # Rebuild Chunk objects for ChromaStore
    chunks = []
    for c in chunks_data:
        chunk = Chunk(
            text=c["text"],
            chunk_index=c["chunk_index"],
            metadata={
                "source_file": doc["file_name"],
                "source_path": doc.get("storage_path", ""),
                "file_type": doc["file_type"],
                "page_number": c["page_number"],
                "importance": c["importance"],
                "related_products": c["related_products"],
            },
            category=c["category"],
            keywords=[k.strip() for k in c["keywords"].split(",") if k.strip()] if c["keywords"] else [],
            summary=c["summary"],
        )
        chunks.append(chunk)

    # If previously published, delete old data from ChromaDB first
    if doc["status"] == "published":
        try:
            _rag_engine._store.delete_by_source(doc["file_name"])
        except Exception as e:
            logger.warning(f"Failed to delete old chunks for {doc['file_name']}: {e}")

    # Add to ChromaDB
    added = await asyncio.to_thread(_rag_engine._store.add_chunks, chunks)

    update_kb_document(
        doc_id,
        status="published",
        published_at=datetime.now().isoformat(),
    )

    return {"status": "ok", "chunks_published": added}


@app.delete("/api/admin/kb/documents/{doc_id}")
async def admin_kb_delete_document(
    doc_id: int,
    authorization: str | None = Header(None),
):
    """Delete a document and its chunks. Also remove from ChromaDB if published."""
    if not _check_admin(authorization):
        raise HTTPException(status_code=401, detail="未授权")
    from database import get_kb_document, delete_kb_document

    doc = get_kb_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="文档不存在")

    # Remove from ChromaDB if published
    if doc["status"] == "published":
        try:
            _rag_engine._store.delete_by_source(doc["file_name"])
        except Exception as e:
            logger.warning(f"Failed to delete from ChromaDB: {e}")

    # Remove staging file
    if doc["storage_path"] and os.path.exists(doc["storage_path"]):
        try:
            os.remove(doc["storage_path"])
        except Exception:
            pass

    # Delete from SQLite (CASCADE deletes chunks)
    delete_kb_document(doc_id)
    return {"status": "ok"}


# ── Serve static files ───────────────────────────────────────────

# Mount static directory for WeChat card images
if not os.path.exists(os.path.join(_BASE_DIR, "static", "wechat")):
    os.makedirs(os.path.join(_BASE_DIR, "static", "wechat"), exist_ok=True)
app.mount("/static", StaticFiles(directory=os.path.join(_BASE_DIR, "static")), name="static")


# ── Web UI ───────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return _CHAT_HTML


@app.get("/admin", response_class=HTMLResponse)
async def admin_page():
    return _ADMIN_HTML


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

  /* Material card - file send style */
  .material-card { background: linear-gradient(135deg, #eff6ff, #e0ecff); border: 1px solid #c4d5f0; border-radius: 12px; padding: 14px 16px; margin: 6px 0; display: flex; align-items: center; gap: 14px; animation: fadeIn .3s ease; cursor: pointer; transition: all .2s; }
  .material-card:hover { box-shadow: 0 2px 8px rgba(44,95,138,.15); border-color: #2c5f8a; }
  .material-card .file-icon { width: 42px; height: 42px; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: 700; color: #fff; flex-shrink: 0; text-transform: uppercase; }
  .material-card .file-icon.pdf { background: #e53935; }
  .material-card .file-icon.doc { background: #1565c0; }
  .material-card .file-icon.xls { background: #2e7d32; }
  .material-card .file-icon.ppt { background: #e65100; }
  .material-card .file-icon.txt { background: #757575; }
  .material-card .file-icon.other { background: #546e7a; }
  .material-card .file-info { flex: 1; min-width: 0; }
  .material-card .file-name { font-size: 13px; color: #333; font-weight: 500; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .material-card .file-meta { font-size: 11px; color: #999; margin-top: 3px; }
  .material-card .dl-btn { background: #2c5f8a; color: #fff; border: none; border-radius: 6px; padding: 6px 14px; font-size: 12px; cursor: pointer; transition: all .2s; flex-shrink: 0; }
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
let materialsMap = {}; // filename -> {size, type}

// Load initial data
fetch('/api/stats').then(r=>r.json()).then(d=>{
  document.getElementById('stats').textContent = `知识库 ${d.total_documents} 条`;
}).catch(()=>{});

fetch('/api/wechat-card').then(r=>r.json()).then(d=>{
  wechatConfig = d;
}).catch(()=>{});

fetch('/api/materials').then(r=>r.json()).then(d=>{
  (d.materials || []).forEach(m => {
    materialsMap[m.filename] = { size: m.size, type: m.type };
  });
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

    // Add material cards (file send style)
    for (const mat of materials) {
      const info = materialsMap[mat] || {};
      const ext = mat.split('.').pop().toLowerCase();
      const iconClass = ['pdf'].includes(ext) ? 'pdf'
        : ['doc','docx'].includes(ext) ? 'doc'
        : ['xls','xlsx','csv'].includes(ext) ? 'xls'
        : ['ppt','pptx'].includes(ext) ? 'ppt'
        : ['txt'].includes(ext) ? 'txt' : 'other';
      const sizeTxt = info.size ? info.size : '';
      const typeTxt = info.type ? info.type : ext.toUpperCase();
      html += `<div class="material-card" onclick="downloadMaterial('${escAttr(mat)}')">
        <div class="file-icon ${iconClass}">${typeTxt}</div>
        <div class="file-info">
          <div class="file-name">${escHtml(mat)}</div>
          <div class="file-meta">${sizeTxt ? typeTxt + ' 文件 · ' + sizeTxt : typeTxt + ' 文件'}</div>
        </div>
        <button class="dl-btn">接收文件</button>
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
  const a = document.createElement('a');
  a.href = '/api/materials/' + encodeURIComponent(filename);
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
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


_ADMIN_HTML = """\
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>理博基金 - 线索管理后台</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", "Microsoft YaHei", sans-serif; background: #f5f6fa; min-height: 100vh; }

  .header { background: linear-gradient(135deg, #1a3a5c, #2c5f8a); color: #fff; padding: 14px 24px; display: flex; align-items: center; gap: 14px; box-shadow: 0 2px 12px rgba(0,0,0,.12); }
  .header .logo { font-size: 20px; font-weight: 700; letter-spacing: 2px; }
  .header .logo span { font-size: 12px; font-weight: 400; opacity: .7; letter-spacing: 1px; margin-left: 8px; }
  .header .actions { margin-left: auto; display: flex; gap: 12px; align-items: center; }
  .header a { color: #fff; text-decoration: none; opacity: .8; font-size: 13px; }
  .header a:hover { opacity: 1; }

  /* Login */
  .login-overlay { position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,.4); display: flex; align-items: center; justify-content: center; z-index: 2000; }
  .login-box { background: #fff; border-radius: 16px; padding: 32px; width: 340px; box-shadow: 0 20px 60px rgba(0,0,0,.2); text-align: center; }
  .login-box h3 { font-size: 18px; color: #1a3a5c; margin-bottom: 20px; }
  .login-box input { width: 100%; border: 1.5px solid #dde3ea; border-radius: 10px; padding: 10px 14px; font-size: 14px; outline: none; margin-bottom: 14px; }
  .login-box input:focus { border-color: #2c5f8a; }
  .login-box button { width: 100%; background: #2c5f8a; color: #fff; border: none; border-radius: 10px; padding: 10px; font-size: 14px; cursor: pointer; }
  .login-box button:hover { background: #1a3a5c; }
  .login-box .error { color: #e53935; font-size: 12px; margin-top: 8px; }

  /* Tabs */
  .tabs { display: flex; background: #fff; border-bottom: 1px solid #e4e8ee; padding: 0 24px; }
  .tab { padding: 12px 20px; font-size: 14px; color: #666; cursor: pointer; border-bottom: 2px solid transparent; transition: all .2s; }
  .tab:hover { color: #2c5f8a; }
  .tab.active { color: #2c5f8a; border-bottom-color: #2c5f8a; font-weight: 600; }

  .content { max-width: 1200px; margin: 0 auto; padding: 20px; }

  .panel { display: none; }
  .panel.active { display: block; }

  /* Dashboard cards */
  .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 24px; }
  .stat-card { background: #fff; border-radius: 12px; padding: 20px; box-shadow: 0 1px 6px rgba(0,0,0,.06); }
  .stat-card .label { font-size: 13px; color: #888; margin-bottom: 6px; }
  .stat-card .value { font-size: 28px; font-weight: 700; color: #1a3a5c; }

  /* Tables */
  .card { background: #fff; border-radius: 12px; box-shadow: 0 1px 6px rgba(0,0,0,.06); overflow: hidden; margin-bottom: 20px; }
  .card-header { padding: 16px 20px; border-bottom: 1px solid #f0f0f0; display: flex; align-items: center; gap: 12px; }
  .card-header h3 { font-size: 16px; color: #333; }
  .filter-btns { display: flex; gap: 6px; margin-left: auto; }
  .filter-btn { padding: 4px 12px; border-radius: 14px; border: 1px solid #dde3ea; background: #fff; font-size: 12px; cursor: pointer; color: #666; transition: all .2s; }
  .filter-btn:hover, .filter-btn.active { background: #2c5f8a; color: #fff; border-color: #2c5f8a; }

  table { width: 100%; border-collapse: collapse; }
  th { background: #f8f9fb; text-align: left; padding: 10px 16px; font-size: 12px; color: #888; font-weight: 600; }
  td { padding: 12px 16px; font-size: 13px; color: #333; border-top: 1px solid #f0f0f0; }
  tr:hover td { background: #fafbfc; }

  .badge { display: inline-block; padding: 2px 10px; border-radius: 10px; font-size: 11px; font-weight: 600; }
  .badge.high { background: #fce4ec; color: #c62828; }
  .badge.medium { background: #fff3e0; color: #e65100; }
  .badge.low { background: #e8f5e9; color: #2e7d32; }
  .badge.none { background: #f5f5f5; color: #999; }
  .badge.pending { background: #fff3e0; color: #e65100; }
  .badge.confirmed { background: #e3f2fd; color: #1565c0; }
  .badge.message_generated { background: #e8f5e9; color: #2e7d32; }
  .badge.sent { background: #e8f5e9; color: #1b5e20; }
  .badge.failed { background: #fce4ec; color: #c62828; }

  .btn { padding: 5px 14px; border-radius: 6px; font-size: 12px; cursor: pointer; transition: all .2s; border: none; }
  .btn-primary { background: #2c5f8a; color: #fff; }
  .btn-primary:hover { background: #1a3a5c; }
  .btn-success { background: #2e7d32; color: #fff; }
  .btn-success:hover { background: #1b5e20; }
  .btn-danger { background: #e74c3c; color: #fff; }
  .btn-danger:hover { background: #c0392b; }
  .btn-sm { padding: 3px 10px; font-size: 11px; }

  .text-truncate { max-width: 300px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .text-muted { color: #999; font-size: 12px; }

  .empty { text-align: center; padding: 40px; color: #999; font-size: 14px; }

  /* Modal */
  .modal-overlay { display: none; position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,.4); z-index: 1000; }
  .modal-overlay.show { display: flex; align-items: center; justify-content: center; }
  .modal { background: #fff; border-radius: 16px; padding: 28px; max-width: 600px; width: 90%; max-height: 80vh; overflow-y: auto; box-shadow: 0 20px 60px rgba(0,0,0,.2); }
  .modal h3 { font-size: 18px; color: #1a3a5c; margin-bottom: 16px; }
  .modal textarea { width: 100%; border: 1.5px solid #dde3ea; border-radius: 10px; padding: 12px; font-size: 13px; resize: vertical; min-height: 100px; outline: none; font-family: inherit; margin: 10px 0; }
  .modal textarea:focus { border-color: #2c5f8a; }
  .modal-actions { display: flex; gap: 10px; justify-content: flex-end; margin-top: 16px; }
  .modal label { font-size: 13px; color: #666; display: block; margin-bottom: 4px; }

  /* Detail panel */
  .detail-section { margin-bottom: 20px; }
  .detail-section h4 { font-size: 14px; color: #555; margin-bottom: 10px; font-weight: 600; }
  .msg-list { max-height: 300px; overflow-y: auto; }
  .msg-item { padding: 8px 12px; border-left: 3px solid #dde3ea; margin-bottom: 6px; background: #fafbfc; border-radius: 0 8px 8px 0; }
  .msg-item .msg-meta { font-size: 11px; color: #999; margin-bottom: 2px; }
  .msg-item .msg-text { font-size: 13px; color: #333; line-height: 1.6; }

  /* KB Upload */
  .upload-zone { border: 2px dashed #c5cdd8; border-radius: 12px; padding: 40px; text-align: center; cursor: pointer; transition: all .2s; background: #fafbfc; margin-bottom: 20px; }
  .upload-zone:hover, .upload-zone.drag-over { border-color: #2c5f8a; background: #eef3f8; }
  .upload-zone .icon { font-size: 36px; margin-bottom: 10px; }
  .upload-zone .text { font-size: 14px; color: #666; }
  .upload-zone .hint { font-size: 12px; color: #999; margin-top: 6px; }
  .upload-zone input[type=file] { display: none; }

  .kb-status { display: inline-block; padding: 2px 10px; border-radius: 10px; font-size: 11px; font-weight: 600; }
  .kb-status.uploaded { background: #e3f2fd; color: #1565c0; }
  .kb-status.processing { background: #fff3e0; color: #e65100; }
  .kb-status.ready { background: #e8f5e9; color: #2e7d32; }
  .kb-status.published { background: #ede7f6; color: #4527a0; }
  .kb-status.failed { background: #fce4ec; color: #c62828; }

  /* Chunk review modal (large) */
  .modal-lg { max-width: 900px; width: 95%; }
  .chunk-table { width: 100%; border-collapse: collapse; font-size: 12px; }
  .chunk-table th { background: #f8f9fb; padding: 8px 10px; text-align: left; font-size: 11px; color: #888; font-weight: 600; position: sticky; top: 0; }
  .chunk-table td { padding: 8px 10px; border-top: 1px solid #f0f0f0; vertical-align: top; }
  .chunk-text-preview { max-width: 280px; max-height: 60px; overflow: hidden; text-overflow: ellipsis; font-size: 12px; color: #555; line-height: 1.5; }

  /* Chunk edit form */
  .edit-form label { font-size: 13px; color: #555; display: block; margin: 12px 0 4px; font-weight: 500; }
  .edit-form select, .edit-form input[type=text] { width: 100%; border: 1.5px solid #dde3ea; border-radius: 8px; padding: 8px 12px; font-size: 13px; outline: none; font-family: inherit; }
  .edit-form select:focus, .edit-form input[type=text]:focus { border-color: #2c5f8a; }
  .edit-form .checkbox-group { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 4px; }
  .edit-form .checkbox-group label { display: flex; align-items: center; gap: 4px; font-weight: 400; margin: 0; cursor: pointer; }
  .edit-form .chunk-preview { background: #f5f6fa; border-radius: 8px; padding: 12px; max-height: 120px; overflow-y: auto; font-size: 12px; color: #555; line-height: 1.6; margin-bottom: 8px; }
  .kb-stats-row { display: flex; gap: 16px; margin-bottom: 16px; }
  .kb-stats-row .stat-card { flex: 1; }
</style>
</head>
<body>

<div class="header">
  <div class="logo">理博基金<span>线索管理后台</span></div>
  <div class="actions">
    <a href="/">返回客服</a>
  </div>
</div>

<!-- Login overlay -->
<div class="login-overlay" id="loginOverlay">
  <div class="login-box">
    <h3>管理员登录</h3>
    <input type="password" id="loginPwd" placeholder="请输入管理密码" onkeydown="if(event.key==='Enter')doLogin()">
    <button onclick="doLogin()">登录</button>
    <div class="error" id="loginError"></div>
  </div>
</div>

<!-- Tabs -->
<div class="tabs">
  <div class="tab active" onclick="switchTab('dashboard')">仪表盘</div>
  <div class="tab" onclick="switchTab('leads')">客户线索</div>
  <div class="tab" onclick="switchTab('followups')">跟进管理</div>
  <div class="tab" onclick="switchTab('messages')">消息记录</div>
  <div class="tab" onclick="switchTab('kb')">知识库管理</div>
  <div class="tab" onclick="switchTab('team')">团队管理</div>
</div>

<div class="content">

  <!-- Dashboard -->
  <div class="panel active" id="panel-dashboard">
    <div class="stats-grid" id="statsGrid"></div>
  </div>

  <!-- Leads -->
  <div class="panel" id="panel-leads">
    <div class="card">
      <div class="card-header">
        <h3>客户线索</h3>
        <div class="filter-btns">
          <button class="filter-btn active" onclick="filterLeads(null, this)">全部</button>
          <button class="filter-btn" onclick="filterLeads('high', this)">高意向</button>
          <button class="filter-btn" onclick="filterLeads('medium', this)">中意向</button>
        </div>
      </div>
      <table>
        <thead><tr><th>客户</th><th>来源群</th><th>最新消息</th><th>意向</th><th>分数</th><th>时间</th><th>操作</th></tr></thead>
        <tbody id="leadsBody"></tbody>
      </table>
      <div class="empty" id="leadsEmpty" style="display:none;">暂无线索数据</div>
    </div>
  </div>

  <!-- Follow-ups -->
  <div class="panel" id="panel-followups">
    <div class="card">
      <div class="card-header">
        <h3>跟进管理</h3>
        <div class="filter-btns">
          <button class="filter-btn active" onclick="filterFollowUps(null, this)">全部</button>
          <button class="filter-btn" onclick="filterFollowUps('pending_approval', this)">待审核</button>
          <button class="filter-btn" onclick="filterFollowUps('message_generated', this)">待发送</button>
          <button class="filter-btn" onclick="filterFollowUps('sent', this)">已发送</button>
          <button class="filter-btn" onclick="filterFollowUps('failed', this)">失败</button>
        </div>
      </div>
      <table>
        <thead><tr><th>客户</th><th>类型</th><th>状态</th><th>回复消息</th><th>更新时间</th><th>操作</th></tr></thead>
        <tbody id="followupsBody"></tbody>
      </table>
      <div class="empty" id="followupsEmpty" style="display:none;">暂无跟进记录</div>
    </div>
  </div>

  <!-- Messages -->
  <div class="panel" id="panel-messages">
    <div class="card">
      <div class="card-header">
        <h3>消息记录</h3>
        <button class="btn btn-danger" onclick="clearAllMessages()" style="margin-left:auto;">清空全部</button>
      </div>
      <table>
        <thead><tr><th>客户</th><th>群</th><th>内容</th><th>时间</th><th>操作</th></tr></thead>
        <tbody id="messagesBody"></tbody>
      </table>
      <div class="empty" id="messagesEmpty" style="display:none;">暂无消息记录</div>
    </div>
  </div>

  <!-- Knowledge Base -->
  <div class="panel" id="panel-kb">
    <div class="kb-stats-row" id="kbStatsRow"></div>

    <div class="upload-zone" id="uploadZone" onclick="document.getElementById('kbFileInput').click()">
      <div class="icon">&#128196;</div>
      <div class="text">点击或拖拽文件到此区域上传</div>
      <div class="hint">支持 PDF / DOCX / PPTX / Excel / 图片 / TXT</div>
      <input type="file" id="kbFileInput" accept=".pdf,.docx,.pptx,.xlsx,.xls,.png,.jpg,.jpeg,.gif,.bmp,.txt,.md,.csv" onchange="handleKBUpload(this)">
    </div>

    <div class="card">
      <div class="card-header">
        <h3>文档列表</h3>
        <div class="filter-btns">
          <button class="filter-btn active" onclick="filterKBDocs(null, this)">全部</button>
          <button class="filter-btn" onclick="filterKBDocs('processing', this)">处理中</button>
          <button class="filter-btn" onclick="filterKBDocs('ready', this)">待发布</button>
          <button class="filter-btn" onclick="filterKBDocs('published', this)">已发布</button>
        </div>
      </div>
      <table>
        <thead><tr><th>文件名</th><th>类型</th><th>大小</th><th>切片数</th><th>状态</th><th>上传时间</th><th>操作</th></tr></thead>
        <tbody id="kbDocsBody"></tbody>
      </table>
      <div class="empty" id="kbDocsEmpty" style="display:none;">暂无文档</div>
    </div>
  </div>

  <!-- Team Members -->
  <div class="panel" id="panel-team">
    <div class="card">
      <div class="card-header">
        <h3>团队成员（自己人微信ID）</h3>
        <button class="btn btn-primary btn-sm" onclick="openAddTeamMember()">+ 添加成员</button>
      </div>
      <p style="color:#888;font-size:13px;margin:0 0 12px;">列表中的微信ID发送的消息将被自动忽略，不保存、不回复。</p>
      <table>
        <thead><tr><th>微信ID</th><th>备注名</th><th>添加时间</th><th>操作</th></tr></thead>
        <tbody id="teamBody"></tbody>
      </table>
      <div class="empty" id="teamEmpty" style="display:none;">暂无团队成员</div>
    </div>
  </div>

</div>

<!-- Add team member modal -->
<div class="modal-overlay" id="addTeamModal">
  <div class="modal">
    <h3>添加团队成员</h3>
    <div style="margin-bottom:12px">
      <label style="display:block;margin-bottom:4px;font-weight:600;font-size:13px">微信ID</label>
      <input type="text" id="teamWechatId" placeholder="例如: wxid_abc123" style="width:100%;padding:8px;border:1px solid #ddd;border-radius:6px;">
    </div>
    <div style="margin-bottom:16px">
      <label style="display:block;margin-bottom:4px;font-weight:600;font-size:13px">备注名称</label>
      <input type="text" id="teamMemberName" placeholder="例如: 张销售" style="width:100%;padding:8px;border:1px solid #ddd;border-radius:6px;">
    </div>
    <div style="display:flex;gap:8px;justify-content:flex-end">
      <button class="btn" onclick="closeModal('addTeamModal')">取消</button>
      <button class="btn btn-primary" id="addTeamBtn" onclick="doAddTeamMember()">添加</button>
    </div>
  </div>
</div>

<!-- Chunk review modal -->
<div class="modal-overlay" id="chunkReviewModal">
  <div class="modal modal-lg" style="max-height:90vh;">
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:16px;">
      <h3 style="margin:0;" id="chunkReviewTitle">切片审核</h3>
      <button class="btn btn-primary" id="publishFromReviewBtn" onclick="doPublishDoc()" style="margin-left:auto;">发布到知识库</button>
      <button class="btn" style="background:#f5f5f5;color:#666;border:1px solid #ddd;" onclick="closeModal('chunkReviewModal')">关闭</button>
    </div>
    <div style="overflow-y:auto;max-height:calc(90vh - 100px);">
      <table class="chunk-table">
        <thead><tr><th>#</th><th>内容预览</th><th>分类</th><th>关键词</th><th>摘要</th><th>重要程度</th><th>关联产品</th><th>操作</th></tr></thead>
        <tbody id="chunkReviewBody"></tbody>
      </table>
    </div>
  </div>
</div>

<!-- Chunk edit modal -->
<div class="modal-overlay" id="chunkEditModal">
  <div class="modal" style="max-width:550px;">
    <h3>编辑切片标注</h3>
    <div class="edit-form">
      <div class="chunk-preview" id="editChunkPreview"></div>
      <label>分类</label>
      <select id="editCategory">
        <option value="市场分析">市场分析</option>
        <option value="投资策略">投资策略</option>
        <option value="风险提示">风险提示</option>
        <option value="产品说明">产品说明</option>
        <option value="法规政策">法规政策</option>
        <option value="公司研究">公司研究</option>
        <option value="行业研究">行业研究</option>
        <option value="宏观经济">宏观经济</option>
        <option value="技术分析">技术分析</option>
        <option value="客户服务">客户服务</option>
        <option value="财务数据">财务数据</option>
        <option value="其他">其他</option>
      </select>
      <label>关键词（逗号分隔）</label>
      <input type="text" id="editKeywords" placeholder="关键词1, 关键词2, ...">
      <label>摘要</label>
      <input type="text" id="editSummary" placeholder="一句话概括核心内容">
      <label>重要程度</label>
      <select id="editImportance">
        <option value="高">高</option>
        <option value="中" selected>中</option>
        <option value="低">低</option>
      </select>
      <label>关联产品</label>
      <div class="checkbox-group" id="editProducts">
        <label><input type="checkbox" value="理博1号"> 理博1号</label>
        <label><input type="checkbox" value="远景1号"> 远景1号</label>
        <label><input type="checkbox" value="远景2号"> 远景2号</label>
        <label><input type="checkbox" value="飞天1号"> 飞天1号</label>
        <label><input type="checkbox" value="万象7号"> 万象7号</label>
        <label><input type="checkbox" value="其他"> 其他</label>
      </div>
    </div>
    <div class="modal-actions">
      <button class="btn" style="background:#f5f5f5;color:#666;border:1px solid #ddd;" onclick="closeModal('chunkEditModal')">取消</button>
      <button class="btn btn-primary" onclick="doSaveChunkEdit()">保存</button>
    </div>
  </div>
</div>

<!-- Confirm follow-up modal -->
<div class="modal-overlay" id="confirmModal">
  <div class="modal">
    <h3>确认跟进客户</h3>
    <div id="confirmCustomerInfo"></div>
    <label>管理员备注（可选）</label>
    <textarea id="confirmNote" placeholder="输入备注信息，AI会参考此内容生成跟进消息..."></textarea>
    <div class="modal-actions">
      <button class="btn" style="background:#f5f5f5;color:#666;border:1px solid #ddd;" onclick="closeModal('confirmModal')">取消</button>
      <button class="btn btn-primary" id="confirmBtn" onclick="doConfirmFollowUp()">确认并生成消息</button>
    </div>
  </div>
</div>

<!-- Edit & Send modal -->
<div class="modal-overlay" id="sendModal">
  <div class="modal">
    <h3>编辑并发送跟进消息</h3>
    <div id="sendCustomerInfo"></div>
    <label>跟进消息内容</label>
    <textarea id="sendMessage"></textarea>
    <div class="modal-actions">
      <button class="btn" style="background:#f5f5f5;color:#666;border:1px solid #ddd;" onclick="closeModal('sendModal')">取消</button>
      <button class="btn btn-success" id="sendBtn" onclick="doSendFollowUp()">确认发送</button>
    </div>
  </div>
</div>

<!-- Lead detail modal -->
<div class="modal-overlay" id="detailModal">
  <div class="modal" style="max-width:700px;">
    <h3 id="detailTitle">客户详情</h3>
    <div id="detailContent"></div>
    <div class="modal-actions">
      <button class="btn" style="background:#f5f5f5;color:#666;border:1px solid #ddd;" onclick="closeModal('detailModal')">关闭</button>
    </div>
  </div>
</div>

<script>
let token = localStorage.getItem('admin_token') || '';
let currentLeadFilter = null;
let currentFUFilter = null;
let pendingCustomerId = null;
let pendingFollowUpId = null;

// Check auth
if (token) {
  document.getElementById('loginOverlay').style.display = 'none';
  loadDashboard();
} else {
  document.getElementById('loginOverlay').style.display = 'flex';
}

async function doLogin() {
  const pwd = document.getElementById('loginPwd').value;
  try {
    const res = await apiFetch('/api/admin/login', 'POST', { password: pwd });
    if (res.token) {
      token = res.token;
      localStorage.setItem('admin_token', token);
      document.getElementById('loginOverlay').style.display = 'none';
      loadDashboard();
    }
  } catch(e) {
    document.getElementById('loginError').textContent = '密码错误';
  }
}

async function apiFetch(url, method = 'GET', body = null) {
  const opts = { method, headers: { 'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json' } };
  if (body) opts.body = JSON.stringify(body);
  const res = await fetch(url, opts);
  if (res.status === 401) {
    localStorage.removeItem('admin_token');
    token = '';
    document.getElementById('loginOverlay').style.display = 'flex';
    throw new Error('Unauthorized');
  }
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

function switchTab(name) {
  const tabNames = { dashboard: '仪表盘', leads: '客户线索', followups: '跟进管理', messages: '消息记录', kb: '知识库管理', team: '团队管理' };
  document.querySelectorAll('.tab').forEach(t => {
    t.classList.toggle('active', t.textContent.includes(tabNames[name] || ''));
  });
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.getElementById('panel-' + name).classList.add('active');

  if (name === 'dashboard') loadDashboard();
  else if (name === 'leads') loadLeads();
  else if (name === 'followups') loadFollowUps();
  else if (name === 'messages') loadMessages();
  else if (name === 'kb') loadKBDocuments();
  else if (name === 'team') loadTeamMembers();
}

// Dashboard
async function loadDashboard() {
  try {
    const d = await apiFetch('/api/admin/dashboard');
    document.getElementById('statsGrid').innerHTML = `
      <div class="stat-card"><div class="label">今日消息</div><div class="value">${d.today_messages}</div></div>
      <div class="stat-card"><div class="label">高意向线索</div><div class="value">${d.high_intent_leads}</div></div>
      <div class="stat-card"><div class="label">待跟进</div><div class="value">${d.pending_follow_ups}</div></div>
      <div class="stat-card"><div class="label">已发送</div><div class="value">${d.sent_follow_ups}</div></div>
      <div class="stat-card"><div class="label">客户总数</div><div class="value">${d.total_customers}</div></div>
      <div class="stat-card"><div class="label">消息总数</div><div class="value">${d.total_messages}</div></div>
    `;
  } catch(e) {}
}

// Leads
async function loadLeads(level = null) {
  try {
    let url = '/api/admin/leads?limit=100';
    if (level) url += '&intent_level=' + level;
    const d = await apiFetch(url);
    const body = document.getElementById('leadsBody');
    const empty = document.getElementById('leadsEmpty');
    if (!d.leads || d.leads.length === 0) {
      body.innerHTML = '';
      empty.style.display = 'block';
      return;
    }
    empty.style.display = 'none';
    body.innerHTML = d.leads.map(l => `<tr>
      <td><strong>${esc(l.customer_name)}</strong><br><span class="text-muted">${esc(l.wechat_user_id)}</span></td>
      <td>${esc(l.group_chat_name || '-')}</td>
      <td><div class="text-truncate">${esc(l.message_content || '')}</div></td>
      <td><span class="badge ${l.intent_level}">${levelLabel(l.intent_level)}</span></td>
      <td>${(l.intent_score * 100).toFixed(0)}%</td>
      <td class="text-muted">${formatTime(l.analyzed_at)}</td>
      <td>
        <button class="btn btn-primary btn-sm" onclick="showDetail(${l.customer_id})">详情</button>
        <button class="btn btn-success btn-sm" onclick="openConfirm(${l.customer_id}, '${esc(l.customer_name)}')">跟进</button>
      </td>
    </tr>`).join('');
  } catch(e) {}
}

function filterLeads(level, btn) {
  currentLeadFilter = level;
  document.querySelectorAll('#panel-leads .filter-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  loadLeads(level);
}

// Follow-ups
async function loadFollowUps(status = null) {
  try {
    let url = '/api/admin/follow-ups?limit=100';
    if (status) url += '&status=' + status;
    const d = await apiFetch(url);
    const body = document.getElementById('followupsBody');
    const empty = document.getElementById('followupsEmpty');
    if (!d.follow_ups || d.follow_ups.length === 0) {
      body.innerHTML = '';
      empty.style.display = 'block';
      return;
    }
    empty.style.display = 'none';
    d.follow_ups.forEach(f => { _followUpCache[f.id] = f; });
    body.innerHTML = d.follow_ups.map(f => {
      let filesHtml = '';
      try {
        const files = JSON.parse(f.attachment_files || '[]');
        if (files.length > 0) {
          filesHtml = '<div style="margin-top:4px">' + files.map(fn =>
            `<a href="/api/materials/${encodeURIComponent(fn)}" target="_blank" style="font-size:12px;color:#1a73e8;margin-right:8px">📎${esc(fn)}</a>`
          ).join('') + '</div>';
        }
      } catch(e) {}
      const typeLabel = f.reply_type === 'follow_up' ? '<span style="color:#e67e22;font-weight:600">跟单</span>' : '<span style="color:#27ae60">自动</span>';
      return `<tr>
      <td><strong>${esc(f.customer_name)}</strong></td>
      <td>${typeLabel}</td>
      <td><span class="badge ${f.status}">${statusLabel(f.status)}</span></td>
      <td><div class="text-truncate">${esc(f.generated_message || '-')}${filesHtml}</div></td>
      <td class="text-muted">${formatTime(f.updated_at)}</td>
      <td>${renderFollowUpAction(f)}</td>
    </tr>`;
    }).join('');
  } catch(e) {}
}

function filterFollowUps(status, btn) {
  currentFUFilter = status;
  document.querySelectorAll('#panel-followups .filter-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  loadFollowUps(status);
}

// Messages
async function loadMessages() {
  try {
    const d = await apiFetch('/api/admin/messages?limit=200');
    const body = document.getElementById('messagesBody');
    const empty = document.getElementById('messagesEmpty');
    if (!d.messages || d.messages.length === 0) {
      body.innerHTML = '';
      empty.style.display = 'block';
      return;
    }
    empty.style.display = 'none';
    body.innerHTML = d.messages.map(m => `<tr>
      <td><strong>${esc(m.customer_name || '')}</strong></td>
      <td class="text-muted">${esc(m.group_chat_id || '-')}</td>
      <td><div class="text-truncate">${esc(m.content)}</div></td>
      <td class="text-muted">${formatTime(m.received_at)}</td>
      <td><button class="btn btn-danger btn-sm" onclick="deleteMessage(${m.id})">删除</button></td>
    </tr>`).join('');
  } catch(e) {}
}

async function deleteMessage(msgId) {
  if (!confirm('确定要删除这条消息吗？相关的意向分析和跟进记录也会被删除。')) return;
  try {
    await apiFetch('/api/admin/messages/' + msgId, 'DELETE');
    loadMessages();
  } catch(e) { alert('删除失败: ' + e.message); }
}

async function clearAllMessages() {
  if (!confirm('确定要清空所有消息记录吗？此操作不可恢复，所有消息、意向分析和相关跟进记录都会被删除。')) return;
  try {
    const d = await apiFetch('/api/admin/messages', 'DELETE');
    alert(d.message || '已清空');
    loadMessages();
  } catch(e) { alert('清空失败: ' + e.message); }
}

// Detail
async function showDetail(customerId) {
  try {
    const d = await apiFetch('/api/admin/leads/' + customerId);
    document.getElementById('detailTitle').textContent = '客户详情 - ' + (d.customer.name || d.customer.wechat_user_id);
    let html = '';

    html += '<div class="detail-section"><h4>基本信息</h4>';
    html += `<p>微信ID: ${esc(d.customer.wechat_user_id)} | 群: ${esc(d.customer.group_chat_name || '-')} | 首次出现: ${formatTime(d.customer.first_seen_at)}</p></div>`;

    if (d.analyses && d.analyses.length > 0) {
      html += '<div class="detail-section"><h4>意向分析</h4>';
      d.analyses.forEach(a => {
        html += `<div class="msg-item"><div class="msg-meta"><span class="badge ${a.intent_level}">${levelLabel(a.intent_level)}</span> ${(a.intent_score*100).toFixed(0)}% - ${formatTime(a.analyzed_at)}</div><div class="msg-text">${esc(a.intent_summary)}</div></div>`;
      });
      html += '</div>';
    }

    if (d.messages && d.messages.length > 0) {
      html += '<div class="detail-section"><h4>消息记录</h4><div class="msg-list">';
      d.messages.forEach(m => {
        html += `<div class="msg-item"><div class="msg-meta">${formatTime(m.received_at)}</div><div class="msg-text">${esc(m.content)}</div></div>`;
      });
      html += '</div></div>';
    }

    document.getElementById('detailContent').innerHTML = html;
    document.getElementById('detailModal').classList.add('show');
  } catch(e) {}
}

// Confirm follow-up
function openConfirm(customerId, name) {
  pendingCustomerId = customerId;
  document.getElementById('confirmCustomerInfo').innerHTML = `<p>客户: <strong>${esc(name)}</strong></p>`;
  document.getElementById('confirmNote').value = '';
  document.getElementById('confirmModal').classList.add('show');
}

async function doConfirmFollowUp() {
  const note = document.getElementById('confirmNote').value;
  const btn = document.getElementById('confirmBtn');
  btn.disabled = true;
  btn.textContent = '生成中...';
  try {
    const d = await apiFetch('/api/admin/follow-up/' + pendingCustomerId + '/confirm', 'POST', { admin_note: note });
    closeModal('confirmModal');
    if (d.generated_message) {
      pendingFollowUpId = d.follow_up_id;
      let filesInfo = '';
      if (d.attachment_files && d.attachment_files.length > 0) {
        filesInfo = '<p style="margin-top:8px;color:#666">附带文件：' + d.attachment_files.map(fn =>
          `<a href="/api/materials/${encodeURIComponent(fn)}" target="_blank" style="color:#1a73e8">📎${esc(fn)}</a>`
        ).join('  ') + '</p>';
      }
      document.getElementById('sendCustomerInfo').innerHTML = '<p>AI已生成跟进消息，您可以编辑后发送：</p>' + filesInfo;
      document.getElementById('sendMessage').value = d.generated_message;
      document.getElementById('sendModal').classList.add('show');
    } else {
      alert('消息生成失败，请重试');
    }
  } catch(e) {
    alert('操作失败: ' + e.message);
  }
  btn.disabled = false;
  btn.textContent = '确认并生成消息';
}

// Send follow-up
function openSend(followUpId, name, message) {
  pendingFollowUpId = followUpId;
  document.getElementById('sendCustomerInfo').innerHTML = `<p>客户: <strong>${esc(name)}</strong></p>`;
  document.getElementById('sendMessage').value = message;
  document.getElementById('sendModal').classList.add('show');
}

async function doSendFollowUp() {
  const msg = document.getElementById('sendMessage').value.trim();
  if (!msg) { alert('消息不能为空'); return; }
  const btn = document.getElementById('sendBtn');
  btn.disabled = true;
  try {
    await apiFetch('/api/admin/follow-up/send', 'POST', { follow_up_id: pendingFollowUpId, message: msg });
    closeModal('sendModal');
    alert('消息已加入发送队列');
    loadFollowUps(currentFUFilter);
  } catch(e) {
    alert('操作失败: ' + e.message);
  }
  btn.disabled = false;
}

function renderFollowUpAction(f) {
  if (f.status === 'pending_approval') {
    return '<button class="btn btn-primary btn-sm" onclick="openApprove(' + f.id + ')" style="margin-right:4px">审核通过</button>'
         + '<button class="btn btn-sm" style="background:#f5f5f5;color:#e74c3c;border:1px solid #ddd;" onclick="rejectFollowUp(' + f.id + ')">驳回</button>';
  } else if (f.status === 'message_generated') {
    return '<button class="btn btn-success btn-sm" onclick="openSendById(' + f.id + ')">编辑发送</button>';
  } else if (f.status === 'failed') {
    return '<span class="text-muted">' + esc(f.error_message || '发送失败') + '</span>';
  }
  return '';
}

let _followUpCache = {};
function openSendById(followUpId) {
  const f = _followUpCache[followUpId];
  if (f) openSend(f.id, f.customer_name, f.generated_message);
}

function closeModal(id) {
  document.getElementById(id).classList.remove('show');
}

// Helpers
function esc(s) {
  if (!s) return '';
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
function escJs(s) {
  if (!s) return '';
  return String(s).replace(/\\\\/g,'\\\\\\\\').replace(/`/g,'\\\\`').replace(/\\$/g,'\\\\$');
}
function levelLabel(l) {
  return { high: '高意向', medium: '中意向', low: '低意向', none: '无意向' }[l] || l;
}
function statusLabel(s) {
  return { pending: '待处理', confirmed: '已确认', pending_approval: '待审核', message_generated: '待发送', sent: '已发送', failed: '失败', rejected: '已驳回' }[s] || s;
}
function formatTime(t) {
  if (!t) return '-';
  try { return new Date(t).toLocaleString('zh-CN', { month:'2-digit', day:'2-digit', hour:'2-digit', minute:'2-digit' }); }
  catch(e) { return t; }
}

// ── Follow-up Approval ────────────────────────────────────

function openApprove(followUpId) {
  const f = _followUpCache[followUpId];
  if (!f) return;
  pendingFollowUpId = followUpId;
  let filesInfo = '';
  try {
    const files = JSON.parse(f.attachment_files || '[]');
    if (files.length > 0) {
      filesInfo = '<p style="margin-top:8px;color:#666">附带文件：' + files.map(fn =>
        `<a href="/api/materials/${encodeURIComponent(fn)}" target="_blank" style="color:#1a73e8">📎${esc(fn)}</a>`
      ).join('  ') + '</p>';
    }
  } catch(e) {}
  document.getElementById('sendCustomerInfo').innerHTML = '<p>跟单消息审核 - 客户: <strong>' + esc(f.customer_name) + '</strong></p><p style="color:#e67e22;font-size:12px">此为跟单消息，需审核后才会发送</p>' + filesInfo;
  document.getElementById('sendMessage').value = f.generated_message || '';
  document.getElementById('sendModal').classList.add('show');
  // Override send button to do approve
  document.getElementById('sendBtn').onclick = doApproveFollowUp;
  document.getElementById('sendBtn').textContent = '审核通过并发送';
}

async function doApproveFollowUp() {
  const msg = document.getElementById('sendMessage').value.trim();
  if (!msg) { alert('消息不能为空'); return; }
  const btn = document.getElementById('sendBtn');
  btn.disabled = true;
  try {
    await apiFetch('/api/admin/follow-up/' + pendingFollowUpId + '/approve', 'POST', { follow_up_id: pendingFollowUpId, message: msg });
    closeModal('sendModal');
    alert('跟单消息已审核通过，等待机器人发送');
    loadFollowUps(currentFUFilter);
  } catch(e) {
    alert('操作失败: ' + e.message);
  }
  btn.disabled = false;
  // Restore original handler
  btn.onclick = doSendFollowUp;
  btn.textContent = '确认发送';
}

async function rejectFollowUp(followUpId) {
  if (!confirm('确定驳回该跟单消息？')) return;
  try {
    await apiFetch('/api/admin/follow-up/' + followUpId + '/reject', 'POST');
    loadFollowUps(currentFUFilter);
  } catch(e) {
    alert('操作失败: ' + e.message);
  }
}

// ── Team Member Management ────────────────────────────────

async function loadTeamMembers() {
  try {
    const d = await apiFetch('/api/admin/team-members');
    const body = document.getElementById('teamBody');
    const empty = document.getElementById('teamEmpty');
    if (!d.team_members || d.team_members.length === 0) {
      body.innerHTML = '';
      empty.style.display = 'block';
      return;
    }
    empty.style.display = 'none';
    body.innerHTML = d.team_members.map(m => `<tr>
      <td><code>${esc(m.wechat_id)}</code></td>
      <td>${esc(m.name)}</td>
      <td class="text-muted">${formatTime(m.created_at)}</td>
      <td><button class="btn btn-sm" style="background:#fee;color:#e74c3c;border:1px solid #fcc;" onclick="removeTeamMember(${m.id}, '${escJs(m.wechat_id)}')">删除</button></td>
    </tr>`).join('');
  } catch(e) {}
}

function openAddTeamMember() {
  document.getElementById('teamWechatId').value = '';
  document.getElementById('teamMemberName').value = '';
  document.getElementById('addTeamModal').classList.add('show');
}

async function doAddTeamMember() {
  const wechatId = document.getElementById('teamWechatId').value.trim();
  const name = document.getElementById('teamMemberName').value.trim();
  if (!wechatId) { alert('微信ID不能为空'); return; }
  const btn = document.getElementById('addTeamBtn');
  btn.disabled = true;
  try {
    await apiFetch('/api/admin/team-members', 'POST', { wechat_id: wechatId, name: name || wechatId });
    closeModal('addTeamModal');
    loadTeamMembers();
  } catch(e) {
    alert('添加失败: ' + (e.message || '该微信ID可能已存在'));
  }
  btn.disabled = false;
}

async function removeTeamMember(id, wechatId) {
  if (!confirm('确定删除团队成员 ' + wechatId + '？\\n删除后该成员的消息将不再被忽略。')) return;
  try {
    await apiFetch('/api/admin/team-members/' + id, 'DELETE');
    loadTeamMembers();
  } catch(e) {
    alert('删除失败: ' + e.message);
  }
}

// ── Knowledge Base Management ─────────────────────────────

let currentKBFilter = null;
let currentReviewDocId = null;
let editingChunkId = null;

// Upload zone drag & drop
(function() {
  const zone = document.getElementById('uploadZone');
  if (!zone) return;
  zone.addEventListener('dragover', function(e) { e.preventDefault(); zone.classList.add('drag-over'); });
  zone.addEventListener('dragleave', function() { zone.classList.remove('drag-over'); });
  zone.addEventListener('drop', function(e) {
    e.preventDefault();
    zone.classList.remove('drag-over');
    if (e.dataTransfer.files.length > 0) uploadKBFile(e.dataTransfer.files[0]);
  });
})();

function handleKBUpload(input) {
  if (input.files.length > 0) {
    uploadKBFile(input.files[0]);
    input.value = '';
  }
}

async function uploadKBFile(file) {
  const zone = document.getElementById('uploadZone');
  const origText = zone.querySelector('.text').textContent;
  zone.querySelector('.text').textContent = '上传中: ' + file.name + ' ...';
  try {
    const formData = new FormData();
    formData.append('file', file);
    const res = await fetch('/api/admin/kb/upload', {
      method: 'POST',
      headers: { 'Authorization': 'Bearer ' + token },
      body: formData
    });
    if (res.status === 401) { document.getElementById('loginOverlay').style.display = 'flex'; return; }
    if (!res.ok) throw new Error(await res.text());
    const d = await res.json();
    zone.querySelector('.text').textContent = '上传成功! 文档正在处理中...';
    setTimeout(function() { zone.querySelector('.text').textContent = origText; }, 3000);
    loadKBDocuments();
    // Poll for processing completion
    pollDocStatus(d.document_id);
  } catch(e) {
    zone.querySelector('.text').textContent = '上传失败: ' + e.message;
    setTimeout(function() { zone.querySelector('.text').textContent = origText; }, 5000);
  }
}

function pollDocStatus(docId) {
  let attempts = 0;
  const iv = setInterval(async function() {
    attempts++;
    if (attempts > 120) { clearInterval(iv); return; } // max 10 min
    try {
      const d = await apiFetch('/api/admin/kb/documents/' + docId);
      if (d.document && d.document.status !== 'processing' && d.document.status !== 'uploaded') {
        clearInterval(iv);
        loadKBDocuments();
      }
    } catch(e) { clearInterval(iv); }
  }, 5000);
}

function kbStatusLabel(s) {
  return { uploaded: '已上传', processing: '处理中', ready: '待发布', published: '已发布', failed: '失败' }[s] || s;
}

function formatFileSize(bytes) {
  if (!bytes || bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return (bytes / Math.pow(k, i)).toFixed(1) + ' ' + sizes[i];
}

async function loadKBDocuments(status) {
  try {
    let url = '/api/admin/kb/documents?limit=100';
    if (status) url += '&status=' + status;
    const d = await apiFetch(url);
    // Update stats
    if (d.stats) {
      document.getElementById('kbStatsRow').innerHTML =
        '<div class="stat-card"><div class="label">总文档数</div><div class="value">' + d.stats.total_documents + '</div></div>' +
        '<div class="stat-card"><div class="label">已发布</div><div class="value">' + d.stats.published_documents + '</div></div>' +
        '<div class="stat-card"><div class="label">待发布</div><div class="value">' + d.stats.ready_documents + '</div></div>' +
        '<div class="stat-card"><div class="label">处理中</div><div class="value">' + d.stats.processing_documents + '</div></div>' +
        '<div class="stat-card"><div class="label">总切片数</div><div class="value">' + d.stats.total_chunks + '</div></div>';
    }
    const body = document.getElementById('kbDocsBody');
    const empty = document.getElementById('kbDocsEmpty');
    if (!d.documents || d.documents.length === 0) {
      body.innerHTML = '';
      empty.style.display = 'block';
      return;
    }
    empty.style.display = 'none';
    body.innerHTML = d.documents.map(function(doc) {
      let actions = '';
      if (doc.status === 'ready' || doc.status === 'published') {
        actions += '<button class="btn btn-primary btn-sm" onclick="openChunkReview(' + doc.id + ')">审核标注</button> ';
      }
      if (doc.status === 'ready') {
        actions += '<button class="btn btn-success btn-sm" onclick="doPublishDocDirect(' + doc.id + ')">发布</button> ';
      }
      if (doc.status === 'failed') {
        actions += '<span class="text-muted" title="' + esc(doc.error_message) + '">查看错误</span> ';
      }
      actions += '<button class="btn btn-sm" style="background:#fce4ec;color:#c62828;" onclick="doDeleteDoc(' + doc.id + ')">删除</button>';
      return '<tr>' +
        '<td><strong>' + esc(doc.file_name) + '</strong></td>' +
        '<td>' + esc(doc.file_type) + '</td>' +
        '<td>' + formatFileSize(doc.file_size) + '</td>' +
        '<td>' + (doc.total_chunks || 0) + '</td>' +
        '<td><span class="kb-status ' + doc.status + '">' + kbStatusLabel(doc.status) + '</span></td>' +
        '<td class="text-muted">' + formatTime(doc.uploaded_at) + '</td>' +
        '<td>' + actions + '</td>' +
        '</tr>';
    }).join('');
  } catch(e) {}
}

function filterKBDocs(status, btn) {
  currentKBFilter = status;
  document.querySelectorAll('#panel-kb .filter-btn').forEach(function(b) { b.classList.remove('active'); });
  btn.classList.add('active');
  loadKBDocuments(status);
}

// Chunk review
async function openChunkReview(docId) {
  currentReviewDocId = docId;
  try {
    const d = await apiFetch('/api/admin/kb/documents/' + docId);
    document.getElementById('chunkReviewTitle').textContent = '切片审核 - ' + (d.document.file_name || '');
    const pubBtn = document.getElementById('publishFromReviewBtn');
    pubBtn.style.display = (d.document.status === 'ready' || d.document.status === 'published') ? '' : 'none';
    const body = document.getElementById('chunkReviewBody');
    if (!d.chunks || d.chunks.length === 0) {
      body.innerHTML = '<tr><td colspan="8" class="empty">暂无切片数据</td></tr>';
    } else {
      body.innerHTML = d.chunks.map(function(c) {
        return '<tr>' +
          '<td>' + c.chunk_index + '</td>' +
          '<td><div class="chunk-text-preview">' + esc(c.text) + '</div></td>' +
          '<td>' + esc(c.category) + '</td>' +
          '<td>' + esc(c.keywords) + '</td>' +
          '<td><div class="chunk-text-preview">' + esc(c.summary) + '</div></td>' +
          '<td>' + esc(c.importance) + '</td>' +
          '<td>' + esc(c.related_products || '-') + '</td>' +
          '<td><button class="btn btn-primary btn-sm" onclick="openChunkEdit(' + c.id + ')">编辑</button></td>' +
          '</tr>';
      }).join('');
    }
    document.getElementById('chunkReviewModal').classList.add('show');
  } catch(e) { alert('加载失败: ' + e.message); }
}

// Chunk edit
async function openChunkEdit(chunkId) {
  editingChunkId = chunkId;
  try {
    // Find chunk data from the review table
    const d = await apiFetch('/api/admin/kb/documents/' + currentReviewDocId);
    const chunk = d.chunks.find(function(c) { return c.id === chunkId; });
    if (!chunk) { alert('切片不存在'); return; }
    document.getElementById('editChunkPreview').textContent = chunk.text.substring(0, 500);
    document.getElementById('editCategory').value = chunk.category || '其他';
    document.getElementById('editKeywords').value = chunk.keywords || '';
    document.getElementById('editSummary').value = chunk.summary || '';
    document.getElementById('editImportance').value = chunk.importance || '中';
    // Set product checkboxes
    const products = (chunk.related_products || '').split(',').map(function(s) { return s.trim(); });
    document.querySelectorAll('#editProducts input[type=checkbox]').forEach(function(cb) {
      cb.checked = products.indexOf(cb.value) >= 0;
    });
    document.getElementById('chunkEditModal').classList.add('show');
  } catch(e) { alert('加载失败: ' + e.message); }
}

async function doSaveChunkEdit() {
  const category = document.getElementById('editCategory').value;
  const keywords = document.getElementById('editKeywords').value;
  const summary = document.getElementById('editSummary').value;
  const importance = document.getElementById('editImportance').value;
  const products = [];
  document.querySelectorAll('#editProducts input[type=checkbox]:checked').forEach(function(cb) {
    products.push(cb.value);
  });
  try {
    await apiFetch('/api/admin/kb/chunks/' + editingChunkId, 'PUT', {
      category: category,
      keywords: keywords,
      summary: summary,
      importance: importance,
      related_products: products.join(', ')
    });
    closeModal('chunkEditModal');
    // Refresh chunk review
    openChunkReview(currentReviewDocId);
  } catch(e) { alert('保存失败: ' + e.message); }
}

// Publish
async function doPublishDoc() {
  if (!currentReviewDocId) return;
  if (!confirm('确认将文档发布到知识库?')) return;
  try {
    const d = await apiFetch('/api/admin/kb/documents/' + currentReviewDocId + '/publish', 'POST');
    alert('发布成功! 已写入 ' + (d.chunks_published || 0) + ' 个切片到知识库');
    closeModal('chunkReviewModal');
    loadKBDocuments(currentKBFilter);
  } catch(e) { alert('发布失败: ' + e.message); }
}

async function doPublishDocDirect(docId) {
  if (!confirm('确认将文档发布到知识库?')) return;
  try {
    const d = await apiFetch('/api/admin/kb/documents/' + docId + '/publish', 'POST');
    alert('发布成功! 已写入 ' + (d.chunks_published || 0) + ' 个切片到知识库');
    loadKBDocuments(currentKBFilter);
  } catch(e) { alert('发布失败: ' + e.message); }
}

// Delete
async function doDeleteDoc(docId) {
  if (!confirm('确认删除此文档? 如已发布将同时从知识库中移除。')) return;
  try {
    await apiFetch('/api/admin/kb/documents/' + docId, 'DELETE');
    loadKBDocuments(currentKBFilter);
  } catch(e) { alert('删除失败: ' + e.message); }
}
</script>
</body>
</html>
"""
