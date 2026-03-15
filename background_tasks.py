"""Background task processing for message analysis pipeline."""

import asyncio
import json
import os
import re
from datetime import datetime

from config import Config
from database import (
    get_unprocessed_messages,
    mark_message_processed,
    insert_intent_analysis,
    mark_intent_notified,
    get_messages_by_customer,
    get_customer,
    create_follow_up,
    update_follow_up,
    has_recent_wechat_card,
)
from intent_analyzer import analyze_intent
from feishu_notifier import notify_high_intent_lead
from followup_generator import generate_followup_message
from utils.logger import get_logger

logger = get_logger(__name__)

_running = False

# Reference to the RAG engine, set by the server at startup
_rag_engine = None

_BASE_DIR = os.path.dirname(__file__)
_MATERIALS_DIR = os.path.join(_BASE_DIR, "docs", "理博基金知识库")
_WECHAT_IMAGE_DIR = os.path.join(_BASE_DIR, "static", "wechat")

_NO_ANSWER_REPLY = "这个问题我不太确定，我帮您问一下同事，稍后给您回复。"


def _extract_attachments_from_reply(reply_text: str) -> tuple[str, list[str]]:
    """Extract attachment markers from reply text and return cleaned text + file list.

    Handles:
      - [微信名片] → adds wechat card image to attachments
      - [材料推荐:filename] → adds the file to attachments

    Returns (cleaned_text, attachment_filenames)
    """
    attachments = []

    # Extract [材料推荐:filename]
    material_pattern = re.compile(r'\[材料推荐[:：]([^\]]+)\]')
    for match in material_pattern.finditer(reply_text):
        fname = match.group(1).strip()
        fpath = os.path.join(_MATERIALS_DIR, fname)
        if os.path.isfile(fpath):
            attachments.append(fname)
        else:
            logger.warning(f"Material file not found: {fname}")
    reply_text = material_pattern.sub('', reply_text)

    # Extract [微信名片]
    if '[微信名片]' in reply_text:
        # Find the wechat card image file
        wechat_card_file = None
        if os.path.isdir(_WECHAT_IMAGE_DIR):
            for f in os.listdir(_WECHAT_IMAGE_DIR):
                if f.startswith('wechat_card'):
                    wechat_card_file = f
                    break
        if wechat_card_file:
            attachments.append(f"__wechat_card__:{wechat_card_file}")
        reply_text = reply_text.replace('[微信名片]', '')

    # Clean up extra whitespace/newlines left by marker removal
    reply_text = re.sub(r'\n{3,}', '\n\n', reply_text).strip()

    return reply_text, attachments


def set_rag_engine(engine):
    """Set the RAG engine reference for message auto-replies."""
    global _rag_engine
    _rag_engine = engine


async def start_message_processor(interval_seconds: int = 5):
    """Background loop that processes unprocessed messages.

    Called once at startup via lifespan. Runs forever until shutdown.
    """
    global _running
    _running = True
    logger.info("Background message processor started")

    while _running:
        try:
            await asyncio.to_thread(_process_pending_messages)
        except Exception as e:
            logger.error(f"Message processor error: {e}")
        await asyncio.sleep(interval_seconds)


def stop_message_processor():
    global _running
    _running = False
    logger.info("Background message processor stopped")


def _check_knowledge_base(query: str) -> dict | None:
    """Check if the knowledge base has a relevant answer for the query.

    Returns dict with 'reply' and 'has_answer' keys.
    has_answer=True means KB had relevant content, False means no match.
    """
    if _rag_engine is None:
        return None

    try:
        hits = _rag_engine._store.search(query, top_k=4)
        relevant_hits = [h for h in hits if h.get("score", 0) > 0.5]

        if not relevant_hits:
            logger.info(f"No relevant KB content for query: {query[:50]}...")
            return {"reply": "", "has_answer": False}

        reply = _rag_engine.chat(query=query, history=None, top_k=4)

        if not reply or "无法回答" in reply or "暂时出现问题" in reply:
            return {"reply": "", "has_answer": False}

        return {"reply": reply, "has_answer": True}
    except Exception as e:
        logger.error(f"Knowledge base check failed: {e}")
        return None


def _process_pending_messages():
    """Process all pending messages.

    Reply logic:
    1. Use RAG engine (project's own agent) to generate all replies.
    2. Use DeepSeek only for intent analysis (whether to follow up for sales).
    3. Group chat messages:
       - KB has answer → auto-reply directly (no admin approval needed)
       - KB no answer → reply "不太确定，帮您问同事" (no admin approval needed)
    4. Private chat messages:
       - KB has answer → auto-reply directly (no admin approval needed)
       - KB no answer → do NOT reply, wait for manual handling
    5. High/medium intent (any source) → also create a follow-up record
       that requires admin approval before sending (跟单消息).
    """
    messages = get_unprocessed_messages(limit=10)

    for msg in messages:
        try:
            customer = get_customer(msg["customer_id"])
            if not customer:
                mark_message_processed(msg["id"])
                continue

            if len(msg["content"].strip()) < 4:
                mark_message_processed(msg["id"])
                continue

            # Get recent messages for context
            recent = get_messages_by_customer(msg["customer_id"], limit=5)
            recent_context = "\n".join(
                f"[{m['received_at']}] {m['content'][:100]}" for m in recent
            )

            # Determine message source: group_chat_name is the key indicator
            msg_source = "private" if not customer.get("group_chat_name") else "group"

            # ── Step 1: Intent analysis via DeepSeek ──
            result = analyze_intent(
                message_content=msg["content"],
                customer_name=customer["name"],
                recent_context=recent_context,
                msg_source=msg_source,
            )

            analysis_id = insert_intent_analysis(
                message_id=msg["id"],
                customer_id=msg["customer_id"],
                intent_level=result["intent_level"],
                intent_score=result["intent_score"],
                intent_summary=result["intent_summary"],
                keywords=json.dumps(result.get("keywords", []), ensure_ascii=False),
                raw_response=result.get("raw_response", ""),
            )

            # ── Step 2: Generate reply using RAG engine ──
            kb_result = _check_knowledge_base(msg["content"])

            # Check if we recently sent a wechat card (group-level 12h dedup)
            group_id = msg.get("group_chat_id") or customer.get("group_chat_id") or ""
            if group_id:
                skip_wechat_card = has_recent_wechat_card(group_chat_id=group_id, hours=12)
            else:
                skip_wechat_card = has_recent_wechat_card(customer_id=msg["customer_id"], hours=12)

            if msg_source == "group":
                # Group chat: always reply
                if kb_result and kb_result["has_answer"]:
                    reply_text = kb_result["reply"]
                else:
                    reply_text = _NO_ANSWER_REPLY

                # Extract [微信名片] and [材料推荐:xxx] markers into attachment files
                reply_text, reply_attachments = _extract_attachments_from_reply(reply_text)

                # Remove wechat card if already sent recently
                if skip_wechat_card:
                    reply_attachments = [a for a in reply_attachments if not a.startswith("__wechat_card__:")]

                # Auto-reply, no admin approval needed
                follow_up_id = create_follow_up(
                    customer_id=msg["customer_id"],
                    intent_analysis_id=analysis_id,
                    target_user_id=customer["wechat_user_id"],
                    reply_type="auto",
                )
                update_follow_up(
                    follow_up_id,
                    status="message_generated",
                    generated_message=reply_text,
                    attachment_files=json.dumps(reply_attachments, ensure_ascii=False) if reply_attachments else "[]",
                    confirmed_at=datetime.now().isoformat(),
                )
                logger.info(
                    f"Auto-reply queued for group message {msg['id']} "
                    f"from {customer['name']} (kb_match={kb_result and kb_result['has_answer']}, "
                    f"attachments={len(reply_attachments)}, card_skipped={skip_wechat_card})"
                )

            elif msg_source == "private":
                # Private chat: only reply if KB has answer
                if kb_result and kb_result["has_answer"]:
                    reply_text = kb_result["reply"]
                    reply_text, reply_attachments = _extract_attachments_from_reply(reply_text)

                    # Remove wechat card if already sent recently
                    if skip_wechat_card:
                        reply_attachments = [a for a in reply_attachments if not a.startswith("__wechat_card__:")]

                    follow_up_id = create_follow_up(
                        customer_id=msg["customer_id"],
                        intent_analysis_id=analysis_id,
                        target_user_id=customer["wechat_user_id"],
                        reply_type="auto",
                    )
                    update_follow_up(
                        follow_up_id,
                        status="message_generated",
                        generated_message=reply_text,
                        attachment_files=json.dumps(reply_attachments, ensure_ascii=False) if reply_attachments else "[]",
                        confirmed_at=datetime.now().isoformat(),
                    )
                    logger.info(
                        f"Auto-reply queued for private message {msg['id']} "
                        f"from {customer['name']} (attachments={len(reply_attachments)}, card_skipped={skip_wechat_card})"
                    )
                else:
                    logger.info(
                        f"No KB answer for private message {msg['id']} "
                        f"from {customer['name']}, waiting for manual reply"
                    )

            # ── Step 3: High/medium intent → create follow-up for sales (needs admin approval) ──
            if result["intent_level"] in ("high", "medium"):
                # Generate sales follow-up message with file recommendations
                followup_result = generate_followup_message(
                    customer_name=customer["name"],
                    message_content=msg["content"],
                    intent_summary=result["intent_summary"],
                    recent_messages=recent,
                )
                suggested_reply = followup_result.get("message", "") if followup_result else ""
                attachment_files = followup_result.get("recommended_files", []) if followup_result else []

                # Create follow-up record, status=pending_approval (needs admin to approve)
                fu_id = create_follow_up(
                    customer_id=msg["customer_id"],
                    intent_analysis_id=analysis_id,
                    target_user_id=customer["wechat_user_id"],
                    reply_type="follow_up",
                )
                update_follow_up(
                    fu_id,
                    status="pending_approval",
                    generated_message=suggested_reply,
                    attachment_files=json.dumps(attachment_files, ensure_ascii=False),
                )
                logger.info(
                    f"Follow-up created (pending_approval) for message {msg['id']}, "
                    f"intent={result['intent_level']}"
                )

                # Notify admin via Feishu
                success = notify_high_intent_lead(
                    customer_name=customer["name"],
                    customer_wechat_id=customer["wechat_user_id"],
                    group_chat_name=customer.get("group_chat_name", ""),
                    message_content=msg["content"],
                    intent_level=result["intent_level"],
                    intent_score=result["intent_score"],
                    intent_summary=result["intent_summary"],
                    keywords=result.get("keywords", []),
                    suggested_reply=suggested_reply,
                    attachment_files=attachment_files,
                )
                if success:
                    mark_intent_notified(analysis_id)

            mark_message_processed(msg["id"])
            logger.info(
                f"Processed message {msg['id']}: intent={result['intent_level']} "
                f"score={result['intent_score']:.2f} source={msg_source}"
            )
        except Exception as e:
            logger.error(f"Failed to process message {msg['id']}: {e}")
