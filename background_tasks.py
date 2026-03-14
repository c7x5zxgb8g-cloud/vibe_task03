"""Background task processing for message analysis pipeline."""

import asyncio
import json

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
)
from intent_analyzer import analyze_intent
from feishu_notifier import notify_high_intent_lead
from followup_generator import generate_followup_message
from utils.logger import get_logger

logger = get_logger(__name__)

_running = False

# Reference to the RAG engine, set by the server at startup
_rag_engine = None


def set_rag_engine(engine):
    """Set the RAG engine reference for private chat auto-replies."""
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


def _check_knowledge_base_has_answer(query: str) -> dict | None:
    """Check if the knowledge base has a relevant answer for the query.

    Returns dict with 'reply' key if a relevant answer exists, None otherwise.
    Only replies when there are sufficiently relevant KB results.
    """
    if _rag_engine is None:
        return None

    try:
        # Search the knowledge base for relevant content
        hits = _rag_engine._store.search(query, top_k=4)

        # Filter for highly relevant results only (score > 0.5)
        relevant_hits = [h for h in hits if h.get("score", 0) > 0.5]

        if not relevant_hits:
            logger.info(f"No relevant KB content for query: {query[:50]}...")
            return None

        # Generate answer using RAG engine
        reply = _rag_engine.chat(query=query, history=None, top_k=4)

        if not reply or "无法回答" in reply or "暂时出现问题" in reply:
            return None

        return {"reply": reply}
    except Exception as e:
        logger.error(f"Knowledge base check failed: {e}")
        return None


def _process_pending_messages():
    """Process all pending messages: analyze intent + notify if high.

    Reply logic:
    1. For high/medium intent: pre-generate suggested reply and notify
       sales on Feishu for confirmation before sending.
    2. For private chat messages: auto-reply ONLY if the answer exists
       in the knowledge base. Otherwise skip and wait for manual reply.
    """
    messages = get_unprocessed_messages(limit=10)

    for msg in messages:
        try:
            customer = get_customer(msg["customer_id"])
            if not customer:
                mark_message_processed(msg["id"])
                continue

            # Skip very short messages (greetings, etc.)
            if len(msg["content"].strip()) < 4:
                mark_message_processed(msg["id"])
                continue

            # Get recent messages for context
            recent = get_messages_by_customer(msg["customer_id"], limit=5)
            recent_context = "\n".join(
                f"[{m['received_at']}] {m['content'][:100]}" for m in recent
            )

            # Determine message source: private or group
            msg_source = "private" if not msg.get("group_chat_id") else "group"

            # Analyze intent via DeepSeek
            result = analyze_intent(
                message_content=msg["content"],
                customer_name=customer["name"],
                recent_context=recent_context,
                msg_source=msg_source,
            )

            # Store analysis
            analysis_id = insert_intent_analysis(
                message_id=msg["id"],
                customer_id=msg["customer_id"],
                intent_level=result["intent_level"],
                intent_score=result["intent_score"],
                intent_summary=result["intent_summary"],
                keywords=json.dumps(result.get("keywords", []), ensure_ascii=False),
                raw_response=result.get("raw_response", ""),
            )

            # Handle high/medium intent: pre-generate reply, notify sales on Feishu
            if result["intent_level"] in ("high", "medium"):
                # Pre-generate suggested follow-up message with file recommendations
                followup_result = generate_followup_message(
                    customer_name=customer["name"],
                    message_content=msg["content"],
                    intent_summary=result["intent_summary"],
                    recent_messages=recent,
                )
                suggested_reply = followup_result.get("message", "") if followup_result else ""
                attachment_files = followup_result.get("recommended_files", []) if followup_result else []

                # Notify admin via Feishu with suggested reply and attachments
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

            # Handle private chat auto-reply (only for non-high-intent messages)
            elif msg_source == "private" and result["intent_level"] in ("low", "none"):
                # Only auto-reply if answer exists in the knowledge base
                kb_result = _check_knowledge_base_has_answer(msg["content"])
                if kb_result:
                    # Create and auto-send a follow-up with the KB answer
                    follow_up_id = create_follow_up(
                        customer_id=msg["customer_id"],
                        intent_analysis_id=analysis_id,
                        target_user_id=customer["wechat_user_id"],
                    )
                    from datetime import datetime
                    update_follow_up(
                        follow_up_id,
                        status="message_generated",
                        generated_message=kb_result["reply"],
                        confirmed_at=datetime.now().isoformat(),
                    )
                    logger.info(
                        f"Auto-reply queued for private message {msg['id']} "
                        f"from {customer['name']}"
                    )
                else:
                    logger.info(
                        f"No KB answer for private message {msg['id']} "
                        f"from {customer['name']}, waiting for manual reply"
                    )

            mark_message_processed(msg["id"])
            logger.info(
                f"Processed message {msg['id']}: intent={result['intent_level']} "
                f"score={result['intent_score']:.2f} source={msg_source}"
            )
        except Exception as e:
            logger.error(f"Failed to process message {msg['id']}: {e}")
            # Don't mark as processed; will retry next cycle
