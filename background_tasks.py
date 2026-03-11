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
)
from intent_analyzer import analyze_intent
from feishu_notifier import notify_high_intent_lead
from utils.logger import get_logger

logger = get_logger(__name__)

_running = False


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


def _process_pending_messages():
    """Process all pending messages: analyze intent + notify if high."""
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

            # Analyze intent via DeepSeek
            result = analyze_intent(
                message_content=msg["content"],
                customer_name=customer["name"],
                recent_context=recent_context,
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

            # Notify admin if high/medium intent
            if result["intent_level"] in ("high", "medium"):
                success = notify_high_intent_lead(
                    customer_name=customer["name"],
                    customer_wechat_id=customer["wechat_user_id"],
                    group_chat_name=customer.get("group_chat_name", ""),
                    message_content=msg["content"],
                    intent_level=result["intent_level"],
                    intent_score=result["intent_score"],
                    intent_summary=result["intent_summary"],
                    keywords=result.get("keywords", []),
                )
                if success:
                    mark_intent_notified(analysis_id)

            mark_message_processed(msg["id"])
            logger.info(
                f"Processed message {msg['id']}: intent={result['intent_level']} "
                f"score={result['intent_score']:.2f}"
            )
        except Exception as e:
            logger.error(f"Failed to process message {msg['id']}: {e}")
            # Don't mark as processed; will retry next cycle
