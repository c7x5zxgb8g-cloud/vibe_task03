"""Feishu (飞书) webhook robot notification."""

import httpx

from config import Config
from utils.logger import get_logger

logger = get_logger(__name__)


def notify_high_intent_lead(
    customer_name: str,
    customer_wechat_id: str,
    group_chat_name: str,
    message_content: str,
    intent_level: str,
    intent_score: float,
    intent_summary: str,
    keywords: list[str],
    admin_page_url: str = "",
    suggested_reply: str = "",
    attachment_files: list[str] | None = None,
) -> bool:
    """Send a Feishu notification about a high-intent lead.

    Returns True if sent successfully.
    """
    webhook_url = Config.FEISHU_WEBHOOK_URL
    if not webhook_url:
        logger.warning("Feishu webhook URL not configured, skipping notification")
        return False

    level_label = {"high": "高", "medium": "中"}.get(intent_level, intent_level)
    level_color = "red" if intent_level == "high" else "orange"

    elements = [
        {
            "tag": "div",
            "fields": [
                {"is_short": True, "text": {"tag": "lark_md", "content": f"**客户**: {customer_name}"}},
                {"is_short": True, "text": {"tag": "lark_md", "content": f"**来源群**: {group_chat_name}"}},
                {"is_short": True, "text": {"tag": "lark_md", "content": f"**意向等级**: {level_label}"}},
                {"is_short": True, "text": {"tag": "lark_md", "content": f"**意向分数**: {intent_score:.0%}"}},
            ],
        },
        {"tag": "div", "text": {"tag": "lark_md", "content": f"**客户消息**:\n{message_content[:500]}"}},
        {"tag": "div", "text": {"tag": "lark_md", "content": f"**判断依据**: {intent_summary}"}},
        {"tag": "div", "text": {"tag": "lark_md", "content": f"**关键词**: {', '.join(keywords)}"}},
    ]

    # Add suggested reply if available
    if suggested_reply:
        elements.append({"tag": "hr"})
        elements.append({"tag": "div", "text": {"tag": "lark_md", "content": f"**建议回复**:\n{suggested_reply[:500]}"}})

    # Add attachment file info if available
    if attachment_files:
        base_url = admin_page_url or Config.ADMIN_BASE_URL
        files_text = "\n".join(f"- [{f}]({base_url}/api/materials/{f})" for f in attachment_files)
        elements.append({"tag": "div", "text": {"tag": "lark_md", "content": f"**推荐附件**:\n{files_text}"}})

    elements.append({"tag": "hr"})
    elements.append({
        "tag": "div",
        "text": {
            "tag": "lark_md",
            "content": f"请前往 [管理后台]({admin_page_url or Config.ADMIN_BASE_URL + '/admin'}) 确认跟进",
        },
    })

    card = {
        "msg_type": "interactive",
        "card": {
            "header": {
                "title": {
                    "tag": "plain_text",
                    "content": f"🔥 {level_label}意向客户提醒 - {customer_name}",
                },
                "template": level_color,
            },
            "elements": elements,
        },
    }

    try:
        resp = httpx.post(webhook_url, json=card, timeout=10)
        resp.raise_for_status()
        result = resp.json()
        if result.get("code") == 0 or result.get("StatusCode") == 0:
            logger.info(f"Feishu notification sent for customer: {customer_name}")
            return True
        else:
            logger.error(f"Feishu notification failed: {result}")
            return False
    except Exception as e:
        logger.error(f"Feishu notification error: {e}")
        return False
