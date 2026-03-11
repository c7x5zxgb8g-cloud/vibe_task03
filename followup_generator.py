"""AI-powered follow-up message generation using DeepSeek."""

from openai import OpenAI

from config import Config
from utils.logger import get_logger

logger = get_logger(__name__)

_FOLLOWUP_SYSTEM_PROMPT = """你是理博基金的客户跟进助手。你需要根据客户的聊天记录和意向分析，生成一条个性化的私信跟进消息。

## 要求
1. 语气专业、温暖、不过于销售化
2. 结合客户具体询问的内容来回应
3. 长度控制在100-200字
4. 自然引导客户进一步了解或预约咨询
5. 不做收益承诺，涉及业绩数据时附加风险提示
6. 自称"我"，称客户为"您"
7. 不使用markdown格式

## 公司核心信息
- 公司: 杭州理博私募基金管理有限公司
- 品牌理念: 辞简理博 / LESS IS MORE
- 核心策略: 量化选股、可转债增强、股票择时
- 最低起投: 100万
- 创始人: 王黎博士（清华+密歇根大学，前贝莱德基金经理）
"""


def generate_followup_message(
    customer_name: str,
    message_content: str,
    intent_summary: str,
    recent_messages: list[dict] | None = None,
    admin_note: str = "",
) -> str:
    """Generate a personalized follow-up message for the customer.

    Returns the generated message text, or empty string on failure.
    """
    client = OpenAI(
        api_key=Config.DEEPSEEK_API_KEY,
        base_url=Config.DEEPSEEK_BASE_URL,
    )

    user_prompt = f"客户名称: {customer_name}\n"
    user_prompt += f"客户消息: {message_content}\n"
    user_prompt += f"意向分析: {intent_summary}\n"

    if recent_messages:
        context_lines = [f"- {m.get('content', '')[:100]}" for m in recent_messages[-5:]]
        user_prompt += "近期消息:\n" + "\n".join(context_lines) + "\n"

    if admin_note:
        user_prompt += f"管理员备注: {admin_note}\n"

    user_prompt += "\n请生成跟进消息:"

    try:
        resp = client.chat.completions.create(
            model=Config.DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": _FOLLOWUP_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=500,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Follow-up generation failed: {e}")
        return ""
