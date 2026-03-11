"""Purchase intent analysis using DeepSeek API (OpenAI-compatible)."""

import json

from openai import OpenAI

from config import Config
from utils.logger import get_logger

logger = get_logger(__name__)

_deepseek_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _deepseek_client
    if _deepseek_client is None:
        _deepseek_client = OpenAI(
            api_key=Config.DEEPSEEK_API_KEY,
            base_url=Config.DEEPSEEK_BASE_URL,
        )
    return _deepseek_client


_INTENT_SYSTEM_PROMPT = """你是一个购买意向分析专家，专门分析私募基金客户的对话消息。

你需要分析客户消息中的购买意向，并给出判断结果。

## 判断标准

**高意向 (high)** - 分数 0.7-1.0:
- 直接询问如何购买/申购基金产品
- 询问具体产品的费率、起投金额、开放日
- 表示想要投资/配置资金
- 询问合格投资者要求（暗示自己可能符合条件）
- 提到具体金额打算投入
- 询问认购流程或合同签署

**中意向 (medium)** - 分数 0.4-0.69:
- 询问产品业绩、收益情况
- 对比不同策略/产品
- 询问公司资质、团队背景
- 询问资金安全性
- 表示对量化投资感兴趣
- 要求发送产品资料

**低意向 (low)** - 分数 0.1-0.39:
- 一般性咨询公司信息
- 询问行业/市场观点
- 转发新闻要求评论
- 泛泛而谈的投资话题

**无意向 (none)** - 分数 0.0-0.09:
- 闲聊、问候
- 无关话题
- 广告/垃圾信息

## 输出格式 (严格JSON)

{
  "intent_level": "high|medium|low|none",
  "intent_score": 0.0到1.0的浮点数,
  "intent_summary": "一句话说明判断依据",
  "keywords": ["关键词1", "关键词2"]
}"""


def analyze_intent(message_content: str, customer_name: str = "",
                   recent_context: str = "") -> dict:
    """Analyze a message for purchase intent.

    Returns dict with: intent_level, intent_score, intent_summary, keywords, raw_response
    """
    client = _get_client()

    user_prompt = f"客户名称: {customer_name}\n"
    if recent_context:
        user_prompt += f"近期消息上下文:\n{recent_context}\n\n"
    user_prompt += f"当前消息: {message_content}\n\n请分析购买意向:"

    try:
        resp = client.chat.completions.create(
            model=Config.DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": _INTENT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=300,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content
        result = json.loads(raw)
        result["raw_response"] = raw
        return result
    except Exception as e:
        logger.error(f"Intent analysis failed: {e}")
        return {
            "intent_level": "none",
            "intent_score": 0.0,
            "intent_summary": f"分析失败: {e}",
            "keywords": [],
            "raw_response": "",
        }
