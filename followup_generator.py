"""AI-powered follow-up message generation using DeepSeek."""

import json
import os

from openai import OpenAI

from config import Config
from utils.logger import get_logger

logger = get_logger(__name__)

# Materials directory
_MATERIALS_DIR = os.path.join(os.path.dirname(__file__), "docs", "理博基金知识库")
_SHAREABLE_EXTENSIONS = {".pdf", ".docx", ".doc", ".xlsx", ".xls", ".pptx", ".ppt", ".txt"}

_FOLLOWUP_SYSTEM_PROMPT = """你是理博基金的客户跟进助手。你需要根据客户的聊天记录和意向分析，生成一条个性化的私信跟进消息。

## 要求
1. 语气专业、温暖、不过于销售化，像真人在微信上打字一样自然
2. 结合客户具体询问的内容来回应
3. 长度控制在100-200字
4. 自然引导客户进一步了解或预约咨询
5. 不做收益承诺，涉及业绩数据时附加风险提示
6. 自称"我"，称客户为"您"
7. 绝对不要使用任何markdown格式（不要用**加粗、#标题、- 列表、1. 编号等），直接用自然语言
8. 如果有相关的材料可以推荐给客户，在消息末尾提到"我给您发一份相关材料供参考"

## 公司核心信息
- 公司: 杭州理博私募基金管理有限公司
- 品牌理念: 辞简理博 / LESS IS MORE
- 核心策略: 量化选股、可转债增强、股票择时
- 最低起投: 100万
- 创始人: 王黎博士（清华+密歇根大学，前贝莱德基金经理）

## 可用材料列表
{materials_list}

## 输出格式 (严格JSON)
请以JSON格式返回，包含以下字段：
{{
  "message": "跟进消息文本内容",
  "recommended_files": ["推荐的文件名1", "推荐的文件名2"]
}}

recommended_files 中的文件名必须是上方可用材料列表中的文件名，如果没有合适的材料则返回空数组。
"""


def _get_available_materials() -> list[str]:
    """Return list of shareable material filenames."""
    materials = []
    if not os.path.isdir(_MATERIALS_DIR):
        return materials
    for f in sorted(os.listdir(_MATERIALS_DIR)):
        ext = os.path.splitext(f)[1].lower()
        if ext in _SHAREABLE_EXTENSIONS:
            materials.append(f)
    return materials


def generate_followup_message(
    customer_name: str,
    message_content: str,
    intent_summary: str,
    recent_messages: list[dict] | None = None,
    admin_note: str = "",
) -> dict:
    """Generate a personalized follow-up message for the customer.

    Returns dict with keys:
      - message: str (generated text)
      - recommended_files: list[str] (filenames to attach)
    Returns empty dict on failure.
    """
    client = OpenAI(
        api_key=Config.DEEPSEEK_API_KEY,
        base_url=Config.DEEPSEEK_BASE_URL,
    )

    # Build materials list for the prompt
    materials = _get_available_materials()
    if materials:
        materials_list = "\n".join(f"- {m}" for m in materials)
    else:
        materials_list = "（暂无可分享的材料）"

    system_prompt = _FOLLOWUP_SYSTEM_PROMPT.format(materials_list=materials_list)

    user_prompt = f"客户名称: {customer_name}\n"
    user_prompt += f"客户消息: {message_content}\n"
    user_prompt += f"意向分析: {intent_summary}\n"

    if recent_messages:
        context_lines = [f"- {m.get('content', '')[:100]}" for m in recent_messages[-5:]]
        user_prompt += "近期消息:\n" + "\n".join(context_lines) + "\n"

    if admin_note:
        user_prompt += f"管理员备注: {admin_note}\n"

    user_prompt += "\n请生成跟进消息（JSON格式）:"

    try:
        resp = client.chat.completions.create(
            model=Config.DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=600,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content.strip()
        result = json.loads(raw)

        message = result.get("message", "")
        recommended_files = result.get("recommended_files", [])

        # Validate recommended files exist
        valid_files = []
        for fname in recommended_files:
            fpath = os.path.join(_MATERIALS_DIR, fname)
            if os.path.isfile(fpath):
                valid_files.append(fname)
            else:
                logger.warning(f"Recommended file not found: {fname}")

        return {
            "message": message,
            "recommended_files": valid_files,
        }
    except Exception as e:
        logger.error(f"Follow-up generation failed: {e}")
        return {}
