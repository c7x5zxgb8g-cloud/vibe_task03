import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # OpenAI
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")

    # ChromaDB
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "securities_kb")

    # Processing
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "512"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "64"))
    LABELING_BATCH_SIZE: int = int(os.getenv("LABELING_BATCH_SIZE", "5"))

    # Tesseract
    TESSERACT_CMD: str = os.getenv("TESSERACT_CMD", "tesseract")

    # DeepSeek API (OpenAI-compatible)
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_BASE_URL: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    DEEPSEEK_MODEL: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    # Feishu Webhook
    FEISHU_WEBHOOK_URL: str = os.getenv("FEISHU_WEBHOOK_URL", "")

    # Admin
    ADMIN_BASE_URL: str = os.getenv("ADMIN_BASE_URL", "http://localhost:8000")
    ADMIN_PASSWORD: str = os.getenv("ADMIN_PASSWORD", "admin123")

    # Webhook secret (optional, for verifying robot requests)
    WEBHOOK_SECRET: str = os.getenv("WEBHOOK_SECRET", "")

    # Supported file extensions
    SUPPORTED_EXTENSIONS: dict = {
        "pdf": [".pdf"],
        "pptx": [".pptx", ".ppt"],
        "docx": [".docx", ".doc"],
        "excel": [".xlsx", ".xls", ".csv"],
        "image": [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"],
        "txt": [".txt"],
    }
