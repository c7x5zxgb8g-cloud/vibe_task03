"""Plain-text (.txt) file parser."""

from __future__ import annotations

import os

from .base import ParsedDocument, ParsedPage
from utils.logger import get_logger

logger = get_logger(__name__)


def parse_txt(file_path: str) -> ParsedDocument:
    """Parse a plain text file into a ParsedDocument."""
    file_name = os.path.basename(file_path)
    logger.info(f"Parsing TXT: {file_name}")

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    # Treat the entire file as a single page
    page = ParsedPage(page_number=1, text=content.strip())

    return ParsedDocument(
        file_name=file_name,
        file_path=file_path,
        file_type="txt",
        pages=[page],
        metadata={"char_count": len(content)},
    )
