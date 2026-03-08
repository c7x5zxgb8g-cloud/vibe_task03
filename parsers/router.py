from __future__ import annotations

import os

from config import Config
from .base import ParsedDocument
from utils.logger import get_logger

logger = get_logger(__name__)


def parse_file(file_path: str) -> ParsedDocument:
    """Auto-detect file type and route to the appropriate parser."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext in Config.SUPPORTED_EXTENSIONS["pdf"]:
        from .pdf_parser import parse_pdf

        return parse_pdf(file_path)

    if ext in Config.SUPPORTED_EXTENSIONS["pptx"]:
        from .pptx_parser import parse_pptx

        return parse_pptx(file_path)

    if ext in Config.SUPPORTED_EXTENSIONS["docx"]:
        from .docx_parser import parse_docx

        return parse_docx(file_path)

    if ext in Config.SUPPORTED_EXTENSIONS["excel"]:
        from .excel_parser import parse_excel

        return parse_excel(file_path)

    if ext in Config.SUPPORTED_EXTENSIONS["image"]:
        from .image_parser import parse_image

        return parse_image(file_path)

    if ext in Config.SUPPORTED_EXTENSIONS["txt"]:
        from .txt_parser import parse_txt

        return parse_txt(file_path)

    raise ValueError(
        f"Unsupported file type: {ext}. "
        f"Supported: {', '.join(sum(Config.SUPPORTED_EXTENSIONS.values(), []))}"
    )


def get_all_files(input_path: str) -> list[str]:
    """Recursively collect all supported files from a directory or return single file."""
    all_exts = set(sum(Config.SUPPORTED_EXTENSIONS.values(), []))

    if os.path.isfile(input_path):
        ext = os.path.splitext(input_path)[1].lower()
        if ext in all_exts:
            return [input_path]
        logger.warning(f"Skipping unsupported file: {input_path}")
        return []

    if os.path.isdir(input_path):
        files = []
        for root, _, filenames in os.walk(input_path):
            for fname in sorted(filenames):
                ext = os.path.splitext(fname)[1].lower()
                if ext in all_exts:
                    files.append(os.path.join(root, fname))
        logger.info(f"Found {len(files)} supported files in {input_path}")
        return files

    raise FileNotFoundError(f"Path not found: {input_path}")
