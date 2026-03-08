#!/usr/bin/env python3
"""Rebuild knowledge base: clear existing data and re-ingest from docs/理博基金知识库."""

from __future__ import annotations

import os
import shutil

from config import Config
from pipeline import process_files
from utils.logger import get_logger

logger = get_logger(__name__)

KB_DIR = os.path.join(os.path.dirname(__file__), "docs", "理博基金知识库")
CHROMA_DIR = Config.CHROMA_PERSIST_DIR

# Files to skip (non-document files)
SKIP_EXTENSIONS = {".zip", ".rar", ".7z"}


def rebuild():
    """Clear ChromaDB and re-process all documents in the knowledge base directory."""
    # Step 1: Clear existing ChromaDB data
    if os.path.exists(CHROMA_DIR):
        print(f"正在清除旧的知识库数据: {CHROMA_DIR}")
        shutil.rmtree(CHROMA_DIR)
        print("已清除。")

    # Step 2: Verify KB directory exists
    if not os.path.isdir(KB_DIR):
        print(f"错误: 知识库目录不存在: {KB_DIR}")
        return

    # List all files
    all_files = []
    for f in os.listdir(KB_DIR):
        fpath = os.path.join(KB_DIR, f)
        if os.path.isfile(fpath):
            ext = os.path.splitext(f)[1].lower()
            if ext in SKIP_EXTENSIONS:
                print(f"跳过: {f} (不支持的格式)")
                continue
            all_files.append(f)

    print(f"\n共发现 {len(all_files)} 个文件待处理")
    print("=" * 60)

    # Step 3: Process all files in the KB directory
    report = process_files(KB_DIR, enable_labeling=True)

    # Step 4: Print report
    print(report.summary())


if __name__ == "__main__":
    rebuild()
