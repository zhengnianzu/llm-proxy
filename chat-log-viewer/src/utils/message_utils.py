"""
utils/message_utils.py — 消息解析通用工具

核心逻辑已迁移至项目根目录 utils/message_common.py，
本文件作为 thin wrapper 保持接口兼容。
"""

import json
import sys
from pathlib import Path
from typing import Any, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from utils.message_common import (
    count_user_messages,
    extract_messages,
    get_first_user_text,
    get_text_from_content,
    parse_response,
    parse_streaming_response,
)

__all__ = [
    "extract_messages",
    "count_user_messages",
    "get_first_user_text",
    "get_text_from_content",
    "parse_streaming_response",
    "parse_response",
    "load_json",
]


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
