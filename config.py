"""配置：全部从环境变量读，不把 key 写进代码。"""

import os

from dotenv import load_dotenv

load_dotenv()

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"  # deepseek-v4-flash 是同一模型的别名


def get_api_key() -> str:
    key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "没有找到 DEEPSEEK_API_KEY，请在系统环境变量或 .env 里设置"
        )
    return key


def get_base_url() -> str:
    return os.getenv("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL)


def get_model() -> str:
    return os.getenv("DEEPSEEK_MODEL", DEFAULT_MODEL)


def thinking_enabled() -> bool:
    """V4 默认开思考；设 DEEPSEEK_THINKING=1 才打开（思考 token 按输出计费）。"""
    return os.getenv("DEEPSEEK_THINKING", "0") == "1"
