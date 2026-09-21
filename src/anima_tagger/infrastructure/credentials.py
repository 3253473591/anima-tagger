from __future__ import annotations

import os


SERVICE = "AnimaTagger"


def set_api_key(reference: str, value: str) -> None:
    if not value.strip():
        return
    try:
        import keyring
    except ImportError as exc:
        raise RuntimeError("请安装 keyring 后再保存 API Key") from exc
    keyring.set_password(SERVICE, reference, value.strip())


def get_api_key(reference: str) -> str | None:
    try:
        import keyring
        value = keyring.get_password(SERVICE, reference)
    except (ImportError, Exception):
        value = None
    return value or os.getenv("ANIMA_TAGGER_API_KEY")

