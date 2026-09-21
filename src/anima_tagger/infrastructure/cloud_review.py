from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path

from anima_tagger.domain.errors import CloudReviewError


class CloudReviewAdapter:
    def __init__(self, api_key: str, model: str, base_url: str, proxy_url: str | None = None):
        import httpx
        from openai import OpenAI
        client = httpx.Client(proxy=proxy_url, timeout=120) if proxy_url else None
        self.client = OpenAI(api_key=api_key, base_url=base_url.rstrip("/") + "/", http_client=client)
        self.model = model

    @classmethod
    def list_models(cls, api_key: str, base_url: str, proxy_url: str | None = None) -> list[str]:
        """Return model IDs exposed by an OpenAI-compatible provider."""
        adapter = cls(api_key, "", base_url, proxy_url)
        try:
            return sorted({item.id for item in adapter.client.models.list().data if item.id})
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            if status in {401, 403}:
                raise CloudReviewError("API Key 无效或无权读取模型列表", code="AUTH_FAILED", retryable=False) from exc
            raise CloudReviewError(f"无法获取云端模型列表: {exc}", code="MODELS_LIST_FAILED") from exc
        finally:
            adapter.client.close()

    def review(self, image: Path, prompt: str, correction: str = "") -> str:
        mime = mimetypes.guess_type(image.name)[0] or "image/png"
        uri = f"data:{mime};base64,{base64.b64encode(image.read_bytes()).decode('ascii')}"
        schema_instruction = """只输出一个 JSON 对象，不使用 Markdown。必须包含：
schema_version=1；score（0-10 整数）；comment；final_caption；
first_caption_validity 数组（每项包含 tag/valid/suggestion）；
first_review_status（pass 或 needs_revision）。comment 和 suggestion 必须使用简体中文；
没有修改建议时 suggestion 使用 null。"""
        try:
            response = self.client.chat.completions.create(model=self.model, messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt + "\n\n" + schema_instruction + correction},
                {"type": "image_url", "image_url": {"url": uri}},
            # Low temperature keeps tagging deterministic. Providers that reject a custom
            # value accept only their default of 1; raise this if a provider errors on it.
            ]}], temperature=0.1, response_format={"type": "json_object"})
            return response.choices[0].message.content or ""
        except Exception as exc:
            detail = self._error_detail(exc)
            status = getattr(exc, "status_code", None)
            # Gemini rejects prohibited prompts/images before generation; retrying unchanged input cannot succeed.
            if "prompt_blocked" in detail.casefold() or "prohibited_content" in detail.casefold():
                raise CloudReviewError(detail, code="CONTENT_BLOCKED", retryable=False) from exc
            if status in {401, 403}:
                raise CloudReviewError(detail, code="AUTH_FAILED", retryable=False) from exc
            if status == 429:
                raise CloudReviewError(detail, code="RATE_LIMITED") from exc
            raise CloudReviewError(detail, code="NETWORK_OR_API_ERROR") from exc

    @staticmethod
    def _error_detail(exc: Exception) -> str:
        """Keep the provider's useful error body without exposing credentials."""
        status = getattr(exc, "status_code", None)
        body = getattr(exc, "body", None)
        if body is not None:
            try:
                body_text = json.dumps(body, ensure_ascii=False)
            except (TypeError, ValueError):
                body_text = str(body)
        else:
            body_text = str(exc)
        prefix = f"HTTP {status}: " if status else ""
        return (prefix + body_text).replace("Authorization", "[redacted header]")[:1800]
