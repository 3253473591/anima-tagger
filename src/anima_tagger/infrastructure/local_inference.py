from __future__ import annotations

import base64
import json
import mimetypes
import threading
from pathlib import Path

from anima_tagger.domain.errors import LocalInferenceError, ModelLoadError


class LocalInferenceAdapter:
    def __init__(self, model_path: Path, mmproj_path: Path, *, n_ctx: int = 8192, n_gpu_layers: int = -1, verbose: bool = False):
        self.model_path = model_path
        self.mmproj_path = mmproj_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.verbose = verbose
        self._model = None
        self._text_model = None
        self._lock = threading.Lock()

    def load(self) -> None:
        if self._model is not None:
            return
        if not self.model_path.is_file() or not self.mmproj_path.is_file():
            raise ModelLoadError("主模型或 mmproj 文件不存在")
        try:
            from llama_cpp import Llama
            from llama_cpp.llama_chat_format import Gemma4ChatHandler
            handler = Gemma4ChatHandler(clip_model_path=str(self.mmproj_path), verbose=self.verbose, use_gpu=self.n_gpu_layers != 0)
            self._model = Llama(model_path=str(self.model_path), chat_handler=handler, n_ctx=self.n_ctx, n_batch=512, n_gpu_layers=self.n_gpu_layers, flash_attn=True, verbose=self.verbose)
        except Exception as exc:
            raise ModelLoadError(str(exc)) from exc

    @staticmethod
    def _data_uri(image: Path) -> str:
        mime = mimetypes.guess_type(image.name)[0] or "image/png"
        return f"data:{mime};base64,{base64.b64encode(image.read_bytes()).decode('ascii')}"

    def caption(self, image: Path, prompt: str, *, max_tokens: int = 768) -> str:
        self.load()
        try:
            with self._lock:
                response = self._model.create_chat_completion(
                    messages=[{"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": self._data_uri(image)}},
                    ]}], temperature=0.15, top_p=0.9, max_tokens=max_tokens,
                )
            content = response["choices"][0]["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise ValueError("模型返回空内容")
            return content.strip().removeprefix("```text").removesuffix("```").strip()
        except Exception as exc:
            raise LocalInferenceError(str(exc)) from exc

    def translate_tags(self, tags: list[str], *, max_tokens: int = 512) -> dict[str, str]:
        """Translate editor tags with the base LLM only; no image or mmproj is loaded."""
        unique_tags = list(dict.fromkeys(tag.strip() for tag in tags if tag.strip()))
        if not unique_tags:
            return {}
        self._load_text_model()
        prompt = (
            "Translate these Danbooru-style English tags into concise Simplified Chinese. "
            "Return only a JSON object mapping each original English tag to its Chinese translation. "
            "Keep untranslatable trigger words unchanged. Do not add or omit tags.\n\n"
            + json.dumps(unique_tags, ensure_ascii=False)
        )
        try:
            with self._lock:
                response = self._text_model.create_chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=max_tokens,
                )
            content = response["choices"][0]["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise ValueError("model returned empty content")
            payload = json.loads(content.strip().removeprefix("```json").removesuffix("```").strip())
            if not isinstance(payload, dict):
                raise ValueError("model did not return a JSON object")
            wanted = {tag.casefold(): tag for tag in unique_tags}
            return {
                wanted[key.casefold()]: value.strip()
                for key, value in payload.items()
                if isinstance(key, str)
                and key.casefold() in wanted
                and isinstance(value, str)
                and value.strip()
            }
        except Exception as exc:
            raise LocalInferenceError(str(exc)) from exc

    def _load_text_model(self) -> None:
        if self._text_model is not None:
            return
        if not self.model_path.is_file():
            raise ModelLoadError("main model file does not exist")
        try:
            from llama_cpp import Llama
            self._text_model = Llama(
                model_path=str(self.model_path),
                n_ctx=self.n_ctx,
                n_batch=512,
                n_gpu_layers=self.n_gpu_layers,
                flash_attn=True,
                verbose=self.verbose,
            )
        except Exception as exc:
            raise ModelLoadError(str(exc)) from exc
