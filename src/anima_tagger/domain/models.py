from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any


class LoraMode(StrEnum):
    CHARACTER = "character"
    STYLE = "style"


class Stage(StrEnum):
    FIRST_REVIEW = "first_review"
    SECOND_REVIEW = "second_review"


class JobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    SUCCEEDED = "succeeded"
    RETRY_WAIT = "retry_wait"
    FAILED = "failed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"


class TagSource(StrEnum):
    FIRST_REVIEW = "first_review"
    SECOND_REVIEW = "second_review"
    MANUAL = "manual"


@dataclass(frozen=True)
class RunConfig:
    folder: Path
    lora_mode: LoraMode
    trigger_word: str
    gemma4_gguf_path: Path
    mmproj_path: Path
    mtp_path: Path | None = None
    first_review_concurrency: int = 1
    second_review_concurrency: int = 2
    gemini_api_key_ref: str = "default"
    proxy_url: str | None = None
    cloud_model: str = "gemini-3.5-flash"
    cloud_base_url: str = "https://api.example.com/v1"  # 占位示例：指向自有的 OpenAI 兼容端点
    thumbnail_columns: int = 10
    api_retry_count: int = 5
    resume_enabled: bool = True
    first_caption_rule_mode: str = "filter"
    n_ctx: int = 8192
    n_gpu_layers: int = -1

    def __post_init__(self) -> None:
        if not self.folder.is_dir():
            raise ValueError(f"图片文件夹不存在: {self.folder}")
        if not self.trigger_word.strip() or "," in self.trigger_word:
            raise ValueError("触发词不能为空且不能包含逗号")
        if self.first_review_concurrency < 1 or self.second_review_concurrency < 1:
            raise ValueError("并发数必须至少为 1")
        if not 2 <= self.thumbnail_columns <= 20:
            raise ValueError("每行缩略图数量必须在 2 到 20 之间")

        if not 1 <= self.api_retry_count <= 20:
            raise ValueError("API retry count must be between 1 and 20")
        if self.first_caption_rule_mode not in {"filter", "second_review"}:
            raise ValueError("一审规则处理模式无效")

    def safe_snapshot(self) -> dict[str, Any]:
        data = asdict(self)
        data["folder"] = str(self.folder)
        data["gemma4_gguf_path"] = str(self.gemma4_gguf_path)
        data["mmproj_path"] = str(self.mmproj_path)
        data["mtp_path"] = str(self.mtp_path) if self.mtp_path else None
        data["lora_mode"] = self.lora_mode.value
        return data


@dataclass(frozen=True)
class ImageAsset:
    image_id: str
    folder_id: str
    absolute_path: str
    relative_path: str
    stem: str
    extension: str
    size_bytes: int
    modified_ns: int
    content_fingerprint: str

    @property
    def path(self) -> Path:
        return Path(self.absolute_path)


@dataclass
class TagRow:
    tag_id: str
    image_id: str
    source: TagSource
    en: str
    cn: str | None = None
    suggestion: str | None = None
    enabled: bool = True
    position: int = 0
    is_natural_language: bool = False
    manually_overridden: bool = False
