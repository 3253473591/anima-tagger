from __future__ import annotations

from pathlib import Path

from anima_tagger.domain.models import ImageAsset, LoraMode
from anima_tagger.domain.validation import validate_caption
from anima_tagger.infrastructure.files import atomic_write, final_path
from anima_tagger.infrastructure.repository import Repository


class ReviewService:
    def __init__(self, repository: Repository):
        self.repository = repository

    def export(self, asset: ImageAsset, trigger_word: str, mode: LoraMode, *, overwrite: bool = False, skip_validation: bool = False) -> Path:
        rows = self.repository.tags(asset.image_id)
        enabled = [row for row in rows if row["enabled"]]
        tags: list[str] = []
        natural: list[str] = []
        seen: set[str] = set()
        for row in enabled:
            value = row["en"].strip()
            if row["is_natural_language"]:
                natural.append(value)
            elif value.casefold() not in seen:
                tags.append(value)
                seen.add(value.casefold())
        # The trigger must be first even when both review sources contain it.
        tags = [trigger_word] + [tag for tag in tags if tag != trigger_word]
        caption = ", ".join(tags)
        if mode is LoraMode.STYLE and natural:
            caption += ". " + natural[0]
        if not skip_validation:
            validate_caption(caption, trigger_word, mode)
        destination = final_path(asset.path)
        if not atomic_write(destination, caption + "\n", overwrite=overwrite):
            raise FileExistsError(f"最终文件已存在: {destination}")
        return destination

