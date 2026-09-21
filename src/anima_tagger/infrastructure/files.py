from __future__ import annotations

import hashlib
import os
import uuid
from pathlib import Path

from anima_tagger.domain.errors import ImageChangedError
from anima_tagger.domain.models import ImageAsset

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def fingerprint(path: Path) -> tuple[int, int, str]:
    stat = path.stat()
    raw = f"{path.resolve()}\0{stat.st_size}\0{stat.st_mtime_ns}".encode("utf-8")
    return stat.st_size, stat.st_mtime_ns, hashlib.sha256(raw).hexdigest()


def scan_images(root: Path, folder_id: str) -> list[ImageAsset]:
    root = root.resolve()
    assets: list[ImageAsset] = []
    for path in sorted(root.rglob("*"), key=lambda item: str(item).casefold()):
        if not path.is_file() or path.is_symlink() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            continue
        size, modified, value = fingerprint(resolved)
        relative = resolved.relative_to(root).as_posix()
        image_id = hashlib.sha256(f"{folder_id}\0{relative}".encode()).hexdigest()
        assets.append(ImageAsset(image_id, folder_id, str(resolved), relative, resolved.stem, resolved.suffix.lower(), size, modified, value))
    return assets


def stage_path(image: Path, stage_number: int) -> Path:
    return image.with_name(f"{image.stem}-{stage_number}.txt")


def final_path(image: Path) -> Path:
    return image.with_suffix(".txt")


def is_nonempty(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0 and bool(path.read_text(encoding="utf-8").strip())
    except (OSError, UnicodeError):
        return False


def ensure_unchanged(asset: ImageAsset) -> None:
    _, _, current = fingerprint(asset.path)
    if current != asset.content_fingerprint:
        raise ImageChangedError(f"图片在任务期间发生变化: {asset.path}")


def atomic_write(path: Path, content: str, *, overwrite: bool = False) -> bool:
    if not overwrite and is_nonempty(path):
        return False
    temp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("x", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        return True
    finally:
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass

