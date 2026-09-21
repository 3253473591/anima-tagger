from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from importlib.resources import files
from typing import Iterable

from .errors import CaptionValidationError
from .models import LoraMode


@dataclass(frozen=True)
class ParsedCaption:
    tags: list[str]
    natural_language: str | None = None


def _resource_terms(name: str) -> set[str]:
    text = files("anima_tagger.resources").joinpath(name).read_text(encoding="utf-8")
    return {line.strip().casefold() for line in text.splitlines() if line.strip() and not line.startswith("#")}


def parse_caption(caption: str, mode: LoraMode) -> ParsedCaption:
    value = unicodedata.normalize("NFC", caption).strip()
    natural = None
    tag_part = value
    if mode is LoraMode.STYLE and ". " in value:
        tag_part, natural = value.split(". ", 1)
        natural = natural.strip()
    tags = [item.strip() for item in tag_part.split(",") if item.strip()]
    return ParsedCaption(tags, natural)


def validate_first_caption(caption: str, trigger_word: str, mode: LoraMode) -> ParsedCaption:
    """Validate only requirements that make an initial caption unusable as review input."""
    issues: list[str] = []
    if not caption.strip():
        issues.append("caption 为空")
    if "\n" in caption or "\r" in caption:
        issues.append("caption 必须为单行")
    parsed = parse_caption(caption, mode)
    if not parsed.tags or parsed.tags[0] != trigger_word:
        issues.append(f"caption 必须以 {trigger_word}, 开头")
    if mode is LoraMode.CHARACTER and parsed.natural_language:
        issues.append("角色模式不允许自然语言")
    if parsed.natural_language and len(re.findall(r"[.!?](?:\s|$)", parsed.natural_language)) > 1:
        issues.append("自然语言最多一句")
    if issues:
        raise CaptionValidationError(issues)
    return parsed


def normalize_first_caption(caption: str, trigger_word: str, mode: LoraMode) -> str:
    """Apply mechanical tag rules before writing an initial caption when configured."""
    parsed = validate_first_caption(caption, trigger_word, mode)
    forbidden = _resource_terms("forbidden_common.txt")
    forbidden |= _resource_terms("forbidden_character.txt" if mode is LoraMode.CHARACTER else "forbidden_style.txt")
    tags = [trigger_word]
    seen = {trigger_word.casefold()}
    for raw_tag in parsed.tags[1:]:
        tag = unicodedata.normalize("NFC", raw_tag).strip().lower()
        normalized = tag.casefold()
        if (
            not tag
            or normalized in seen
            or normalized in forbidden
            or re.search(r"[#()\[\]{}]|\b\d+(?:\.\d+)?::?", tag)
        ):
            continue
        seen.add(normalized)
        tags.append(tag)
    output = ", ".join(tags)
    if mode is LoraMode.STYLE and parsed.natural_language:
        output += ". " + parsed.natural_language
    return output


def validate_caption(caption: str, trigger_word: str, mode: LoraMode) -> ParsedCaption:
    issues: list[str] = []
    if not caption.strip():
        raise CaptionValidationError(["caption 为空"])
    if "\n" in caption or "\r" in caption:
        issues.append("caption 必须为单行")
    parsed = parse_caption(caption, mode)
    if not parsed.tags or parsed.tags[0] != trigger_word:
        issues.append(f"caption 必须以 {trigger_word}, 开头")
    normalized = [unicodedata.normalize("NFC", tag).casefold() for tag in parsed.tags]
    if len(normalized) != len(set(normalized)):
        issues.append("包含重复标签")
    for tag in parsed.tags[1:]:
        if tag != tag.lower():
            issues.append(f"标签必须小写: {tag}")
            break
    if re.search(r"[#()\[\]{}]|\b\d+(?:\.\d+)?::?", ", ".join(parsed.tags[1:])):
        issues.append("包含井号、括号或权重语法")
    forbidden = _resource_terms("forbidden_common.txt")
    forbidden |= _resource_terms("forbidden_character.txt" if mode is LoraMode.CHARACTER else "forbidden_style.txt")
    hits = sorted({tag for tag in normalized[1:] if tag in forbidden})
    if hits:
        issues.append("包含禁用词: " + ", ".join(hits))
    if mode is LoraMode.CHARACTER and parsed.natural_language:
        issues.append("角色模式不允许自然语言")
    if parsed.natural_language and len(re.findall(r"[.!?](?:\s|$)", parsed.natural_language)) > 1:
        issues.append("自然语言最多一句")
    if issues:
        raise CaptionValidationError(issues)
    return parsed


def validate_second_review(
    payload: str | dict,
    trigger_word: str,
    mode: LoraMode,
    *,
    require_translations: bool = False,
    required_translation_tags: Iterable[str] = (),
) -> dict:
    try:
        data = json.loads(payload) if isinstance(payload, str) else payload
    except json.JSONDecodeError as exc:
        raise CaptionValidationError([f"二审不是有效 JSON: {exc.msg}"]) from exc
    required = {"schema_version", "score", "comment", "final_caption", "first_caption_validity", "first_review_status"}
    missing = required - set(data) if isinstance(data, dict) else required
    issues = ["二审缺少字段: " + ", ".join(sorted(missing))] if missing else []
    if not isinstance(data, dict):
        issues.append("二审根节点必须是 JSON 对象")
    else:
        score = data.get("score")
        status = data.get("first_review_status")
        if type(score) is not int or not 0 <= score <= 10:
            issues.append("score 必须是 0 到 10 的整数")
        if status not in {"pass", "needs_revision"}:
            issues.append("first_review_status 非法")
        elif type(score) is int and ((score >= 9) != (status == "pass")):
            issues.append("评分与 first_review_status 不一致")
        if not isinstance(data.get("comment"), str):
            issues.append("comment 必须是字符串")
        validity = data.get("first_caption_validity")
        if not isinstance(validity, list) or any(not isinstance(x, dict) or type(x.get("valid")) is not bool or not isinstance(x.get("tag"), str) for x in validity):
            issues.append("first_caption_validity 格式非法")
        elif len({x["tag"].casefold() for x in validity}) != len(validity):
            issues.append("一审标签存在重复有效性结论")
        translations = data.get("tag_translations")
        if require_translations and translations is None:
            issues.append("二审缺少 tag_translations")
        elif translations is not None:
            if not isinstance(translations, list) or any(
                not isinstance(x, dict)
                or not isinstance(x.get("tag"), str)
                or not isinstance(x.get("cn"), str)
                or not x["cn"].strip()
                for x in translations
            ):
                issues.append("tag_translations 格式非法")
            elif len({x["tag"].casefold() for x in translations}) != len(translations):
                issues.append("tag_translations 存在重复标签")
            elif require_translations:
                translated = {x["tag"].casefold() for x in translations}
                required_tags = {
                    tag.casefold()
                    for tag in parse_caption(data.get("final_caption", ""), mode).tags
                }
                required_tags.update(tag.casefold() for tag in required_translation_tags)
                if isinstance(validity, list):
                    required_tags.update(
                        x["tag"].casefold()
                        for x in validity
                        if isinstance(x, dict) and isinstance(x.get("tag"), str)
                    )
                missing_translations = sorted(required_tags - translated)
                if missing_translations:
                    issues.append("tag_translations 缺少标签: " + ", ".join(missing_translations))
        try:
            validate_caption(data.get("final_caption", ""), trigger_word, mode)
        except CaptionValidationError as exc:
            issues.extend(exc.issues)
    if issues:
        raise CaptionValidationError(issues)
    return data
